package engines

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// anthropicSSE joins named-event SSE frames the way the Anthropic API
// wires them: `event: <type>` line, `data: <json>` line, blank line.
func anthropicSSE(frames ...[2]string) string {
	var b strings.Builder
	for _, f := range frames {
		b.WriteString("event: " + f[0] + "\n")
		b.WriteString("data: " + f[1] + "\n\n")
	}
	return b.String()
}

// TestAnthropicStream_HappyPath covers the full Stream() path against a
// mock /v1/messages server: auth headers (x-api-key, anthropic-version),
// request translation (system → top-level field, max_tokens default),
// text deltas, the terminal done event, and usage — including the
// cache-read adjustment (TokensIn = input_tokens + cache_read).
func TestAnthropicStream_HappyPath(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := anthropicSSE(
		[2]string{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":40,"cache_creation_input_tokens":15}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, "}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world."}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewAnthropic(AnthropicConfig{URL: srv.URL, APIKey: "test-key", Model: "claude-haiku-4-5"})
	msgs := []LLMMessage{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "Say hello."},
	}
	events, err := runLLMStream(t, context.Background(), eng, msgs, nil)
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	if got := joinTextDeltas(events); got != "Hello, world." {
		t.Errorf("assembled text = %q, want %q", got, "Hello, world.")
	}
	done := findEvent(events, "done")
	if done == nil || done.FinishReason != "end_turn" {
		t.Errorf("done = %+v, want FinishReason end_turn", done)
	}
	if events[len(events)-1].Type != "done" {
		t.Errorf("last event type = %q, want done", events[len(events)-1].Type)
	}

	// TokensIn is input_tokens PLUS cache_read (Anthropic excludes cache
	// reads from input_tokens; cadence normalizes to the OpenAI total).
	// TokensOut comes from message_delta's cumulative count, not
	// message_start's placeholder 1.
	usage := eng.(UsageReporter).Usage()
	if usage.TokensIn != 140 {
		t.Errorf("TokensIn = %d, want 140 (100 input + 40 cache read)", usage.TokensIn)
	}
	if usage.TokensOut != 12 {
		t.Errorf("TokensOut = %d, want 12", usage.TokensOut)
	}
	if usage.CachedTokens != 40 || usage.CacheCreationTokens != 15 {
		t.Errorf("cache usage = read %d / write %d, want 40 / 15", usage.CachedTokens, usage.CacheCreationTokens)
	}

	// Request-side assertions.
	req := <-reqCh
	if req.path != "/v1/messages" {
		t.Errorf("path = %q, want /v1/messages", req.path)
	}
	if got := req.header.Get("x-api-key"); got != "test-key" {
		t.Errorf("x-api-key = %q, want test-key", got)
	}
	if got := req.header.Get("anthropic-version"); got != anthropicAPIVersion {
		t.Errorf("anthropic-version = %q, want %q", got, anthropicAPIVersion)
	}
	if req.body["model"] != "claude-haiku-4-5" {
		t.Errorf("model = %v", req.body["model"])
	}
	if req.body["max_tokens"] != float64(4096) {
		t.Errorf("max_tokens = %v, want 4096 default", req.body["max_tokens"])
	}
	if req.body["stream"] != true {
		t.Errorf("stream = %v, want true", req.body["stream"])
	}
	// system messages become the top-level field, not a message.
	if req.body["system"] != "You are terse." {
		t.Errorf("system = %v, want top-level system text", req.body["system"])
	}
	bodyMsgs, _ := req.body["messages"].([]interface{})
	if len(bodyMsgs) != 1 {
		t.Fatalf("messages len = %d, want 1 (system hoisted out)", len(bodyMsgs))
	}
	first, _ := bodyMsgs[0].(map[string]interface{})
	if first["role"] != "user" || first["content"] != "Say hello." {
		t.Errorf("messages[0] = %v", first)
	}
}

// TestAnthropicStream_ToolCalls verifies tool_use content blocks:
// content_block_start carries id + name, input_json_delta fragments
// accumulate into the arguments string, and a second tool_use block
// with no deltas defaults to "{}". Also checks the input_schema tools
// serialization on the request.
func TestAnthropicStream_ToolCalls(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := anthropicSSE(
		[2]string{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":50,"output_tokens":1}}}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"get_weather"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`},
		[2]string{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		[2]string{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_b","name":"get_time"}}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		[2]string{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}`},
		[2]string{"message_stop", `{"type":"message_stop"}`},
	)
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewAnthropic(AnthropicConfig{URL: srv.URL, APIKey: "k", Model: "claude-haiku-4-5"})
	tools := []ToolDef{{
		Name:        "get_weather",
		Description: "Current weather for a city",
		Parameters:  map[string]interface{}{"type": "object"},
	}}
	events, err := runLLMStream(t, context.Background(), eng, []LLMMessage{{Role: "user", Content: "weather?"}}, tools)
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	tc := findEvent(events, "tool_calls")
	if tc == nil {
		t.Fatal("no tool_calls event emitted")
	}
	if len(tc.ToolCalls) != 2 {
		t.Fatalf("tool calls = %d, want 2", len(tc.ToolCalls))
	}
	if tc.ToolCalls[0].ID != "toolu_a" || tc.ToolCalls[0].Name != "get_weather" ||
		tc.ToolCalls[0].Arguments != `{"city":"Paris"}` {
		t.Errorf("call[0] = %+v", tc.ToolCalls[0])
	}
	// No input_json_delta → arguments default to "{}", never "".
	if tc.ToolCalls[1].ID != "toolu_b" || tc.ToolCalls[1].Arguments != "{}" {
		t.Errorf("call[1] = %+v, want empty-args default {}", tc.ToolCalls[1])
	}
	if done := findEvent(events, "done"); done == nil || done.FinishReason != "tool_use" {
		t.Errorf("done = %+v, want FinishReason tool_use", done)
	}

	// tools serialization: Anthropic uses name/description/input_schema
	// (no {type:function} wrapper).
	req := <-reqCh
	bodyTools, _ := req.body["tools"].([]interface{})
	if len(bodyTools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(bodyTools))
	}
	tool0, _ := bodyTools[0].(map[string]interface{})
	if tool0["name"] != "get_weather" {
		t.Errorf("tools[0].name = %v", tool0["name"])
	}
	if _, ok := tool0["input_schema"].(map[string]interface{}); !ok {
		t.Errorf("tools[0].input_schema missing, got %v", tool0)
	}
}

// TestBuildAnthropicRequest_Translation exercises the non-1:1 message
// translations directly: assistant-with-tools → tool_use blocks, tool
// role → user message with a tool_result block.
func TestBuildAnthropicRequest_Translation(t *testing.T) {
	msgs := []LLMMessage{
		{Role: "assistant", Content: "Checking.", ToolCalls: []ToolCall{
			{ID: "toolu_a", Name: "get_weather", Arguments: `{"city":"Paris"}`},
		}},
		{Role: "tool", Name: "get_weather", ToolCallID: "toolu_a", Content: `{"temp":18}`},
	}
	req := buildAnthropicRequest(AnthropicConfig{Model: "m", MaxTokens: 4096}, msgs, nil)

	out, _ := req["messages"].([]map[string]interface{})
	if len(out) != 2 {
		t.Fatalf("messages len = %d, want 2", len(out))
	}

	// Assistant turn: text block first, then the tool_use block with
	// arguments decoded from JSON string to object.
	blocks, _ := out[0]["content"].([]map[string]interface{})
	if len(blocks) != 2 {
		t.Fatalf("assistant blocks = %d, want 2 (text + tool_use)", len(blocks))
	}
	if blocks[0]["type"] != "text" || blocks[0]["text"] != "Checking." {
		t.Errorf("blocks[0] = %v", blocks[0])
	}
	if blocks[1]["type"] != "tool_use" || blocks[1]["id"] != "toolu_a" || blocks[1]["name"] != "get_weather" {
		t.Errorf("blocks[1] = %v", blocks[1])
	}
	input, _ := blocks[1]["input"].(map[string]interface{})
	if input["city"] != "Paris" {
		t.Errorf("tool_use input = %v, want decoded object", blocks[1]["input"])
	}

	// Tool result: Anthropic has no "tool" role — it becomes a user
	// message carrying a tool_result block keyed by tool_use_id.
	if out[1]["role"] != "user" {
		t.Errorf("tool message role = %v, want user", out[1]["role"])
	}
	rblocks, _ := out[1]["content"].([]map[string]interface{})
	if len(rblocks) != 1 || rblocks[0]["type"] != "tool_result" || rblocks[0]["tool_use_id"] != "toolu_a" {
		t.Errorf("tool_result blocks = %v", rblocks)
	}
}

// TestAnthropicStream_ContextCancel: server drips one delta then holds
// the connection; ctx cancellation must unblock Stream promptly.
func TestAnthropicStream_ContextCancel(t *testing.T) {
	srv := httptest.NewServer(dripSSEHandler(
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n"))
	defer srv.Close()

	eng := NewAnthropic(AnthropicConfig{URL: srv.URL, APIKey: "k", Model: "m"})
	cancelMidStream(t, eng)
}

// TestAnthropicStream_HTTPError: a 500 with Anthropic's JSON error body
// must surface as a descriptive error plus a terminal done/error event.
func TestAnthropicStream_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"overloaded"}}`))
	}))
	defer srv.Close()

	eng := NewAnthropic(AnthropicConfig{URL: srv.URL, APIKey: "k", Model: "m"})
	events, err := runLLMStream(t, context.Background(), eng, []LLMMessage{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected error on 500, got nil")
	}
	if !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "overloaded") {
		t.Errorf("error = %v, want status + body", err)
	}
	expectTerminalDone(t, events, "error")
}

// TestAnthropicStream_MalformedDataLine: a garbage data line is skipped
// (tolerant parser) and the surrounding events still flow.
func TestAnthropicStream_MalformedDataLine(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"before\"}}\n\n" +
		"data: {not json at all\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" after\"}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewAnthropic(AnthropicConfig{URL: srv.URL, APIKey: "k", Model: "m"})
	events, err := runLLMStream(t, context.Background(), eng, []LLMMessage{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Stream error: %v (malformed lines must be skipped, not fatal)", err)
	}
	if got := joinTextDeltas(events); got != "before after" {
		t.Errorf("assembled text = %q, want %q", got, "before after")
	}
	if done := findEvent(events, "done"); done == nil || done.FinishReason != "end_turn" {
		t.Errorf("done = %+v, want FinishReason end_turn", done)
	}
}
