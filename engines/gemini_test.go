package engines

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestGeminiStream_HappyPath covers the full Stream() path against a
// mock streamGenerateContent server: endpoint shape (model in path,
// alt=sse query), x-goog-api-key auth, systemInstruction hoisting, text
// deltas, the terminal done event, and usageMetadata extraction.
func TestGeminiStream_HappyPath(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello, "}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}`,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"world."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}`,
		``,
	}, "\n\n")
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewGemini(GeminiConfig{URL: srv.URL, APIKey: "test-key", Model: "gemini-2.5-flash"})
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
	if done == nil || done.FinishReason != "STOP" {
		t.Errorf("done = %+v, want FinishReason STOP", done)
	}
	if events[len(events)-1].Type != "done" {
		t.Errorf("last event type = %q, want done", events[len(events)-1].Type)
	}

	// usageMetadata is cumulative per chunk — the final chunk's totals
	// win (overwrite, not sum).
	usage := eng.(UsageReporter).Usage()
	if usage.TokensIn != 10 || usage.TokensOut != 5 {
		t.Errorf("usage = in %d / out %d, want 10 / 5", usage.TokensIn, usage.TokensOut)
	}

	// Request-side assertions.
	req := <-reqCh
	if req.path != "/v1beta/models/gemini-2.5-flash:streamGenerateContent" {
		t.Errorf("path = %q, want model-scoped streamGenerateContent", req.path)
	}
	if req.query.Get("alt") != "sse" {
		t.Errorf("alt = %q, want sse", req.query.Get("alt"))
	}
	if got := req.header.Get("x-goog-api-key"); got != "test-key" {
		t.Errorf("x-goog-api-key = %q, want test-key", got)
	}
	// system messages become the top-level systemInstruction field.
	si, _ := req.body["systemInstruction"].(map[string]interface{})
	siParts, _ := si["parts"].([]interface{})
	if len(siParts) != 1 {
		t.Fatalf("systemInstruction parts = %v", si)
	}
	sp0, _ := siParts[0].(map[string]interface{})
	if sp0["text"] != "You are terse." {
		t.Errorf("systemInstruction text = %v", sp0["text"])
	}
	contents, _ := req.body["contents"].([]interface{})
	if len(contents) != 1 {
		t.Fatalf("contents len = %d, want 1 (system hoisted out)", len(contents))
	}
	c0, _ := contents[0].(map[string]interface{})
	if c0["role"] != "user" {
		t.Errorf("contents[0].role = %v, want user", c0["role"])
	}
}

// TestGeminiStream_FunctionCalls verifies functionCall parts: args
// re-marshalled to a JSON string, the synthesized "gemini_<name>_<idx>"
// id (Gemini supplies none), batching into one tool_calls event, and
// the tools[0].functionDeclarations request envelope.
func TestGeminiStream_FunctionCalls(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris","units":"c"}}}]}}]}`,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_time","args":{}}}]},"finishReason":"STOP"}]}`,
		``,
	}, "\n\n")
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewGemini(GeminiConfig{URL: srv.URL, APIKey: "k", Model: "gemini-2.5-flash"})
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
		t.Fatalf("tool calls = %d, want 2 (batched across chunks)", len(tc.ToolCalls))
	}
	// encoding/json sorts map keys, so the re-marshalled args string is
	// deterministic.
	if tc.ToolCalls[0].ID != "gemini_get_weather_0" || tc.ToolCalls[0].Name != "get_weather" ||
		tc.ToolCalls[0].Arguments != `{"city":"Paris","units":"c"}` {
		t.Errorf("call[0] = %+v", tc.ToolCalls[0])
	}
	if tc.ToolCalls[1].ID != "gemini_get_time_1" || tc.ToolCalls[1].Arguments != "{}" {
		t.Errorf("call[1] = %+v", tc.ToolCalls[1])
	}
	if done := findEvent(events, "done"); done == nil || done.FinishReason != "STOP" {
		t.Errorf("done = %+v, want FinishReason STOP", done)
	}

	// Same function called twice in one turn must still get distinct
	// IDs, or tool_result correlation breaks downstream.
	srv2 := httptest.NewServer(sseHandler(make(chan capturedLLMRequest, 1),
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"ping","args":{}}},{"functionCall":{"name":"ping","args":{}}}]},"finishReason":"STOP"}]}`+"\n\n"))
	defer srv2.Close()
	eng2 := NewGemini(GeminiConfig{URL: srv2.URL, APIKey: "k", Model: "gemini-2.5-flash"})
	events2, err := runLLMStream(t, context.Background(), eng2, []LLMMessage{{Role: "user", Content: "go"}}, nil)
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	tc2 := findEvent(events2, "tool_calls")
	if tc2 == nil || len(tc2.ToolCalls) != 2 {
		t.Fatalf("duplicate-call event = %+v, want 2 tool calls", tc2)
	}
	if tc2.ToolCalls[0].ID == tc2.ToolCalls[1].ID {
		t.Errorf("duplicate function calls share ID %q", tc2.ToolCalls[0].ID)
	}

	// tools serialization: single tools envelope wrapping functionDeclarations.
	req := <-reqCh
	bodyTools, _ := req.body["tools"].([]interface{})
	if len(bodyTools) != 1 {
		t.Fatalf("tools len = %d, want 1 envelope", len(bodyTools))
	}
	env, _ := bodyTools[0].(map[string]interface{})
	decls, _ := env["functionDeclarations"].([]interface{})
	if len(decls) != 1 {
		t.Fatalf("functionDeclarations len = %d, want 1", len(decls))
	}
	d0, _ := decls[0].(map[string]interface{})
	if d0["name"] != "get_weather" || d0["description"] != "Current weather for a city" {
		t.Errorf("functionDeclarations[0] = %v", d0)
	}
}

// TestBuildGeminiRequest_MergesAdjacentUserContents exercises the
// barge-in shape directly: consecutive user messages (no assistant in
// between) must fold into one contents entry so Gemini's strict
// user↔model alternation holds.
func TestBuildGeminiRequest_MergesAdjacentUserContents(t *testing.T) {
	msgs := []LLMMessage{
		{Role: "user", Content: "first"},
		{Role: "user", Content: "second"},
		{Role: "assistant", Content: "reply"},
	}
	req := buildGeminiRequest(GeminiConfig{Model: "m"}, msgs, nil)

	contents, _ := req["contents"].([]map[string]interface{})
	if len(contents) != 2 {
		t.Fatalf("contents len = %d, want 2 (users merged, then model)", len(contents))
	}
	parts, _ := contents[0]["parts"].([]map[string]interface{})
	if len(parts) != 2 || parts[0]["text"] != "first" || parts[1]["text"] != "second" {
		t.Errorf("merged user parts = %v", parts)
	}
	if contents[1]["role"] != "model" {
		t.Errorf("contents[1].role = %v, want model (assistant renamed)", contents[1]["role"])
	}
}

// TestGeminiStream_ContextCancel: server drips one delta then holds the
// connection; ctx cancellation must unblock Stream promptly.
func TestGeminiStream_ContextCancel(t *testing.T) {
	srv := httptest.NewServer(dripSSEHandler(
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}` + "\n\n"))
	defer srv.Close()

	eng := NewGemini(GeminiConfig{URL: srv.URL, APIKey: "k", Model: "m"})
	cancelMidStream(t, eng)
}

// TestGeminiStream_HTTPError: a 429 with Google's JSON error body must
// surface as a descriptive error with no events emitted.
func TestGeminiStream_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED"}}`))
	}))
	defer srv.Close()

	eng := NewGemini(GeminiConfig{URL: srv.URL, APIKey: "k", Model: "m"})
	events, err := runLLMStream(t, context.Background(), eng, []LLMMessage{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected error on 429, got nil")
	}
	if !strings.Contains(err.Error(), "429") || !strings.Contains(err.Error(), "RESOURCE_EXHAUSTED") {
		t.Errorf("error = %v, want status + body", err)
	}
	expectTerminalDone(t, events, "error")
}

// TestGeminiStream_NoModelConfigured: a missing model is rejected
// before any HTTP request, with a terminal done/error event emitted
// and the events channel closed.
func TestGeminiStream_NoModelConfigured(t *testing.T) {
	eng := NewGemini(GeminiConfig{URL: "http://127.0.0.1:0", APIKey: "k"})
	events, err := runLLMStream(t, context.Background(), eng, []LLMMessage{{Role: "user", Content: "hi"}}, nil)
	if err == nil || !strings.Contains(err.Error(), "model not configured") {
		t.Errorf("error = %v, want model-not-configured", err)
	}
	expectTerminalDone(t, events, "error")
}

// TestGeminiStream_MalformedDataLine: a garbage data line is skipped
// (tolerant parser) and the surrounding chunks still flow.
func TestGeminiStream_MalformedDataLine(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"before"}]}}]}`,
		`data: {not json at all`,
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":" after"}]},"finishReason":"STOP"}]}`,
		``,
	}, "\n\n")
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewGemini(GeminiConfig{URL: srv.URL, APIKey: "k", Model: "m"})
	events, err := runLLMStream(t, context.Background(), eng, []LLMMessage{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Stream error: %v (malformed lines must be skipped, not fatal)", err)
	}
	if got := joinTextDeltas(events); got != "before after" {
		t.Errorf("assembled text = %q, want %q", got, "before after")
	}
	if done := findEvent(events, "done"); done == nil || done.FinishReason != "STOP" {
		t.Errorf("done = %+v, want FinishReason STOP", done)
	}
}
