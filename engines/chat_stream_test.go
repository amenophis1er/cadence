package engines

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

// streamTestTimeout bounds every wait in the engine stream tests so a
// wedged engine fails the test instead of hanging the suite.
const streamTestTimeout = 5 * time.Second

// capturedLLMRequest is what the mock SSE servers record about the
// single request an engine sends: enough to assert auth headers and
// body serialization from the test goroutine without racing the
// handler goroutine.
type capturedLLMRequest struct {
	method string
	path   string
	query  url.Values
	header http.Header
	body   map[string]interface{}
}

// sseHandler builds an http.HandlerFunc that captures the request into
// reqCh (buffered, non-blocking) and then writes the given SSE payload
// verbatim. Shared across the chat / anthropic / gemini stream tests.
func sseHandler(reqCh chan<- capturedLLMRequest, sse string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]interface{}
		_ = json.Unmarshal(raw, &body)
		select {
		case reqCh <- capturedLLMRequest{
			method: r.Method,
			path:   r.URL.Path,
			query:  r.URL.Query(),
			header: r.Header.Clone(),
			body:   body,
		}:
		default:
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	}
}

// runLLMStream drives one Stream call to completion, collecting every
// emitted LLMEvent. Returns once the engine closes the events channel
// and Stream itself returns; both waits are deadline-bounded.
func runLLMStream(t *testing.T, ctx context.Context, eng LLMEngine, msgs []LLMMessage, tools []ToolDef) ([]LLMEvent, error) {
	t.Helper()
	out := make(chan LLMEvent, 64)
	errCh := make(chan error, 1)
	go func() { errCh <- eng.Stream(ctx, msgs, tools, out) }()

	var events []LLMEvent
	deadline := time.After(streamTestTimeout)
	for {
		select {
		case ev, ok := <-out:
			if !ok {
				select {
				case err := <-errCh:
					return events, err
				case <-deadline:
					t.Fatal("Stream did not return after closing events channel")
				}
			}
			events = append(events, ev)
		case <-deadline:
			t.Fatal("timed out waiting for stream events")
		}
	}
}

// joinTextDeltas concatenates every text event's delta — the "full
// assistant text" a consumer would have assembled.
func joinTextDeltas(events []LLMEvent) string {
	var b strings.Builder
	for _, ev := range events {
		if ev.Type == "text" {
			b.WriteString(ev.TextDelta)
		}
	}
	return b.String()
}

// findEvent returns the first event of the given type, or nil.
func findEvent(events []LLMEvent, typ string) *LLMEvent {
	for i := range events {
		if events[i].Type == typ {
			return &events[i]
		}
	}
	return nil
}

// dripSSEHandler writes one text-delta line, flushes it, then holds the
// connection open until the client goes away. Used by the cancellation
// tests: the engine must unblock via ctx, not via stream end.
func dripSSEHandler(firstLine string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(firstLine))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}
}

// cancelMidStream runs the shared cancellation scenario: start Stream,
// wait for the first event to prove the connection is live, cancel the
// context, and require Stream to return an error promptly with the
// events channel closed.
func cancelMidStream(t *testing.T, eng LLMEngine) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	out := make(chan LLMEvent, 64)
	errCh := make(chan error, 1)
	go func() { errCh <- eng.Stream(ctx, []LLMMessage{{Role: "user", Content: "hi"}}, nil, out) }()

	select {
	case <-out:
		// first delta arrived — the stream is live; now pull the plug.
	case <-time.After(streamTestTimeout):
		t.Fatal("no event before cancellation window")
	}
	cancel()

	select {
	case err := <-errCh:
		if err == nil {
			t.Error("Stream returned nil error after mid-stream cancellation")
		}
	case <-time.After(streamTestTimeout):
		t.Fatal("Stream did not return after context cancellation")
	}
	// Engine closes `out` on return; drain to confirm (bounded).
	for {
		select {
		case _, ok := <-out:
			if !ok {
				return
			}
		case <-time.After(streamTestTimeout):
			t.Fatal("events channel never closed after cancellation")
		}
	}
}

// TestChatStream_HappyPath covers the full Stream() path against a mock
// OpenAI-compatible server: request serialization (auth header, model,
// messages, stream flags), text deltas in order, the terminal done
// event, and usage accumulation via UsageReporter.
func TestChatStream_HappyPath(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hello, "},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"content":"world."},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":57,"completion_tokens":9,"prompt_tokens_details":{"cached_tokens":32}}}`,
		`data: [DONE]`,
		``,
	}, "\n\n")
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewOpenAICompat(ChatConfig{URL: srv.URL, APIKey: "test-key", Model: "gpt-4o-mini"})
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
	if done == nil {
		t.Fatal("no done event emitted")
	}
	if done.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", done.FinishReason, "stop")
	}
	// done must be the terminal event.
	if events[len(events)-1].Type != "done" {
		t.Errorf("last event type = %q, want done", events[len(events)-1].Type)
	}

	// Usage counters accumulated from the final usage chunk.
	usage := eng.(UsageReporter).Usage()
	if usage.TokensIn != 57 || usage.TokensOut != 9 || usage.CachedTokens != 32 {
		t.Errorf("usage = %+v, want in=57 out=9 cached=32", usage)
	}

	// Request-side assertions.
	req := <-reqCh
	if req.method != http.MethodPost {
		t.Errorf("method = %s, want POST", req.method)
	}
	if got := req.header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", got)
	}
	if req.body["model"] != "gpt-4o-mini" {
		t.Errorf("model = %v, want gpt-4o-mini", req.body["model"])
	}
	if req.body["stream"] != true {
		t.Errorf("stream = %v, want true", req.body["stream"])
	}
	so, _ := req.body["stream_options"].(map[string]interface{})
	if so["include_usage"] != true {
		t.Errorf("stream_options.include_usage = %v, want true", so["include_usage"])
	}
	bodyMsgs, _ := req.body["messages"].([]interface{})
	if len(bodyMsgs) != 2 {
		t.Fatalf("messages len = %d, want 2", len(bodyMsgs))
	}
	first, _ := bodyMsgs[0].(map[string]interface{})
	if first["role"] != "system" || first["content"] != "You are terse." {
		t.Errorf("messages[0] = %v", first)
	}
}

// TestChatStream_ToolCalls verifies that tool-call fragments spread
// across multiple chunks (id/name first, arguments dribbled after, two
// parallel calls interleaved by index) are assembled into one batched
// tool_calls event in index order, plus tools request serialization.
func TestChatStream_ToolCalls(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"get_time","arguments":"{}"}}]},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		``,
	}, "\n\n")
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewOpenAICompat(ChatConfig{URL: srv.URL, Model: "gpt-4o-mini"})
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
	if tc.ToolCalls[0].ID != "call_a" || tc.ToolCalls[0].Name != "get_weather" {
		t.Errorf("call[0] = %+v", tc.ToolCalls[0])
	}
	if tc.ToolCalls[0].Arguments != `{"city":"Paris"}` {
		t.Errorf("call[0].Arguments = %q, want accumulated JSON", tc.ToolCalls[0].Arguments)
	}
	if tc.ToolCalls[1].ID != "call_b" || tc.ToolCalls[1].Name != "get_time" || tc.ToolCalls[1].Arguments != "{}" {
		t.Errorf("call[1] = %+v", tc.ToolCalls[1])
	}
	done := findEvent(events, "done")
	if done == nil || done.FinishReason != "tool_calls" {
		t.Errorf("done = %+v, want FinishReason tool_calls", done)
	}

	// tools serialization: OpenAI wraps each def in {type:function,function:{...}}.
	req := <-reqCh
	bodyTools, _ := req.body["tools"].([]interface{})
	if len(bodyTools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(bodyTools))
	}
	tool0, _ := bodyTools[0].(map[string]interface{})
	if tool0["type"] != "function" {
		t.Errorf("tools[0].type = %v, want function", tool0["type"])
	}
	fn, _ := tool0["function"].(map[string]interface{})
	if fn["name"] != "get_weather" || fn["description"] != "Current weather for a city" {
		t.Errorf("tools[0].function = %v", fn)
	}
}

// TestChatStream_ContextCancel: the server drips one delta then holds
// the connection open; cancelling the caller's context must unblock
// Stream promptly with an error and a closed events channel.
func TestChatStream_ContextCancel(t *testing.T) {
	srv := httptest.NewServer(dripSSEHandler(
		"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n"))
	defer srv.Close()

	eng := NewOpenAICompat(ChatConfig{URL: srv.URL, Model: "m"})
	cancelMidStream(t, eng)
}

// TestChatStream_HTTPError: a 429 with a JSON error body must surface
// as a descriptive error (status + body) with no events emitted.
func TestChatStream_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded"}}`))
	}))
	defer srv.Close()

	eng := NewOpenAICompat(ChatConfig{URL: srv.URL, Model: "m"})
	events, err := runLLMStream(t, context.Background(), eng, []LLMMessage{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected error on 429, got nil")
	}
	if !strings.Contains(err.Error(), "429") || !strings.Contains(err.Error(), "rate limit exceeded") {
		t.Errorf("error = %v, want status + body", err)
	}
	if len(events) != 0 {
		t.Errorf("events = %+v, want none on HTTP error", events)
	}
}

// TestChatStream_MalformedDataLine: a garbage data line between valid
// chunks is skipped (the parser is deliberately tolerant), and the
// stream still completes normally.
func TestChatStream_MalformedDataLine(t *testing.T) {
	reqCh := make(chan capturedLLMRequest, 1)
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"before"},"finish_reason":null}]}`,
		`data: {this is not json`,
		`data: {"choices":[{"delta":{"content":" after"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
		``,
	}, "\n\n")
	srv := httptest.NewServer(sseHandler(reqCh, sse))
	defer srv.Close()

	eng := NewOpenAICompat(ChatConfig{URL: srv.URL, Model: "m"})
	events, err := runLLMStream(t, context.Background(), eng, []LLMMessage{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Stream error: %v (malformed lines must be skipped, not fatal)", err)
	}
	if got := joinTextDeltas(events); got != "before after" {
		t.Errorf("assembled text = %q, want %q", got, "before after")
	}
	if done := findEvent(events, "done"); done == nil || done.FinishReason != "stop" {
		t.Errorf("done = %+v, want FinishReason stop", done)
	}
}
