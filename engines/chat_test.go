package engines

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestParseSSE_CapturesCachedTokens verifies that parseSSE extracts the
// nested prompt_tokens_details.cached_tokens field from the final SSE
// chunk. This is the load-bearing signal for whether prompt caching is
// actually paying off — without this parse, the prefix-stabilization
// work is invisible.
func TestParseSSE_CapturesCachedTokens(t *testing.T) {
	// Minimal OpenAI-shaped SSE stream: one text delta, one usage chunk
	// (which is what `stream_options.include_usage: true` produces), then
	// [DONE]. Mirrors the shape OpenAI / Groq / LiteLLM emit when caching
	// is enabled upstream.
	stream := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":2147,"completion_tokens":42,"prompt_tokens_details":{"cached_tokens":2048}}}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	out := make(chan LLMEvent, 16)
	usage, err := parseSSE(context.Background(), strings.NewReader(stream), out)
	if err != nil {
		t.Fatalf("parseSSE error: %v", err)
	}

	if usage.TokensIn != 2147 {
		t.Errorf("TokensIn = %d, want 2147", usage.TokensIn)
	}
	if usage.TokensOut != 42 {
		t.Errorf("TokensOut = %d, want 42", usage.TokensOut)
	}
	if usage.CachedTokens != 2048 {
		t.Errorf("CachedTokens = %d, want 2048 (the cache-hit signal)", usage.CachedTokens)
	}
}

// TestParseSSE_AbsentCachedTokensIsZero confirms that vendors without
// prefix caching (or without the prompt_tokens_details field) still
// produce a valid usage — CachedTokens just stays at zero. We don't
// want a missing field to cause a parse failure or false signal.
func TestParseSSE_AbsentCachedTokensIsZero(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":3}}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	out := make(chan LLMEvent, 16)
	usage, err := parseSSE(context.Background(), strings.NewReader(stream), out)
	if err != nil {
		t.Fatalf("parseSSE error: %v", err)
	}

	if usage.TokensIn != 120 || usage.TokensOut != 3 {
		t.Errorf("usage mismatch: in=%d out=%d", usage.TokensIn, usage.TokensOut)
	}
	if usage.CachedTokens != 0 {
		t.Errorf("CachedTokens = %d, want 0 (no caching field present)", usage.CachedTokens)
	}
}

// recordingTransport is a stand-in for a real network transport: it
// answers every request from a canned body and keeps what it was asked
// for. Shared by the chat and Anthropic injected-client tests — an
// engine that routes through this transport at all is proof it used the
// caller's client, since neither test's URL is reachable otherwise.
type recordingTransport struct {
	body     string
	calls    int
	lastPath string
	lastBody string
}

func (rt *recordingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	rt.calls++
	rt.lastPath = r.URL.Path
	if r.Body != nil {
		raw, _ := io.ReadAll(r.Body)
		rt.lastBody = string(raw)
	}
	return &http.Response{
		StatusCode: 200,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(rt.body)),
		Request:    r,
	}, nil
}

// TestBuildChatRequest_MaxTokensEmitted covers the cap that keeps an
// interactive reply inside its slot. The key must be the current
// OpenAI-compatible spelling (max_completion_tokens) and must NOT be
// accompanied by the legacy max_tokens alias — servers that accept both
// reject a request carrying the pair.
func TestBuildChatRequest_MaxTokensEmitted(t *testing.T) {
	cfg := ChatConfig{Model: "gpt-4o-mini", MaxTokens: 160}
	req := buildChatRequest(cfg, []LLMMessage{{Role: "user", Content: "hi"}}, nil)

	got, ok := req["max_completion_tokens"]
	if !ok {
		t.Fatalf("max_completion_tokens absent, want 160")
	}
	if got != 160 {
		t.Errorf("max_completion_tokens = %v, want 160", got)
	}
	if _, legacy := req["max_tokens"]; legacy {
		t.Errorf("legacy max_tokens also emitted; only the current name may be sent")
	}
}

// TestBuildChatRequest_MaxTokensOmittedWhenZero pins the backward
// compatibility promise: the zero value must produce exactly the
// request shape callers got before the field existed, leaving the
// length to the server default.
func TestBuildChatRequest_MaxTokensOmittedWhenZero(t *testing.T) {
	cfg := ChatConfig{Model: "gpt-4o-mini"}
	req := buildChatRequest(cfg, []LLMMessage{{Role: "user", Content: "hi"}}, nil)

	if _, ok := req["max_completion_tokens"]; ok {
		t.Errorf("max_completion_tokens present with MaxTokens=0, want omitted")
	}
	if _, ok := req["max_tokens"]; ok {
		t.Errorf("max_tokens present with MaxTokens=0, want omitted")
	}
}

// TestNewOpenAICompat_UsesInjectedHTTPClient verifies the caller's
// client is used verbatim — the reason a real-time consumer can set its
// own ResponseHeaderTimeout and have a stalling vendor become an error
// on its clock. The URL points at a closed port, so a request that
// reached the default client would fail with connection refused; only
// the injected transport can answer it.
func TestNewOpenAICompat_UsesInjectedHTTPClient(t *testing.T) {
	rt := &recordingTransport{body: "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"}
	injected := &http.Client{Transport: rt}

	cfg := ChatConfig{
		URL:        "http://127.0.0.1:1/v1/chat/completions",
		Model:      "gpt-4o-mini",
		MaxTokens:  160,
		HTTPClient: injected,
	}
	eng := NewOpenAICompat(cfg)

	if got := eng.(*chatEngine).client; got != injected {
		t.Fatalf("engine client = %p, want the injected client %p", got, injected)
	}

	out := make(chan LLMEvent, 16)
	if err := eng.Stream(context.Background(), []LLMMessage{{Role: "user", Content: "hi"}}, nil, out); err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	if rt.calls != 1 {
		t.Fatalf("injected transport saw %d requests, want 1", rt.calls)
	}
	if !strings.Contains(rt.lastBody, `"max_completion_tokens":160`) {
		t.Errorf("request body missing the cap: %s", rt.lastBody)
	}

	var text string
	for ev := range out {
		if ev.Type == "text" {
			text += ev.TextDelta
		}
	}
	if text != "ok" {
		t.Errorf("streamed text = %q, want %q", text, "ok")
	}
}

// TestNewOpenAICompat_NilHTTPClientUsesDefault confirms the nil case is
// untouched: the engine still builds its own client and talks to a real
// server over a real socket.
func TestNewOpenAICompat_NilHTTPClientUsesDefault(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hey\"},\"finish_reason\":\"stop\"}]}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	eng := NewOpenAICompat(ChatConfig{URL: srv.URL, Model: "gpt-4o-mini"})
	if eng.(*chatEngine).client == nil {
		t.Fatal("engine built with a nil client")
	}

	out := make(chan LLMEvent, 16)
	if err := eng.Stream(context.Background(), []LLMMessage{{Role: "user", Content: "hi"}}, nil, out); err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var text string
	for ev := range out {
		if ev.Type == "text" {
			text += ev.TextDelta
		}
	}
	if text != "hey" {
		t.Errorf("streamed text = %q, want %q", text, "hey")
	}
}
