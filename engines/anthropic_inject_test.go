package engines

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// anthropicInjectSSE is a minimal well-formed Messages stream: usage on
// message_start, one text delta, terminal message_stop. Deliberately a
// fixed const rather than a reuse of anthropic_test.go's anthropicSSE()
// frame builder — these tests assert on transport wiring, not on stream
// shape, so a literal that never moves is the clearer fixture (and the
// distinct name keeps the two files from colliding in-package).
const anthropicInjectSSE = `data: {"type":"message_start","message":{"usage":{"input_tokens":11,"output_tokens":1}}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

data: {"type":"message_stop"}

`

// TestNewAnthropic_UsesInjectedHTTPClient verifies the caller's client
// is used verbatim, for the same reason as the chat engine: a real-time
// consumer needs a transport it sized itself so a vendor stalling
// before response headers becomes an error its failover can act on.
// The configured host is a closed port — only the injected transport
// can answer, so a response at all proves the injection took.
func TestNewAnthropic_UsesInjectedHTTPClient(t *testing.T) {
	rt := &recordingTransport{body: anthropicInjectSSE}
	injected := &http.Client{Transport: rt}

	eng := NewAnthropic(AnthropicConfig{
		URL:        "http://127.0.0.1:1",
		APIKey:     "test-key",
		Model:      "claude-haiku-4-5",
		HTTPClient: injected,
	})

	if got := eng.(*anthropicEngine).client; got != injected {
		t.Fatalf("engine client = %p, want the injected client %p", got, injected)
	}

	out := make(chan LLMEvent, 16)
	if err := eng.Stream(context.Background(), []LLMMessage{{Role: "user", Content: "hi"}}, nil, out); err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	if rt.calls != 1 {
		t.Fatalf("injected transport saw %d requests, want 1", rt.calls)
	}
	if rt.lastPath != "/v1/messages" {
		t.Errorf("path = %q, want /v1/messages", rt.lastPath)
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

// TestNewAnthropic_NilHTTPClientUsesDefault confirms the nil case is
// untouched: the engine builds its own client and talks to a real
// server over a real socket.
func TestNewAnthropic_NilHTTPClientUsesDefault(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %q, want /v1/messages", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, anthropicInjectSSE)
	}))
	defer srv.Close()

	eng := NewAnthropic(AnthropicConfig{URL: srv.URL, APIKey: "test-key", Model: "claude-haiku-4-5"})
	if eng.(*anthropicEngine).client == nil {
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
	if text != "ok" {
		t.Errorf("streamed text = %q, want %q", text, "ok")
	}
	if u := eng.(*anthropicEngine).Usage(); u.TokensIn != 11 {
		t.Errorf("TokensIn = %d, want 11", u.TokensIn)
	}
}
