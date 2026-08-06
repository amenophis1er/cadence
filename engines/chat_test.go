package engines

import (
	"context"
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
