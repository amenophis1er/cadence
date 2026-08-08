package cadence

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/amenophis1er/cadence/engines"
)

func collect(t *testing.T, cfg SentenceBufferConfig, deltas []string) ([]string, SentenceBufferResult) {
	t.Helper()
	in := make(chan engines.LLMEvent)
	out := make(chan string, 32)
	done := make(chan SentenceBufferResult, 1)
	go func() { done <- RunSentenceBuffer(context.Background(), cfg, in, out) }()
	for _, d := range deltas {
		in <- engines.LLMEvent{Type: "text", TextDelta: d}
	}
	close(in)
	res := <-done
	var chunks []string
	for c := range out {
		chunks = append(chunks, c)
	}
	return chunks, res
}

func TestFlushesOnSentenceTerminator(t *testing.T) {
	chunks, res := collect(t, DefaultSentenceBufferConfig(),
		[]string{"Hello ", "there. ", "How are ", "you?"})
	if len(chunks) != 2 {
		t.Fatalf("want 2 sentence chunks, got %d: %q", len(chunks), chunks)
	}
	if chunks[0] != "Hello there." {
		t.Fatalf("first chunk = %q", chunks[0])
	}
	if res.FullText != "Hello there. How are you?" {
		t.Fatalf("full text = %q", res.FullText)
	}
}

func TestClauseBoundaryOnlyAfterMinWords(t *testing.T) {
	cfg := DefaultSentenceBufferConfig()
	cfg.ClauseMinWords = 3
	chunks, _ := collect(t, cfg, []string{"one two three four,", " tail."})
	if len(chunks) != 2 || !strings.HasSuffix(chunks[0], ",") {
		t.Fatalf("clause flush expected after min words: %q", chunks)
	}
	// Below the minimum, a comma must NOT flush.
	chunks, _ = collect(t, cfg, []string{"one,", " two."})
	if len(chunks) != 1 {
		t.Fatalf("comma below min words must not flush: %q", chunks)
	}
}

func TestHardWordCapFlushes(t *testing.T) {
	cfg := DefaultSentenceBufferConfig()
	cfg.MaxWords = 4
	chunks, _ := collect(t, cfg, []string{"a b ", "c d ", "e f."})
	if len(chunks) != 2 {
		t.Fatalf("cap flush expected: %q", chunks)
	}
}

func TestIdleTimeoutFlushes(t *testing.T) {
	cfg := DefaultSentenceBufferConfig()
	cfg.FlushTimeout = 30 * time.Millisecond
	in := make(chan engines.LLMEvent)
	out := make(chan string, 4)
	go RunSentenceBuffer(context.Background(), cfg, in, out)
	in <- engines.LLMEvent{Type: "text", TextDelta: "no punctuation here"}
	select {
	case c := <-out:
		if c != "no punctuation here" {
			t.Fatalf("timeout flush = %q", c)
		}
	case <-time.After(time.Second):
		t.Fatal("idle timeout never flushed")
	}
	close(in)
}

func TestCancelReturnsPartialResult(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan engines.LLMEvent)
	out := make(chan string, 4)
	done := make(chan SentenceBufferResult, 1)
	go func() { done <- RunSentenceBuffer(ctx, DefaultSentenceBufferConfig(), in, out) }()
	in <- engines.LLMEvent{Type: "text", TextDelta: "partial answer"}
	cancel()
	select {
	case res := <-done:
		if res.FullText != "partial answer" {
			t.Fatalf("full text after cancel = %q", res.FullText)
		}
	case <-time.After(time.Second):
		t.Fatal("cancel did not unblock RunSentenceBuffer")
	}
}

func TestCancelUnblocksStalledFlush(t *testing.T) {
	// Unbuffered out with no reader: a terminator-triggered flush blocks on
	// the send. Cancelling ctx must still let the buffer return.
	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan engines.LLMEvent, 1)
	out := make(chan string) // no reader
	done := make(chan SentenceBufferResult, 1)
	go func() { done <- RunSentenceBuffer(ctx, DefaultSentenceBufferConfig(), in, out) }()
	in <- engines.LLMEvent{Type: "text", TextDelta: "Stuck sentence."}
	time.Sleep(20 * time.Millisecond) // let the flush reach the blocked send
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cancel did not unblock a flush stalled on out")
	}
}

func TestEmptyAndWhitespaceDeltasProduceNoChunks(t *testing.T) {
	chunks, res := collect(t, DefaultSentenceBufferConfig(), []string{"", "   ", "\n"})
	if len(chunks) != 0 {
		t.Fatalf("whitespace-only input must emit no chunks, got %q", chunks)
	}
	if res.FullText != "   \n" {
		t.Fatalf("full text should keep raw whitespace deltas, got %q", res.FullText)
	}
}

func TestToolCallsSurviveToResult(t *testing.T) {
	in := make(chan engines.LLMEvent)
	out := make(chan string, 4)
	done := make(chan SentenceBufferResult, 1)
	go func() { done <- RunSentenceBuffer(context.Background(), DefaultSentenceBufferConfig(), in, out) }()
	in <- engines.LLMEvent{Type: "tool_calls", ToolCalls: []engines.ToolCall{{Name: "hangup"}}}
	close(in)
	res := <-done
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "hangup" {
		t.Fatalf("tool calls lost: %+v", res.ToolCalls)
	}
}
