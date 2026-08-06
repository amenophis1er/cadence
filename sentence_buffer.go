package cadence

import (
	"context"
	"strings"
	"time"
	"unicode"

	"github.com/amenophis1er/cadence/engines"
)

// SentenceBufferConfig tunes the LLM-to-TTS framing pipeline. Defaults match
// the original Rust prototype: flush on hard sentence terminators, on clause
// boundaries once enough words have accumulated, on inactivity, or on a hard
// word cap. The numbers trade first-token latency against TTS prosody.
type SentenceBufferConfig struct {
	ClauseMinWords int           // flush at "," ";" ":" "—" only after this many words
	MaxWords       int           // hard cap regardless of punctuation
	FlushTimeout   time.Duration // flush after this much idle time since last delta
}

func DefaultSentenceBufferConfig() SentenceBufferConfig {
	return SentenceBufferConfig{
		ClauseMinWords: 15,
		MaxWords:       40,
		FlushTimeout:   300 * time.Millisecond,
	}
}

// SentenceBufferResult is what RunSentenceBuffer hands back when the upstream
// LLM event channel closes.
type SentenceBufferResult struct {
	FullText  string
	ToolCalls []engines.ToolCall
}

// RunSentenceBuffer consumes a stream of LLMEvents (text deltas + tool calls)
// and emits sentence-aligned text chunks downstream so TTS can begin
// synthesising before the LLM has finished generating. Returns once `in`
// closes or `ctx` cancels.
//
// `out` is closed before return so the consuming TTS goroutine unblocks.
func RunSentenceBuffer(
	ctx context.Context,
	cfg SentenceBufferConfig,
	in <-chan engines.LLMEvent,
	out chan<- string,
) SentenceBufferResult {
	defer close(out)

	var fullText strings.Builder
	var toolCalls []engines.ToolCall
	var pending strings.Builder
	pendingWords := 0

	timer := time.NewTimer(cfg.FlushTimeout)
	timer.Stop()
	timerArmed := false

	// Stop the timer on every return path so a pending fire doesn't leak
	// past the goroutine's lifetime.
	defer func() {
		if timerArmed && !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
	}()

	flush := func() bool {
		if pending.Len() == 0 {
			return true
		}
		chunk := strings.TrimSpace(pending.String())
		pending.Reset()
		pendingWords = 0
		if timerArmed {
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timerArmed = false
		}
		if chunk == "" {
			return true
		}
		select {
		case out <- chunk:
			return true
		case <-ctx.Done():
			return false
		}
	}

	for {
		select {
		case <-ctx.Done():
			return SentenceBufferResult{FullText: fullText.String(), ToolCalls: toolCalls}

		case <-timer.C:
			timerArmed = false
			if !flush() {
				return SentenceBufferResult{FullText: fullText.String(), ToolCalls: toolCalls}
			}

		case ev, ok := <-in:
			if !ok {
				_ = flush()
				return SentenceBufferResult{FullText: fullText.String(), ToolCalls: toolCalls}
			}

			switch ev.Type {
			case "tool_calls":
				toolCalls = ev.ToolCalls

			case "text":
				if ev.TextDelta == "" {
					continue
				}
				fullText.WriteString(ev.TextDelta)
				pending.WriteString(ev.TextDelta)
				pendingWords += countWords(ev.TextDelta)

				flushed := false
				if hasSentenceTerminator(ev.TextDelta) {
					if !flush() {
						return SentenceBufferResult{FullText: fullText.String(), ToolCalls: toolCalls}
					}
					flushed = true
				} else if pendingWords >= cfg.MaxWords {
					if !flush() {
						return SentenceBufferResult{FullText: fullText.String(), ToolCalls: toolCalls}
					}
					flushed = true
				} else if pendingWords >= cfg.ClauseMinWords && hasClauseBoundary(ev.TextDelta) {
					if !flush() {
						return SentenceBufferResult{FullText: fullText.String(), ToolCalls: toolCalls}
					}
					flushed = true
				}

				if !flushed {
					if timerArmed {
						if !timer.Stop() {
							select {
							case <-timer.C:
							default:
							}
						}
					}
					timer.Reset(cfg.FlushTimeout)
					timerArmed = true
				}
			}
		}
	}
}

// hasSentenceTerminator reports whether the delta contains "." "!" "?" — the
// hard end-of-sentence markers TTS should flush on.
func hasSentenceTerminator(s string) bool {
	return strings.ContainsAny(s, ".!?")
}

// hasClauseBoundary reports whether the delta contains "," ";" ":" "—" — soft
// breaks we'll flush at only after ClauseMinWords have accumulated.
func hasClauseBoundary(s string) bool {
	return strings.ContainsAny(s, ",;:—")
}

func countWords(s string) int {
	n := 0
	inWord := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if inWord {
				n++
				inWord = false
			}
		} else {
			inWord = true
		}
	}
	if inWord {
		n++
	}
	return n
}
