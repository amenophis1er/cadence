package cadence

import (
	"context"
	"strings"
	"time"
	"unicode"

	"github.com/amenophis1er/cadence/engines"
)

// sentenceBufferConfig tunes the LLM-to-TTS framing pipeline. Defaults match
// the Rust fusion prototype: flush on hard sentence terminators, on clause
// boundaries once enough words have accumulated, on inactivity, or on a hard
// word cap. The numbers trade first-token latency against TTS prosody.
type sentenceBufferConfig struct {
	clauseMinWords int           // flush at "," ";" ":" "—" only after this many words
	maxWords       int           // hard cap regardless of punctuation
	flushTimeout   time.Duration // flush after this much idle time since last delta
}

func defaultSentenceBufferConfig() sentenceBufferConfig {
	return sentenceBufferConfig{
		clauseMinWords: 15,
		maxWords:       40,
		flushTimeout:   300 * time.Millisecond,
	}
}

// sentenceBufferResult is what runSentenceBuffer hands back when the upstream
// LLM event channel closes.
type sentenceBufferResult struct {
	fullText  string
	toolCalls []engines.ToolCall
}

// runSentenceBuffer consumes a stream of LLMEvents (text deltas + tool calls)
// and emits sentence-aligned text chunks downstream so TTS can begin
// synthesising before the LLM has finished generating. Returns once `in`
// closes or `ctx` cancels.
//
// `out` is closed before return so the consuming TTS goroutine unblocks.
func runSentenceBuffer(
	ctx context.Context,
	cfg sentenceBufferConfig,
	in <-chan engines.LLMEvent,
	out chan<- string,
) sentenceBufferResult {
	defer close(out)

	var fullText strings.Builder
	var toolCalls []engines.ToolCall
	var pending strings.Builder
	pendingWords := 0

	timer := time.NewTimer(cfg.flushTimeout)
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
			return sentenceBufferResult{fullText: fullText.String(), toolCalls: toolCalls}

		case <-timer.C:
			timerArmed = false
			if !flush() {
				return sentenceBufferResult{fullText: fullText.String(), toolCalls: toolCalls}
			}

		case ev, ok := <-in:
			if !ok {
				_ = flush()
				return sentenceBufferResult{fullText: fullText.String(), toolCalls: toolCalls}
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
						return sentenceBufferResult{fullText: fullText.String(), toolCalls: toolCalls}
					}
					flushed = true
				} else if pendingWords >= cfg.maxWords {
					if !flush() {
						return sentenceBufferResult{fullText: fullText.String(), toolCalls: toolCalls}
					}
					flushed = true
				} else if pendingWords >= cfg.clauseMinWords && hasClauseBoundary(ev.TextDelta) {
					if !flush() {
						return sentenceBufferResult{fullText: fullText.String(), toolCalls: toolCalls}
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
					timer.Reset(cfg.flushTimeout)
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
// breaks we'll flush at only after clauseMinWords have accumulated.
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
