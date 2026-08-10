// Package engines defines the pluggable STT, LLM, and TTS interfaces that the
// cadence orchestrator composes into a voice pipeline. Concrete implementations
// (whisper.go, kokoro.go, chat.go) live in this same package.
package engines

import (
	"context"
	"io"
)

// PCM16 frames flow through the pipeline in 16 kHz mono int16 little-endian
// bytes (the lingua franca of speaches/whisper/kokoro). The cadence provider
// translates to/from mu-law 8 kHz at the supervisor boundary.
type PCM16Frame []byte

// TranscriptSegment is one STT result. Partial segments may arrive while the
// user is still speaking; only IsFinal=true segments commit a turn.
type TranscriptSegment struct {
	Text    string
	IsFinal bool
}

// LLMMessage is one message in the chat history, OpenAI-shaped.
type LLMMessage struct {
	Role       string     // "system", "user", "assistant", "tool"
	Content    string     // text content (may be empty when ToolCalls is set)
	Name       string     // tool name (when Role == "tool")
	ToolCallID string     // matching tool call id (when Role == "tool")
	ToolCalls  []ToolCall // populated on assistant turns that invoke tools
}

// ToolCall is the LLM's request to invoke a function.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // raw JSON string of arguments
}

// ToolDef is one function exposed to the LLM (OpenAI tools schema).
type ToolDef struct {
	Name        string
	Description string
	Parameters  map[string]interface{}
}

// LLMEvent is one item in the streaming response from an LLM.
type LLMEvent struct {
	// TextDelta is a chunk of assistant text. Empty unless Type == "text".
	TextDelta string
	// ToolCalls is the finalized list of tool calls, sent once when the model
	// finishes a tool-using turn. Empty unless Type == "tool_calls".
	ToolCalls []ToolCall
	// Type is one of: "text", "tool_calls", "done". Every Stream ends
	// with a "done" event — including on request failure or context
	// cancellation (best-effort: dropped rather than blocking if the
	// consumer has stopped draining). Stream's error return remains the
	// authoritative failure signal.
	Type string
	// FinishReason is set on "done" events: vendor reasons on success
	// ("stop", "tool_calls", "length", ...), "cancelled" when the
	// context was cancelled, "error" on any other failure.
	FinishReason string
}

// STTEngine streams audio in, transcripts out.
type STTEngine interface {
	// Transcribe consumes one buffered utterance (16 kHz PCM16 mono) and
	// returns a single final transcript. Used by the batch/VAD-driven path
	// in the mediator: caller buffers an utterance locally, calls Transcribe
	// once, gets back the final text.
	Transcribe(ctx context.Context, pcm16 PCM16Frame) (string, error)
	Name() string
}

// StreamingSTTEngine is the optional richer interface that real-time STT
// servers (OpenAI Realtime, speaches /v1/realtime, WhisperLive, …) expose.
// When the cadence mediator detects an STT engine that satisfies this
// interface it skips local energy VAD and lets the engine drive endpointing
// + partial transcripts as audio flows in.
//
// Lifecycle:
//
//	Start(ctx, events) → PushAudio frames continuously → Stop()
//
// Events flow asynchronously on the channel passed to Start. The engine
// closes the channel when its session ends.
type StreamingSTTEngine interface {
	STTEngine

	// Start opens the engine session. Audio frames pushed via PushAudio
	// after Start returns will be processed and transcript/speech events
	// delivered on `events`. Caller owns the events channel; engine writes
	// only and closes it on session end.
	Start(ctx context.Context, events chan<- STTEvent) error

	// PushAudio feeds one mu-law 8 kHz frame (typically 160 bytes / 20 ms)
	// to the engine. Non-blocking — drops on overrun rather than stalling
	// the audio loop.
	PushAudio(frame []byte)

	// Stop closes the session and releases resources. Idempotent.
	// Engine-specific shutdown sentinels (e.g. WhisperLive's
	// END_OF_AUDIO) are sent inside Stop, not exposed to the mediator —
	// per-utterance finalization is handled by each engine's reader
	// (e.g. trailing-segment repeat tracking for WhisperLive).
	Stop() error
}

// STTEvent is one item the streaming STT engine emits.
type STTEvent struct {
	Type STTEventType
	Text string // populated for TranscriptDelta + TranscriptFinal
	Err  error  // populated for Error

	// CommittedBy names the rule that ended the utterance. Populated on
	// TranscriptFinal only; empty on every other event type.
	//
	// This matters because the paths differ by roughly a second of
	// caller-perceived latency, and without it a consumer cannot tell a
	// snappy provider-endpointed turn from one that sat waiting on the
	// fallback debounce — they arrive as the same event.
	CommittedBy CommitPolicy

	// HeldMs is how long THIS engine held the utterance locally before
	// committing it: measured from the last finalized (is_final) result
	// to the commit. Populated on TranscriptFinal only. For multi-segment
	// utterances the anchor is the LAST segment, so earlier segments'
	// hold time is not included; on a CommitSessionEnd flush it can be
	// large if the stream sat idle after the last final before ending.
	//
	// Note what it does NOT include: when CommittedBy is
	// CommitSpeechFinal, the provider did its own silence-threshold wait
	// server-side before telling us, and that wait is invisible from here
	// — expect a small HeldMs even though the caller waited longer. Use it
	// to attribute latency this engine controls, not as a total
	// speech-to-commit measure.
	HeldMs int64
}

// CommitPolicy names the rule that committed an utterance, so a consumer can
// attribute endpointing latency instead of inferring it.
type CommitPolicy string

const (
	// CommitSpeechFinal: the provider signalled end-of-utterance itself
	// (e.g. Deepgram speech_final) and the engine committed immediately.
	CommitSpeechFinal CommitPolicy = "speech_final"
	// CommitFlushTimeout: the provider never signalled end-of-utterance, so
	// the engine committed its buffered finals after the flush debounce.
	// Systematically slower than CommitSpeechFinal by roughly the debounce.
	CommitFlushTimeout CommitPolicy = "flush_timeout"
	// CommitSessionEnd: the stream ended (hangup, read error) with text
	// still buffered, and the engine flushed it rather than lose it.
	CommitSessionEnd CommitPolicy = "session_end"
)

// STTEventType enumerates the streaming STT lifecycle.
type STTEventType int

const (
	// STTSpeechStarted: server VAD detected speech onset.
	STTSpeechStarted STTEventType = iota
	// STTSpeechStopped: server VAD detected speech offset.
	STTSpeechStopped
	// STTTranscriptDelta: partial transcript update (live, not final).
	STTTranscriptDelta
	// STTTranscriptFinal: final transcript for one committed utterance.
	STTTranscriptFinal
	// STTError: a non-recoverable engine error occurred.
	STTError
)

// TTSEngine streams text in, audio out.
type TTSEngine interface {
	// Synthesize converts text to a stream of PCM16 mono bytes at SampleRate().
	// Returns an io.ReadCloser so the caller can pipe audio to the telephony
	// output as bytes arrive instead of waiting for the full sentence to be
	// generated. Caller must Close() the returned reader.
	Synthesize(ctx context.Context, text string) (io.ReadCloser, error)
	// SampleRate is the sample rate of the audio Synthesize returns.
	SampleRate() int
	Name() string
}

// TextChunk is a unit of text the LLM has produced for TTS. The
// IsSentenceEnd flag tells the engine when to flush its server-side
// buffer (vendors that support manual flushing).
type TextChunk struct {
	Text          string
	IsSentenceEnd bool
}

// AudioChunk is one frame of synthesized audio. Format depends on the
// engine — typically mu-law 8 kHz for telephony engines after vendor-side
// decode, or PCM16 at SampleRate() if the engine returns raw samples.
type AudioChunk struct {
	Data []byte
}

// StreamingTTSEngine owns a WebSocket (or other long-lived transport)
// for the duration of one Stream() call, mirroring the channel-based
// design of the original Rust prototype. Compared to the
// HTTP-shaped TTSEngine.Synthesize, this lets the engine:
//
//   - hold a warm connection across multiple sentences, avoiding
//     per-sentence handshake cost;
//   - handle protocol-level details (BOS/EOS sentinels, base64-in-JSON
//     decode, vendor request_id capture) inside the engine without
//     forcing the mediator to bridge to an io.ReadCloser;
//   - support cancellation cleanly via ctx — barge-in or end-of-call
//     flips one signal and the engine drains and exits.
//
// Stream contract:
//
//   - The mediator pushes TextChunks to textCh; the channel close
//     signals end-of-input.
//   - The engine pushes AudioChunks to audioCh as audio arrives. The
//     engine MUST close audioCh before returning so the consumer
//     observes end-of-stream.
//   - On ctx cancellation: send vendor's EOS sentinel, drain, return.
//   - Returns nil on graceful end, error on transport / protocol failure.
//   - One Stream() invocation per call (or per cadence Provider lifetime,
//     which is per-module under the current architecture). Engines that
//     also implement TTSEngine can be used in either path; the mediator
//     prefers StreamingTTSEngine when both are satisfied.
type StreamingTTSEngine interface {
	TTSEngine

	// Stream owns the WS for the duration of the call. See contract above.
	Stream(ctx context.Context, textCh <-chan TextChunk, audioCh chan<- AudioChunk) error

	// ProviderRequestID returns the vendor's correlation handle for this
	// session — populated at WS-handshake time (or first server message,
	// vendor-dependent). Empty before the handshake completes; stable
	// thereafter for the life of the Stream() call. Persisted on the
	// per-leg provider_sessions row for support-ticket correlation.
	ProviderRequestID() string
}

// InterruptibleTTSEngine is the optional capability a StreamingTTSEngine
// implements when it can abandon in-flight synthesis WITHOUT ending the
// session — the barge-in primitive.
//
// Why this is separate from ctx cancellation: cancelling a Stream() ends it
// ("one Stream() invocation per call"), which for a barge-in means tearing
// down the warm WebSocket and paying a fresh handshake before the agent can
// speak again. A caller who interrupts twice in one conversation would pay it
// twice. Clear instead discards what is queued and leaves the session warm
// and immediately reusable.
//
// Contract:
//
//   - Clear may be called from a different goroutine than the one driving
//     Stream, and only while a Stream is live; before or after, it is a no-op
//     returning nil.
//   - Audio the vendor had already queued is discarded by the ENGINE, not the
//     consumer: after Clear returns, no further pre-interruption chunks are
//     FORWARDED to audioCh. Chunks the engine had already delivered before
//     Clear may still sit in audioCh's buffer (the channel is caller-owned) —
//     a consumer doing barge-in should drop what it has already dequeued or
//     buffered for playback, but never has to reason about the vendor's
//     in-flight boundary.
//   - The session stays open. Text pushed to textCh after Clear is spoken
//     normally on the same connection.
type InterruptibleTTSEngine interface {
	StreamingTTSEngine

	// Clear abandons queued synthesis and keeps the session usable.
	Clear() error
}

// LLMEngine streams a chat completion event-by-event.
type LLMEngine interface {
	Stream(ctx context.Context, messages []LLMMessage, tools []ToolDef, out chan<- LLMEvent) error
	Name() string
}

// Usage captures per-leg consumption units for cost attribution. Engines
// populate the fields they meaningfully measure; zero values mean "not
// applicable" or "not surfaced by the provider response".
type Usage struct {
	AudioSeconds float64 // STT audio processed; TTS audio generated
	CharsIn      int     // TTS input text length
	CharsOut     int     // reserved for future engines that bill on output chars
	TokensIn     int     // LLM prompt tokens (total, including cached prefix hits)
	TokensOut    int     // LLM completion tokens

	// CachedTokens is the subset of TokensIn that the provider served from
	// its prompt cache. Zero means either the cache missed or the provider
	// doesn't surface the field. Vendor-specific population:
	//   OpenAI compatible: usage.prompt_tokens_details.cached_tokens
	//   Anthropic:         usage.cache_read_input_tokens
	// Used as the primary signal for "is prompt caching paying off?" —
	// cached_tokens / tokens_in is the effective cache hit ratio.
	CachedTokens int

	// CacheCreationTokens is the count of input tokens the provider wrote
	// INTO the cache on this turn (only meaningful when the provider
	// distinguishes write vs. read). Anthropic populates this from
	// usage.cache_creation_input_tokens; OpenAI does not surface a separate
	// write counter (implicit caching is opaque).
	CacheCreationTokens int
}

// UsageReporter is the optional capability an engine implements to expose
// cumulative consumption counters. The mediator polls this at session close
// (and may poll periodically for long-running streams) to write per-leg
// totals onto the matching provider_sessions row. Counters are cumulative
// since engine instantiation; engines are single-call-scoped today, so this
// is equivalent to per-call totals.
type UsageReporter interface {
	Usage() Usage
}
