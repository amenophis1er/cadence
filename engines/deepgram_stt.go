package engines

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// DeepgramSTTConfig configures a streaming-STT client against Deepgram's
// /v1/listen websocket. Cadence speaks mu-law 8 kHz natively (the Twilio
// telephony format), so we ask Deepgram to consume the same encoding —
// no resampling on our end, no quality loss from a forced PCM16 round-trip.
//
// Deepgram drives utterance commits via the `endpointing` parameter
// (silence threshold → Results with is_final=true + speech_final=true).
// We deliberately do NOT enable `vad_events` / `utterance_end_ms`: those
// were observed to fire 0 events on telephony streams across multi-turn
// calls, leaving accumulated is_final segments uncommitted until hangup.
// Plain endpointing is reliable on mu-law.
type DeepgramSTTConfig struct {
	URL      string // base ws/wss URL — defaults to wss://api.deepgram.com
	APIKey   string // Deepgram API key (Token-prefixed Authorization header)
	Model    string // e.g. nova-3, nova-2, nova-2-phonecall
	Language string // BCP-47 ("en", "en-US", …); Deepgram defaults to en
	// EndpointingMs is the trailing-silence threshold (ms) Deepgram uses to
	// commit Results.is_final + speech_final=true. Default 300 if zero.
	EndpointingMs int
	// FlushTimeoutMs is the debounce window (ms) the reader waits after the
	// most recent is_final before flushing buffered text as a synthetic
	// final, when speech_final never fires. Default 1200 if zero. See the
	// runReader doc comment for the full rationale and version history.
	FlushTimeoutMs int
	// TraceEnvelopes logs one line per Results envelope: the flags, the
	// transcript's LENGTH (bytes after TrimSpace, not runes), and
	// inter-envelope timing.
	// Off by default — it is
	// per-envelope volume (hundreds per call), meant for diagnosing why a
	// provider's endpointing behaves as it does, not for steady state.
	//
	// It never logs transcript text. That is caller speech, and a diagnostic
	// is not a reason to put it in a log line.
	TraceEnvelopes bool
}

// NewDeepgramSTT builds a streaming STT engine talking to Deepgram.
func NewDeepgramSTT(cfg DeepgramSTTConfig) StreamingSTTEngine {
	if cfg.URL == "" {
		cfg.URL = "wss://api.deepgram.com"
	}
	if cfg.Model == "" {
		cfg.Model = "nova-3"
	}
	if cfg.EndpointingMs == 0 {
		cfg.EndpointingMs = 300
	}
	if cfg.FlushTimeoutMs <= 0 {
		cfg.FlushTimeoutMs = 1200
	}
	return &deepgramSTTEngine{cfg: cfg}
}

type deepgramSTTEngine struct {
	cfg     DeepgramSTTConfig
	session wsSTTSession

	// callSID is included in every diagnostic slog line so per-call triage
	// works under concurrent load. Set via SetCallSID before Start; empty
	// means the engine has not been wired up to a specific call.
	callSID string

	// Per-type envelope counters, written by the reader goroutine, read at
	// Stop() for the session-summary slog. Broken out by type so silent-call
	// triage can distinguish "Deepgram never returned any is_final" from
	// "Deepgram returned is_final but speech_final never fired" — the two
	// failure modes look identical with a single total.
	envelopesReceived  atomic.Int64
	resultsPartial     atomic.Int64
	resultsIsFinal     atomic.Int64
	resultsSpeechFinal atomic.Int64
	speechStartedRecv  atomic.Int64
	utteranceEndRecv   atomic.Int64
	finalsEmitted      atomic.Int64
}

// SetCallSID stamps the Twilio CallSid on this engine and propagates it to
// the embedded session so the wsstt_base log lines also carry it. Idempotent;
// safe to call before Start.
func (d *deepgramSTTEngine) SetCallSID(sid string) {
	d.callSID = sid
	d.session.callSID = sid
}

func (d *deepgramSTTEngine) Name() string { return "deepgram-stt" }

// Transcribe satisfies STTEngine. The streaming path is the only one that
// makes sense for Deepgram — fall back here is a programmer error.
func (d *deepgramSTTEngine) Transcribe(ctx context.Context, pcm16 PCM16Frame) (string, error) {
	return "", fmt.Errorf("deepgram-stt: batch Transcribe not supported; use Start/PushAudio")
}

// Start dials Deepgram's /v1/listen and hands ownership of the conn to
// the embedded session, which spawns the reader/writer goroutines.
func (d *deepgramSTTEngine) Start(ctx context.Context, events chan<- STTEvent) error {
	if d.cfg.APIKey == "" {
		return fmt.Errorf("deepgram-stt: API key not configured")
	}

	wsURL, err := buildDeepgramURL(&d.cfg)
	if err != nil {
		return fmt.Errorf("deepgram-stt: build URL: %w", err)
	}

	headers := http.Header{}
	headers.Set("Authorization", "Token "+d.cfg.APIKey)

	conn, err := dialSTT(ctx, wsURL, headers, 10*time.Second)
	if err != nil {
		return fmt.Errorf("deepgram-stt: %w", err)
	}

	d.session.name = "deepgram-stt"
	if err := d.session.start(ctx, conn, events, d.runReader); err != nil {
		_ = conn.Close()
		return err
	}

	slog.Info("deepgram-stt: session started",
		"url", wsURL,
		"model", d.cfg.Model,
		"language", d.cfg.Language,
		"call_sid", d.callSID,
	)
	return nil
}

func (d *deepgramSTTEngine) PushAudio(frame []byte) { d.session.pushAudio(frame) }

func (d *deepgramSTTEngine) Usage() Usage { return d.session.usage() }

// Stop closes the websocket and emits a one-line per-call STT summary. The
// summary is what makes a silent-call triage tractable post-hoc: a single
// grep on call_sid tells you whether frames flowed, whether the writer
// forwarded them, and whether Deepgram replied — disambiguating "real
// silent caller" from "broken pipeline" without needing to correlate
// timestamps across multiple slog lines that don't share a call_sid tag.
//
// Before closing the WS we send Deepgram's `{"type":"CloseStream"}`
// sentinel so any server-buffered finals (the trailing transcript that
// would otherwise wait for the next endpoint event) are flushed back to
// us. Without this, a customer mid-utterance who hangs up sees their
// last transcript dropped — exactly the silent-call pattern that hid
// CA30f6a2b... and similar tickets.
func (d *deepgramSTTEngine) Stop() error {
	// Send the CloseStream sentinel through writeMessage so it
	// serializes with runWriter's audio-frame writes (gorilla/websocket
	// permits only one writer at a time). Errors are ignored: a closed
	// session means the conn is gone and there's nothing to flush.
	if err := d.session.writeMessage(websocket.TextMessage, []byte(`{"type":"CloseStream"}`)); err == nil {
		// Brief grace period so trailing Results envelopes land on the
		// reader before stop() closes the conn out from under it.
		time.Sleep(150 * time.Millisecond)
	}
	err := d.session.stop()
	slog.Info("deepgram-stt: session summary",
		"call_sid", d.callSID,
		"frames_received", d.session.framesReceived.Load(),
		"frames_forwarded", d.session.framesForwarded.Load(),
		"frames_dropped_writer_full", d.session.framesDropped.Load(),
		"envelopes_received", d.envelopesReceived.Load(),
		"results_partial", d.resultsPartial.Load(),
		"results_is_final", d.resultsIsFinal.Load(),
		"results_speech_final", d.resultsSpeechFinal.Load(),
		"speech_started", d.speechStartedRecv.Load(),
		"utterance_end", d.utteranceEndRecv.Load(),
		"finals_emitted", d.finalsEmitted.Load(),
		"audio_seconds", d.session.usage().AudioSeconds,
	)
	return err
}

// runReader translates Deepgram's response messages into STTEvents.
// Message shapes that matter:
//
//	{"type":"Results", "channel":{"alternatives":[{"transcript":"..."}]},
//	 "is_final": bool, "speech_final": bool}
//	{"type":"SpeechStarted", ...}        ← only with vad_events (we don't enable)
//	{"type":"UtteranceEnd", ...}         ← only with vad_events (we don't enable)
//	{"type":"Metadata", ...}             ← ignored
//
// Flush policy is a hybrid: speech_final (when it fires) is the canonical
// commit signal — fast path, immediate emit. When speech_final does NOT
// fire (Deepgram's telephony VAD is unreliable, observed in production),
// we fall back to a debounce timer (DeepgramSTTConfig.FlushTimeoutMs,
// default 1200 ms) — accumulated is_final text flushes if no new
// is_final arrives within the window.
//
// Why this shape:
//   - v0.2.24 used speech_final-only → customers repeated themselves
//     because is_final segments stacked up unflushed: "Yes. Yes. Hello?"
//   - v0.2.25 used per-is_final flush → utterances with internal pauses
//     fragmented: phone number "678" + (2s pause) + "472 3048" arrived
//     as two separate user turns, AI replied to "678" alone with
//     "isn't 10 digits".
//   - v0.2.26 (this) uses fast-path on speech_final + 2s debounce
//     fallback. Short utterances ("Yes.") commit as soon as Deepgram
//     decides; fragmented utterances coalesce within a 2s window. The
//     repeat-yourself bug stays fixed because the timeout caps the
//     buffer-stuck window at 2s instead of "until the next speech_final
//     ever fires".
//
// On read-error exit (caller hangup, server close, network blip) we
// flush any buffered text as a last-resort final and cancel the timer
// so the callback can't fire after the events channel closes.
func (d *deepgramSTTEngine) runReader(ctx context.Context, conn *websocket.Conn, events chan<- STTEvent, stopCh <-chan struct{}) {
	// firstMessageOnce logs the very first envelope received from
	// Deepgram (any type — Metadata, Results, SpeechStarted, etc.).
	// Distinguishes "Deepgram never replied" from "we read it but
	// dropped it" when triaging silent calls. One-shot per session.
	var firstMessageOnce sync.Once

	var utterance strings.Builder
	// inUtterance gates the synthetic STTSpeechStarted emission, derived
	// from the first non-empty partial of each utterance — the signal
	// barge-in needs, without depending on Deepgram's vad_events (which
	// we disable because of telephony-stream unreliability).
	inUtterance := false

	// flushMu serializes access to utterance / inUtterance / flushTimer
	// between this goroutine and the time.AfterFunc callback. emitSTT
	// is called inside the lock so the deferred shutdown below can wait
	// for any in-flight callback to complete before returning — without
	// that, a callback could emit on the events channel after the session
	// wrapper has closed it (panic).
	flushTimeout := time.Duration(d.cfg.FlushTimeoutMs) * time.Millisecond
	var flushMu sync.Mutex
	var flushTimer *time.Timer
	stopped := false // protected by flushMu

	// lastResultAt is when the most recent is_final result arrived —
	// interim results carry speech too but don't update it, since only
	// finals enter the utterance buffer. It anchors STTEvent.HeldMs — how
	// long this engine held the utterance before committing. Protected by
	// flushMu.
	var lastResultAt time.Time
	// Envelope-trace state (TraceEnvelopes only): sequence number and the
	// arrival of the previous Results envelope, for inter-arrival timing.
	envSeq := 0
	var prevEnvAt time.Time
	// maxSegmentGap is the longest is_final inter-arrival gap within the
	// current utterance — what a shorter flush timeout would have cut. Not
	// the audible pause: see STTEvent.MaxSegmentGapMs. Protected by flushMu.
	var maxSegmentGap time.Duration

	// flushGen invalidates stale debounce callbacks. timer.Stop() cannot
	// cancel a callback that has already fired and is waiting on flushMu —
	// without this, such a callback would flush a freshly re-armed
	// utterance the moment the reader releases the lock (an early commit,
	// mislabeled CommitFlushTimeout with HeldMs≈0). Every re-arm and every
	// commit bumps the generation; a callback that wakes up to a different
	// generation than it was armed with returns without flushing.
	// Protected by flushMu.
	var flushGen uint64

	// flushFinalLocked flushes the accumulated text and emits the final
	// event, attributing WHICH rule committed it. Caller MUST hold flushMu.
	flushFinalLocked := func(policy CommitPolicy) {
		flushGen++
		text := strings.TrimSpace(utterance.String())
		utterance.Reset()
		inUtterance = false
		if flushTimer != nil {
			flushTimer.Stop()
			flushTimer = nil
		}
		if text != "" {
			var heldMs int64
			if !lastResultAt.IsZero() {
				heldMs = time.Since(lastResultAt).Milliseconds()
			}
			emitSTT(events, STTEvent{
				Type:            STTTranscriptFinal,
				Text:            text,
				CommittedBy:     policy,
				HeldMs:          heldMs,
				MaxSegmentGapMs: maxSegmentGap.Milliseconds(),
			})
			d.finalsEmitted.Add(1)
			OnSTTCommit(d.Name(), policy, heldMs)
		}
		lastResultAt = time.Time{}
		maxSegmentGap = 0
	}

	// flushFinal is the self-locking entry point for the session-end path
	// (the debounce timer has its own generation-checked callback above).
	flushFinal := func(policy CommitPolicy) {
		flushMu.Lock()
		defer flushMu.Unlock()
		if stopped {
			return
		}
		flushFinalLocked(policy)
	}

	// armFlushTimer (re)arms the debounce timer. MUST be called with
	// flushMu held.
	armFlushTimer := func() {
		if flushTimer != nil {
			flushTimer.Stop()
		}
		flushGen++
		gen := flushGen
		flushTimer = time.AfterFunc(flushTimeout, func() {
			flushMu.Lock()
			defer flushMu.Unlock()
			if stopped || gen != flushGen {
				return
			}
			flushFinalLocked(CommitFlushTimeout)
		})
	}

	// Defer shutdown: cancel any pending timer and gate further flushes.
	// Acquiring flushMu blocks until any in-flight callback completes,
	// so the session wrapper's defer close(events) cannot race with a
	// post-return emitSTT.
	defer func() {
		flushMu.Lock()
		stopped = true
		if flushTimer != nil {
			flushTimer.Stop()
			flushTimer = nil
		}
		flushMu.Unlock()
	}()

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			// Last-resort flush of any buffered is_final segments before
			// we let the deferred shutdown run.
			flushFinal(CommitSessionEnd)
			if !readErrorIsExpected(ctx, stopCh) {
				emitSTT(events, STTEvent{Type: STTError, Err: err})
			}
			return
		}

		var env struct {
			Type        string `json:"type"`
			IsFinal     bool   `json:"is_final"`
			SpeechFinal bool   `json:"speech_final"`
			Channel     *struct {
				Alternatives []struct {
					Transcript string  `json:"transcript"`
					Confidence float64 `json:"confidence"`
				} `json:"alternatives"`
			} `json:"channel,omitempty"`
		}
		if err := json.Unmarshal(msg, &env); err != nil {
			continue
		}

		d.envelopesReceived.Add(1)
		firstMessageOnce.Do(func() {
			slog.Info("deepgram-stt: first envelope received",
				"type", env.Type,
				"call_sid", d.callSID,
			)
		})

		switch env.Type {
		case "SpeechStarted":
			// Only fires when vad_events is enabled (we disable it). Kept
			// for defensive forward compatibility — counts but does not
			// emit, since our synthetic SpeechStarted (first partial)
			// already drives barge-in.
			d.speechStartedRecv.Add(1)

		case "UtteranceEnd":
			// Same as SpeechStarted: vad_events-derived. Counted only.
			d.utteranceEndRecv.Add(1)

		case "Results":
			if d.cfg.TraceEnvelopes {
				textLen := 0
				if env.Channel != nil && len(env.Channel.Alternatives) > 0 {
					textLen = len(strings.TrimSpace(env.Channel.Alternatives[0].Transcript))
				}
				sincePrev := int64(-1)
				if !prevEnvAt.IsZero() {
					sincePrev = time.Since(prevEnvAt).Milliseconds()
				}
				envSeq++
				prevEnvAt = time.Now()
				// utterance is owned by flushMu (the flush-timer callback
				// resets it from its own goroutine). Log while holding it so
				// the trace line is ordered atomically against timer commits
				// — a `buffered=true` line means the flush had not fired at
				// this envelope's arrival, full stop. Emitting under flushMu
				// matches how commits themselves emit (flushFinalLocked).
				flushMu.Lock()
				slog.Info("deepgram-stt: envelope",
					"call_sid", d.callSID,
					"seq", envSeq,
					"is_final", env.IsFinal,
					"speech_final", env.SpeechFinal,
					"text_len", textLen, // length only — never the caller's words
					"since_prev_ms", sincePrev,
					"buffered", utterance.Len() > 0,
				)
				flushMu.Unlock()
			}
			if env.Channel == nil || len(env.Channel.Alternatives) == 0 {
				continue
			}
			text := strings.TrimSpace(env.Channel.Alternatives[0].Transcript)
			if text == "" {
				// An EMPTY final can still carry the endpointing decision:
				// the words were delivered in earlier segments and this
				// envelope says only "the speaker has stopped". Discarding
				// it — correct for an empty interim, which carries nothing —
				// throws away the provider's own turn boundary and sends the
				// utterance down the flush-timeout path instead: a full
				// debounce of pure added latency on a caller who has already
				// finished. Measured on live telephony traffic, this
				// accounted for roughly a third of all committed turns.
				// Count every empty FINAL, with or without speech_final —
				// invisibility in the session summary is how this bug
				// hid: envelopes_received far exceeded the counted
				// results and nothing said where they went.
				if env.IsFinal {
					d.resultsIsFinal.Add(1)
				}
				if env.IsFinal && env.SpeechFinal {
					d.resultsSpeechFinal.Add(1)
					flushMu.Lock()
					flushFinalLocked(CommitSpeechFinal)
					flushMu.Unlock()
				}
				continue
			}
			// First decoded text in this utterance → emit synthetic
			// SpeechStarted. The mediator uses this to drive barge-in
			// (cancel agent turn, drain outbox, fire UserStartedSpeaking).
			flushMu.Lock()
			if !inUtterance {
				inUtterance = true
				emitSTT(events, STTEvent{Type: STTSpeechStarted})
			}
			if env.IsFinal {
				d.resultsIsFinal.Add(1)
				if env.SpeechFinal {
					d.resultsSpeechFinal.Add(1)
				}
				// Append to the accumulated utterance. Multiple is_final
				// segments inside one breath (long phone numbers, hesitant
				// answers) merge here so the consumer sees one user turn.
				if utterance.Len() > 0 {
					utterance.WriteString(" ")
					// A continuation: the debounce ran this long without
					// firing before the next is_final landed, so any flush
					// timeout shorter than this gap would have split the
					// speaker's sentence here.
					if !lastResultAt.IsZero() {
						if gap := time.Since(lastResultAt); gap > maxSegmentGap {
							maxSegmentGap = gap
						}
					}
				}
				utterance.WriteString(text)
				lastResultAt = time.Now()
				if env.SpeechFinal {
					// Fast path: Deepgram says the user is done talking,
					// commit immediately.
					flushFinalLocked(CommitSpeechFinal)
				} else {
					// Slow path: arm the debounce. If another is_final
					// arrives within flushTimeout, it'll re-arm; otherwise
					// the timer fires and flushes whatever's accumulated.
					armFlushTimer()
				}
			} else {
				d.resultsPartial.Add(1)
				emitSTT(events, STTEvent{Type: STTTranscriptDelta, Text: text})
			}
			flushMu.Unlock()

		case "Metadata", "":
			// ignore

		case "Error":
			emitSTT(events, STTEvent{Type: STTError, Err: fmt.Errorf("deepgram-stt: %s", string(msg))})
			return
		}
	}
}

// buildDeepgramURL constructs the /v1/listen websocket URL with the
// listening parameters baked in as query string. Deepgram's listen API
// configures everything (model, encoding, endpointing) via URL params;
// there's no separate handshake message.
func buildDeepgramURL(cfg *DeepgramSTTConfig) (string, error) {
	u, err := url.Parse(cfg.URL)
	if err != nil {
		return "", err
	}
	switch u.Scheme {
	case "http":
		u.Scheme = "ws"
	case "https":
		u.Scheme = "wss"
	case "ws", "wss":
	default:
		return "", fmt.Errorf("unsupported scheme %q", u.Scheme)
	}
	u.Path = "/v1/listen"

	q := u.Query()
	q.Set("model", cfg.Model)
	q.Set("encoding", "mulaw")
	q.Set("sample_rate", "8000")
	q.Set("channels", "1")
	q.Set("punctuate", "true")
	q.Set("interim_results", "true")
	q.Set("endpointing", fmt.Sprintf("%d", cfg.EndpointingMs))
	if cfg.Language != "" {
		q.Set("language", cfg.Language)
	}
	u.RawQuery = q.Encode()
	return u.String(), nil
}
