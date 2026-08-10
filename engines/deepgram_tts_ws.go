package engines

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// deepgramTTSWSEngine is the WebSocket variant of Deepgram TTS — one
// warm connection per Stream() call instead of one HTTP request per
// sentence flush. Wire-compatible with the HTTP engine's audio output
// (linear16 PCM @ 24 kHz) so the mediator's existing fade / convert /
// frame logic processes the bytes identically.
//
// Translated from the Rust reference at
// Differences worth knowing about:
//   - Bidirectional cancel via shared sub-context: either goroutine's
//     exit cancels the other, so a writer error doesn't leave a reader
//     blocked on conn.ReadMessage forever (the production-issue pattern
//     wsstt_base hardened against on the STT side).
//   - Lazy WS connect: we wait for the first TextChunk before opening
//     the WS, avoiding an idle connection during LLM TTFT.
//   - Capture dg-request-id from the WS handshake response header
//     before we drop it. Exposed via ProviderRequestID() for support
//     correlation; persisted on the per-leg provider_sessions row.
type deepgramTTSWSEngine struct {
	cfg DeepgramTTSConfig

	// connMu guards conn AND serialises every write to it. Clear() is
	// called from the consumer's goroutine while runSender writes from its
	// own, and gorilla permits only one concurrent writer per connection.
	// conn is non-nil only while a Stream is live.
	connMu sync.Mutex
	conn   *websocket.Conn

	// discarding is raised by Clear and lowered when the server's Cleared
	// confirmation arrives. While raised, the receiver drops audio frames:
	// those were synthesised before the interruption, so playing them is
	// exactly the stale-speech artefact barge-in exists to prevent.
	discarding atomic.Bool

	// providerRequestID holds the captured dg-request-id. Read from
	// multiple goroutines (engine, mediator's RecordLegSessions);
	// atomic.Value lets the writer in Stream() and the readers in
	// ProviderRequestID() coexist without a lock.
	providerRequestID atomic.Value // string

	// usage counters — same fields the HTTP engine reports so the per-leg
	// provider_sessions row stays cost-attribution-correct under either
	// transport.
	usageMu sync.Mutex
	chars   int
}

// NewDeepgramTTSWS builds the WebSocket variant of the Deepgram TTS engine.
// The factory in cadence.go picks this over the HTTP variant when
// CadenceTTSConfig.Transport == "ws".
func NewDeepgramTTSWS(cfg DeepgramTTSConfig) StreamingTTSEngine {
	if cfg.URL == "" {
		// Default to the WS scheme; the dial code below converts http(s)
		// → ws(s) if a caller passes the HTTP URL by mistake.
		cfg.URL = "wss://api.deepgram.com"
	}
	if cfg.Model == "" {
		cfg.Model = "aura-asteria-en"
	}
	return &deepgramTTSWSEngine{cfg: cfg}
}

func (d *deepgramTTSWSEngine) Name() string    { return "deepgram-tts" }
func (d *deepgramTTSWSEngine) SampleRate() int { return 24000 }

func (d *deepgramTTSWSEngine) Usage() Usage {
	d.usageMu.Lock()
	defer d.usageMu.Unlock()
	return Usage{CharsIn: d.chars}
}

// Synthesize satisfies TTSEngine for back-compat with the legacy HTTP
// path. Streaming engines aren't expected to be called this way (the
// mediator type-asserts to StreamingTTSEngine and uses Stream); we
// return an error rather than silently doing the wrong thing.
func (d *deepgramTTSWSEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	return nil, fmt.Errorf("deepgram-tts-ws: Synthesize not supported on streaming engine; use Stream")
}

// ProviderRequestID returns the vendor's dg-request-id captured from the
// WS handshake response header. Empty until Stream() opens the WS.
func (d *deepgramTTSWSEngine) ProviderRequestID() string {
	if v := d.providerRequestID.Load(); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// Stream opens a Deepgram /v1/speak WebSocket lazily on the first
// TextChunk, forwards each chunk as a Speak control message, flushes
// at sentence boundaries, and pushes received PCM frames to audioCh.
//
// Returns nil on graceful end (text channel closed AND server done
// returning audio) or ctx cancellation. Returns a wrapped error on
// any transport / protocol failure. Always closes audioCh before
// returning so the consumer sees end-of-stream.
func (d *deepgramTTSWSEngine) Stream(
	ctx context.Context,
	textCh <-chan TextChunk,
	audioCh chan<- AudioChunk,
) (retErr error) {
	defer close(audioCh)
	start := time.Now()
	defer func() {
		status := "ok"
		if retErr != nil {
			status = "error"
		}
		OnTTSStreamEnd(d.Name(), status, time.Since(start).Seconds())
	}()

	// Lazy connect — wait for the first chunk OR ctx cancel.
	var first TextChunk
	select {
	case t, ok := <-textCh:
		if !ok {
			return nil // no text to speak, no work to do
		}
		first = t
	case <-ctx.Done():
		return nil
	}

	if d.cfg.APIKey == "" {
		return fmt.Errorf("deepgram-tts-ws: API key not configured")
	}

	wsURL, err := buildDeepgramTTSURL(&d.cfg)
	if err != nil {
		return fmt.Errorf("deepgram-tts-ws: build URL: %w", err)
	}

	headers := http.Header{}
	headers.Set("Authorization", "Token "+d.cfg.APIKey)

	// Dial with a 10s timeout — same as the STT side.
	dialer := *websocket.DefaultDialer
	dialer.HandshakeTimeout = 10 * time.Second
	dialer.EnableCompression = false

	conn, resp, err := dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		// Drain + close the response body if dial gave us one even on
		// failure (gorilla returns it on non-101 responses).
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		return fmt.Errorf("deepgram-tts-ws: dial: %w", err)
	}
	defer conn.Close()

	// Publish the connection so Clear() can reach it, and retract it when
	// the stream ends so a late Clear is a no-op rather than a write to a
	// closed socket.
	d.connMu.Lock()
	d.conn = conn
	d.connMu.Unlock()
	defer func() {
		d.connMu.Lock()
		d.conn = nil
		d.connMu.Unlock()
		d.discarding.Store(false)
	}()

	// Capture dg-request-id from the handshake response BEFORE we drop resp.
	if resp != nil {
		reqID := resp.Header.Get("dg-request-id")
		if reqID != "" {
			d.providerRequestID.Store(reqID)
			slog.Info("cadence-tts: external request id captured",
				"engine", d.Name(),
				"provider_request_id", reqID,
			)
		}
		resp.Body.Close()
	}

	// Asymmetric cancel: receiver always cancels on exit so a stuck
	// sender doesn't deadlock the wg.Wait. Sender only cancels on
	// ERROR — on graceful end (textCh closed cleanly + Close sent), the
	// receiver still has server-side audio in flight to drain, so we
	// let it finish on its own. This matches the reference
	// pattern (deepgram.rs:146): "Only cancel on error — on success,
	// let recv finish receiving audio." Without it, the success case
	// races receiver against the cancel and drops late-arriving audio.
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)

	var sendErr, recvErr error

	go func() {
		defer wg.Done()
		sendErr = d.runSender(streamCtx, conn, first, textCh)
		if sendErr != nil {
			cancel()
		}
	}()

	go func() {
		defer wg.Done()
		defer cancel()
		recvErr = d.runReceiver(streamCtx, conn, audioCh)
	}()

	wg.Wait()

	// Recv error takes priority — if recv failed, the call's audio is
	// incomplete regardless of how send finished. If send failed and
	// recv is clean, that's still a failure (we didn't deliver text we
	// were asked to deliver).
	if recvErr != nil {
		return recvErr
	}
	return sendErr
}

// runSender forwards TextChunks as Deepgram Speak/Flush JSON control
// messages until the channel closes or the context is cancelled.
//
// The first chunk is passed in (already consumed by Stream's lazy-connect
// guard) so we don't lose it.
func (d *deepgramTTSWSEngine) runSender(
	ctx context.Context,
	conn *websocket.Conn,
	first TextChunk,
	textCh <-chan TextChunk,
) error {
	if err := d.sendTextChunk(conn, first); err != nil {
		return err
	}

	for {
		select {
		case <-ctx.Done():
			// Best-effort Close sentinel so Deepgram drains its buffer
			// before we tear down the connection.
			_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Close"}`))
			return nil

		case chunk, ok := <-textCh:
			if !ok {
				// Text channel closed — caller is done. Send Close so
				// Deepgram flushes any in-flight audio for the last
				// utterance, then return. The receive side will exit
				// when the WS closes or the server's Flushed/Metadata
				// stream reaches its natural end.
				_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Close"}`))
				return nil
			}
			if err := d.sendTextChunk(conn, chunk); err != nil {
				return err
			}
		}
	}
}

// sendTextChunk emits a Speak message for the chunk and a Flush message
// after a sentence-end-marked chunk. Tracks chars for cost attribution.
func (d *deepgramTTSWSEngine) sendTextChunk(conn *websocket.Conn, chunk TextChunk) error {
	if chunk.Text != "" {
		d.usageMu.Lock()
		d.chars += len(chunk.Text)
		d.usageMu.Unlock()

		speak, _ := json.Marshal(map[string]string{
			"type": "Speak",
			"text": chunk.Text,
		})
		// Share the writer lock with Clear — see connMu.
		d.connMu.Lock()
		err := conn.WriteMessage(websocket.TextMessage, speak)
		d.connMu.Unlock()
		if err != nil {
			return fmt.Errorf("ws send Speak: %w", err)
		}
	}

	if chunk.IsSentenceEnd {
		d.connMu.Lock()
		err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Flush"}`))
		d.connMu.Unlock()
		if err != nil {
			return fmt.Errorf("ws send Flush: %w", err)
		}
	}
	return nil
}

// Clear implements InterruptibleTTSEngine: abandon queued synthesis, keep the
// session warm. Safe to call from any goroutine, and a no-op when no stream
// is live.
func (d *deepgramTTSWSEngine) Clear() error {
	d.connMu.Lock()
	conn := d.conn
	if conn == nil {
		d.connMu.Unlock()
		return nil // no live stream: nothing queued to abandon
	}
	// Raise the gate BEFORE the wire write, so audio arriving between the
	// write and the server's acknowledgement is already being dropped.
	d.discarding.Store(true)
	err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Clear"}`))
	d.connMu.Unlock()
	if err != nil {
		// The gate would otherwise stay raised forever and mute the stream.
		d.discarding.Store(false)
		return fmt.Errorf("ws send Clear: %w", err)
	}
	return nil
}

// runReceiver pushes binary audio frames to audioCh and ignores text
// control messages from the server (Flushed, Metadata, Warning).
// Returns when the WS closes or the context is cancelled.
//
// We do NOT check ctx.Err() before ReadMessage — a fast sender that
// returns and triggers ctx-cancellation could otherwise short-circuit
// the receiver before it drains audio the server already sent. Instead,
// rely on the fact that ctx-cancel triggers conn.Close (via the
// caller's defer) which unblocks ReadMessage with a closed-network
// error, handled below.
func (d *deepgramTTSWSEngine) runReceiver(
	ctx context.Context,
	conn *websocket.Conn,
	audioCh chan<- AudioChunk,
) error {
	for {
		msgType, data, err := conn.ReadMessage()
		if err != nil {
			// Don't surface "expected" errors from a graceful shutdown
			// (ctx cancel triggers conn.Close which unblocks ReadMessage
			// with a use-of-closed-network-connection error). Treat
			// those as clean exit.
			if ctx.Err() != nil {
				return nil
			}
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				return nil
			}
			return fmt.Errorf("ws read: %w", err)
		}

		switch msgType {
		case websocket.BinaryMessage:
			if len(data) == 0 {
				continue
			}
			// Synthesised before an interruption — dropping it here is
			// what spares every consumer from tracking the vendor's
			// in-flight boundary itself.
			if d.discarding.Load() {
				continue
			}
			// Copy the slice — gorilla reuses its read buffer between
			// calls, and the chunk may be held across goroutine
			// boundaries by the consumer.
			out := make([]byte, len(data))
			copy(out, data)
			select {
			case audioCh <- AudioChunk{Data: out}:
			case <-ctx.Done():
				return nil
			}

		case websocket.TextMessage:
			// Control messages: Flushed (per-utterance), Metadata
			// (informational), Warning. Log Warning loudly; ignore the
			// others — Stream returns when the server closes the WS or
			// the text-side closes our half.
			var env struct {
				Type    string `json:"type"`
				WarnMsg string `json:"warn_msg,omitempty"`
				Message string `json:"message,omitempty"`
			}
			if json.Unmarshal(data, &env) == nil && env.Type == "Cleared" {
				// Everything queued before the Clear has now been
				// accounted for; resume forwarding audio.
				d.discarding.Store(false)
				continue
			}
			if json.Unmarshal(data, &env) == nil && env.Type == "Warning" {
				warn := env.WarnMsg
				if warn == "" {
					warn = env.Message
				}
				slog.Warn("cadence: deepgram TTS warning",
					"engine", d.Name(),
					"warning", warn,
				)
			}

		case websocket.CloseMessage:
			return nil
		}
	}
}

// buildDeepgramTTSURL produces the WS URL with our preferred encoding.
// We use linear16 @ 24 kHz to stay wire-compatible with the existing
// HTTP engine — the mediator's audio path is built around 24 kHz PCM
// (fade-in / fade-out windows are sized in 24 kHz samples; pcm24kToMulaw8k
// handles the final conversion). Switching to mulaw 8 kHz directly would
// skip the conversion but break the fade ramps.
func buildDeepgramTTSURL(cfg *DeepgramTTSConfig) (string, error) {
	base := cfg.URL
	// Allow callers to pass an http(s) URL — convert to ws(s).
	switch {
	case strings.HasPrefix(base, "https://"):
		base = "wss://" + strings.TrimPrefix(base, "https://")
	case strings.HasPrefix(base, "http://"):
		base = "ws://" + strings.TrimPrefix(base, "http://")
	}

	u, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	u.Path = "/v1/speak"
	q := u.Query()
	q.Set("model", cfg.Model)
	q.Set("encoding", "linear16")
	q.Set("sample_rate", "24000")
	u.RawQuery = q.Encode()
	return u.String(), nil
}
