package engines

import (
	"context"
	"encoding/base64"
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

// elevenTTSWSEngine is the WebSocket variant of ElevenLabs TTS.
//
// ElevenLabs differs from Deepgram in three protocol-level ways:
//
//  1. Audio arrives base64-encoded inside JSON messages (not as binary
//     WS frames). The receiver decodes each chunk before pushing.
//  2. The session begins with a BOS message that bundles
//     voice_settings + the first text chunk. Subsequent text-only
//     messages flush each chunk immediately. End-of-input is signalled
//     with `{"text": ""}`.
//  3. The server sends `isFinal: true` after EVERY flushed segment, not
//     just at session end. To distinguish a per-flush ack from a real
//     session-close we only honour isFinal once we've sent EOS — the
//     `eos_sent` flag pattern from the reference implementation.
//
// Ported from the original Rust reference implementation.rs
// (the canonical Phase-2 reference, 312 lines).
//
// Audio format: pcm_24000 (linear16 24 kHz). Wire-compatible with the
// HTTP engine's output and the mediator's fade / pcm24kToMulaw8k path,
// so no downstream changes.
//
// Request_id: ElevenLabs does NOT expose a per-stream correlation
// handle on the WS handshake response or the initial server message
// the way Deepgram does (verified against the reference implementation, which drops
// the response). ProviderRequestID() returns empty for this engine
// for v1; the per-leg `provider_sessions.usage.provider_request_id`
// JSONB will simply be unpopulated for ElevenLabs calls. If
// ElevenLabs ever adds one to the protocol, it slots in here.
type elevenTTSWSEngine struct {
	cfg ElevenTTSConfig

	providerRequestID atomic.Value // string — empty for ElevenLabs (no published handle)

	usageMu sync.Mutex
	chars   int
}

// NewElevenTTSWS builds the WebSocket variant of the ElevenLabs TTS engine.
// The factory in cadence.go picks this over the HTTP variant when
// CadenceTTSConfig.Transport == "ws".
func NewElevenTTSWS(cfg ElevenTTSConfig) StreamingTTSEngine {
	if cfg.URL == "" {
		cfg.URL = "wss://api.elevenlabs.io"
	}
	if cfg.Model == "" {
		cfg.Model = "eleven_flash_v2_5"
	}
	return &elevenTTSWSEngine{cfg: cfg}
}

func (e *elevenTTSWSEngine) Name() string    { return "eleven-tts" }
func (e *elevenTTSWSEngine) SampleRate() int { return 24000 }

func (e *elevenTTSWSEngine) Usage() Usage {
	e.usageMu.Lock()
	defer e.usageMu.Unlock()
	return Usage{CharsIn: e.chars}
}

// Synthesize satisfies TTSEngine for back-compat. Streaming engines
// should be reached via Stream(); the legacy path is unsupported here.
func (e *elevenTTSWSEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	return nil, fmt.Errorf("eleven-tts-ws: Synthesize not supported on streaming engine; use Stream")
}

// ProviderRequestID returns "" for ElevenLabs — no published handle.
func (e *elevenTTSWSEngine) ProviderRequestID() string {
	if v := e.providerRequestID.Load(); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// Stream opens an ElevenLabs stream-input WebSocket lazily on the
// first TextChunk, sends the BOS message bundling voice_settings + the
// first text, forwards subsequent chunks as flushed text messages,
// signals end-of-input with `{"text": ""}` on textCh close or
// ctx.Done(), and pushes decoded PCM frames to audioCh.
func (e *elevenTTSWSEngine) Stream(
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
		OnTTSStreamEnd(e.Name(), status, time.Since(start).Seconds())
	}()

	// Lazy connect — wait for the first chunk OR ctx cancel.
	var first TextChunk
	select {
	case t, ok := <-textCh:
		if !ok {
			return nil // no text to speak
		}
		first = t
	case <-ctx.Done():
		return nil
	}

	if e.cfg.APIKey == "" {
		return fmt.Errorf("eleven-tts-ws: API key not configured")
	}
	if e.cfg.Voice == "" {
		return fmt.Errorf("eleven-tts-ws: voice id not configured")
	}

	wsURL, err := buildElevenTTSURL(&e.cfg)
	if err != nil {
		return fmt.Errorf("eleven-tts-ws: build URL: %w", err)
	}

	headers := http.Header{}
	headers.Set("xi-api-key", e.cfg.APIKey)

	dialer := *websocket.DefaultDialer
	dialer.HandshakeTimeout = 10 * time.Second
	dialer.EnableCompression = false

	conn, resp, err := dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		return fmt.Errorf("eleven-tts-ws: dial: %w", err)
	}
	defer conn.Close()

	if resp != nil {
		resp.Body.Close()
	}

	// eosSent gates the isFinal-as-session-end check. ElevenLabs sends
	// isFinal after each flushed segment, so we ignore it until we've
	// sent the EOS sentinel ourselves.
	var eosSent atomic.Bool

	// Asymmetric cancel matching the Deepgram pattern: receiver always
	// cancels on exit (so a stuck sender unblocks); sender cancels only
	// on error so the success path lets the receiver drain ElevenLabs's
	// in-flight audio after EOS.
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)

	var sendErr, recvErr error

	go func() {
		defer wg.Done()
		sendErr = e.runSender(streamCtx, conn, first, textCh, &eosSent)
		if sendErr != nil {
			cancel()
		}
	}()

	go func() {
		defer wg.Done()
		defer cancel()
		recvErr = e.runReceiver(streamCtx, conn, audioCh, &eosSent)
	}()

	wg.Wait()

	if recvErr != nil {
		return recvErr
	}
	return sendErr
}

// runSender forwards TextChunks as JSON messages over the WS until the
// channel closes or the context is cancelled. The first chunk is bundled
// with voice_settings as the BOS message; subsequent chunks are
// `{"text": ..., "flush": true}` per the vendor protocol. Tracks chars
// for cost attribution and sets eosSent before returning.
func (e *elevenTTSWSEngine) runSender(
	ctx context.Context,
	conn *websocket.Conn,
	first TextChunk,
	textCh <-chan TextChunk,
	eosSent *atomic.Bool,
) error {
	// BOS: first text + voice_settings + flush.
	if err := e.sendBOS(conn, first); err != nil {
		return err
	}

	for {
		select {
		case <-ctx.Done():
			// Send EOS so ElevenLabs flushes final audio + emits its
			// real isFinal; receiver will then exit cleanly.
			_ = e.sendEOS(conn)
			eosSent.Store(true)
			return nil

		case chunk, ok := <-textCh:
			if !ok {
				_ = e.sendEOS(conn)
				eosSent.Store(true)
				return nil
			}
			if err := e.sendChunk(conn, chunk); err != nil {
				return err
			}
		}
	}
}

// sendBOS emits the beginning-of-stream message: first text + voice
// settings + flush. We always flush so audio starts on the first chunk
// rather than waiting for the next.
func (e *elevenTTSWSEngine) sendBOS(conn *websocket.Conn, first TextChunk) error {
	if first.Text != "" {
		e.usageMu.Lock()
		e.chars += len(first.Text)
		e.usageMu.Unlock()
	}
	bos := map[string]interface{}{
		"text": first.Text,
		"voice_settings": map[string]interface{}{
			"stability":        0.5,
			"similarity_boost": 0.75,
		},
		"flush": true,
	}
	data, _ := json.Marshal(bos)
	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		return fmt.Errorf("ws send BOS: %w", err)
	}
	return nil
}

// sendChunk emits a per-chunk text message with flush=true so audio
// starts streaming as soon as ElevenLabs has it.
func (e *elevenTTSWSEngine) sendChunk(conn *websocket.Conn, chunk TextChunk) error {
	if chunk.Text == "" {
		return nil
	}
	e.usageMu.Lock()
	e.chars += len(chunk.Text)
	e.usageMu.Unlock()
	msg, _ := json.Marshal(map[string]interface{}{
		"text":  chunk.Text,
		"flush": true,
	})
	if err := conn.WriteMessage(websocket.TextMessage, msg); err != nil {
		return fmt.Errorf("ws send chunk: %w", err)
	}
	return nil
}

// sendEOS emits the end-of-input sentinel ElevenLabs requires before it
// will close out the stream and emit a real session-final isFinal.
func (e *elevenTTSWSEngine) sendEOS(conn *websocket.Conn) error {
	msg, _ := json.Marshal(map[string]string{"text": ""})
	return conn.WriteMessage(websocket.TextMessage, msg)
}

// runReceiver parses each JSON server message, decodes any base64
// audio, and pushes it to audioCh. Honours session-final isFinal only
// after eosSent is true (matching the vendor protocol).
func (e *elevenTTSWSEngine) runReceiver(
	ctx context.Context,
	conn *websocket.Conn,
	audioCh chan<- AudioChunk,
	eosSent *atomic.Bool,
) error {
	for {
		msgType, data, err := conn.ReadMessage()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				return nil
			}
			return fmt.Errorf("ws read: %w", err)
		}

		// ElevenLabs only sends text messages; ignore unexpected types.
		if msgType != websocket.TextMessage {
			continue
		}

		var env struct {
			Audio   string `json:"audio,omitempty"`
			IsFinal bool   `json:"isFinal,omitempty"`
			Error   string `json:"error,omitempty"`
			Message string `json:"message,omitempty"`
		}
		if err := json.Unmarshal(data, &env); err != nil {
			// Unparseable — skip. ElevenLabs occasionally emits
			// keep-alive frames with no recognised fields.
			continue
		}

		// Surface API errors loudly so configuration mistakes
		// (invalid voice_id, expired key) aren't swallowed.
		if env.Error != "" {
			slog.Error("eleven-tts: API error",
				"engine", e.Name(),
				"error", env.Error,
				"message", env.Message,
			)
		}

		// Decode + emit audio.
		if env.Audio != "" {
			pcm, decErr := base64.StdEncoding.DecodeString(env.Audio)
			if decErr != nil {
				return fmt.Errorf("base64 decode: %w", decErr)
			}
			if len(pcm) > 0 {
				select {
				case audioCh <- AudioChunk{Data: pcm}:
				case <-ctx.Done():
					return nil
				}
			}
		}

		// Treat isFinal as session-end only AFTER we've sent EOS.
		// Pre-EOS isFinal is per-flush bookkeeping we ignore.
		if env.IsFinal && eosSent.Load() {
			return nil
		}
	}
}

// buildElevenTTSURL composes the stream-input WS URL with model_id +
// pcm_24000 output format. Accepts http(s)/ws(s) input and normalises
// to wss://.
func buildElevenTTSURL(cfg *ElevenTTSConfig) (string, error) {
	base := cfg.URL
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
	// Build the path manually with url.JoinPath equivalents — the voice id
	// is already alphanumeric in practice, but pre-percent-encoding via
	// url.PathEscape is cheap insurance.
	u.Path = "/v1/text-to-speech/" + url.PathEscape(cfg.Voice) + "/stream-input"
	q := u.Query()
	q.Set("model_id", cfg.Model)
	q.Set("output_format", "pcm_24000")
	u.RawQuery = q.Encode()
	return u.String(), nil
}
