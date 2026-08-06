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

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// cartesiaTTSWSEngine is the WebSocket variant of Cartesia TTS.
//
// Cartesia's WS protocol differs from Deepgram's binary path:
//
//  1. Each text chunk goes out as a full per-utterance JSON request
//     (model_id, transcript, voice, output_format, context_id,
//     continue, language). `continue:true` keeps the context open;
//     `continue:false` with empty transcript closes it cleanly.
//  2. Audio comes back as base64 strings inside `{"type":"chunk",
//     "data":"..."}` JSON frames (similar to ElevenLabs/Inworld).
//  3. Session end is signalled by an explicit `{"type":"done"}` frame
//     — cleaner than ElevenLabs's isFinal-after-EOS gate.
//  4. Context cancellation uses `{"context_id": ..., "cancel": true}`.
//  5. The Cartesia-Version pin lives in the URL query string
//     (`cartesia_version=...`), not a header.
//  6. Auth header is `X-API-Key` (NOT `Authorization: Bearer ...`,
//     unlike the HTTP path).
//
// Ported from the original Rust reference implementation.
//
// Audio format: pcm_s16le 24 kHz, wire-compatible with the HTTP engine
// and the mediator's downstream pipeline.
//
// Provider request id: Cartesia's WS does not return a server-generated
// request_id. The context_id we generate per Stream() call is the
// correlation handle Cartesia indexes their logs by, so we expose it
// via ProviderRequestID() — same approach Inworld will use.
type cartesiaTTSWSEngine struct {
	cfg CartesiaTTSConfig

	providerRequestID atomic.Value // string — context_id, populated on Stream entry

	usageMu sync.Mutex
	chars   int
}

// NewCartesiaTTSWS builds the WebSocket variant of the Cartesia TTS engine.
// The factory in cadence.go picks this over the HTTP variant when
// CadenceTTSConfig.Transport == "ws".
func NewCartesiaTTSWS(cfg CartesiaTTSConfig) StreamingTTSEngine {
	if cfg.URL == "" {
		cfg.URL = "wss://api.cartesia.ai"
	}
	if cfg.Model == "" {
		cfg.Model = "sonic-3"
	}
	return &cartesiaTTSWSEngine{cfg: cfg}
}

func (c *cartesiaTTSWSEngine) Name() string    { return "cartesia-tts" }
func (c *cartesiaTTSWSEngine) SampleRate() int { return 24000 }

func (c *cartesiaTTSWSEngine) Usage() Usage {
	c.usageMu.Lock()
	defer c.usageMu.Unlock()
	return Usage{CharsIn: c.chars}
}

// Synthesize satisfies TTSEngine for back-compat. Streaming engines
// should be reached via Stream(); the legacy path is unsupported here.
func (c *cartesiaTTSWSEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	return nil, fmt.Errorf("cartesia-tts-ws: Synthesize not supported on streaming engine; use Stream")
}

// ProviderRequestID returns the context_id we generated for this
// Stream() call — empty before Stream opens the WS, stable thereafter.
func (c *cartesiaTTSWSEngine) ProviderRequestID() string {
	if v := c.providerRequestID.Load(); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// Stream opens a Cartesia /tts/websocket WebSocket lazily on the first
// TextChunk, sends one full request per chunk (with continue:true),
// closes the context cleanly on textCh close (continue:false), and
// pushes decoded PCM frames to audioCh. Returns nil on graceful end
// (server "done" or close), or a wrapped error on transport failure.
func (c *cartesiaTTSWSEngine) Stream(
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
		OnTTSStreamEnd(c.Name(), status, time.Since(start).Seconds())
	}()

	// Lazy connect — wait for the first chunk OR ctx cancel.
	var first TextChunk
	select {
	case t, ok := <-textCh:
		if !ok {
			return nil
		}
		first = t
	case <-ctx.Done():
		return nil
	}

	if c.cfg.APIKey == "" {
		return fmt.Errorf("cartesia-tts-ws: API key not configured")
	}
	if c.cfg.Voice == "" {
		return fmt.Errorf("cartesia-tts-ws: voice id not configured")
	}

	wsURL, err := buildCartesiaTTSURL(&c.cfg)
	if err != nil {
		return fmt.Errorf("cartesia-tts-ws: build URL: %w", err)
	}

	headers := http.Header{}
	headers.Set("X-API-Key", c.cfg.APIKey)

	dialer := *websocket.DefaultDialer
	dialer.HandshakeTimeout = 10 * time.Second
	dialer.EnableCompression = false

	conn, resp, err := dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		return fmt.Errorf("cartesia-tts-ws: dial: %w", err)
	}
	defer conn.Close()
	if resp != nil {
		resp.Body.Close()
	}

	// context_id — Cartesia's per-stream correlation handle. Generated
	// client-side; Cartesia indexes their logs by it for support
	// tickets. Stamped on ProviderRequestID() and slog'd.
	contextID := uuid.New().String()
	c.providerRequestID.Store(contextID)
	slog.Info("cadence-tts: external request id captured",
		"engine", c.Name(),
		"provider_request_id", contextID,
	)

	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)

	var sendErr, recvErr error

	go func() {
		defer wg.Done()
		sendErr = c.runSender(streamCtx, conn, contextID, first, textCh)
		if sendErr != nil {
			cancel()
		}
	}()

	go func() {
		defer wg.Done()
		defer cancel()
		recvErr = c.runReceiver(streamCtx, conn, audioCh)
	}()

	wg.Wait()

	if recvErr != nil {
		return recvErr
	}
	return sendErr
}

// runSender forwards each TextChunk as a full Cartesia per-chunk
// request (continue:true). On textCh close, emits a closing message
// (continue:false, empty transcript) so Cartesia flushes any remaining
// audio for this context. On ctx cancel, sends a `cancel:true` message
// if any text was already sent.
func (c *cartesiaTTSWSEngine) runSender(
	ctx context.Context,
	conn *websocket.Conn,
	contextID string,
	first TextChunk,
	textCh <-chan TextChunk,
) error {
	sentAny := false
	if err := c.sendChunk(conn, contextID, first, true); err != nil {
		return err
	}
	if strings.TrimSpace(first.Text) != "" {
		sentAny = true
	}

	for {
		select {
		case <-ctx.Done():
			if sentAny {
				cancelMsg, _ := json.Marshal(map[string]interface{}{
					"context_id": contextID,
					"cancel":     true,
				})
				_ = conn.WriteMessage(websocket.TextMessage, cancelMsg)
			}
			return nil

		case chunk, ok := <-textCh:
			if !ok {
				if sentAny {
					// Final flush: empty transcript with continue:false
					// closes the context cleanly so Cartesia drains
					// remaining audio + emits "done".
					_ = c.sendChunk(conn, contextID, TextChunk{Text: ""}, false)
				}
				return nil
			}
			if err := c.sendChunk(conn, contextID, chunk, true); err != nil {
				return err
			}
			if strings.TrimSpace(chunk.Text) != "" {
				sentAny = true
			}
		}
	}
}

// sendChunk emits one Cartesia per-utterance request. Empty/whitespace
// transcripts are dropped on the continue=true path (Cartesia 400s on
// blank transcripts that aren't the closing one).
func (c *cartesiaTTSWSEngine) sendChunk(conn *websocket.Conn, contextID string, chunk TextChunk, cont bool) error {
	text := chunk.Text
	if cont && strings.TrimSpace(text) == "" {
		// Skip whitespace-only chunks during the continue=true loop;
		// only the final close (cont=false) uses an empty transcript.
		return nil
	}
	if text != "" {
		c.usageMu.Lock()
		c.chars += len(text)
		c.usageMu.Unlock()
	}

	msg := map[string]interface{}{
		"model_id":   c.cfg.Model,
		"transcript": text,
		"voice": map[string]interface{}{
			"mode": "id",
			"id":   c.cfg.Voice,
		},
		"output_format": map[string]interface{}{
			"container":   "raw",
			"encoding":    "pcm_s16le",
			"sample_rate": 24000,
		},
		"context_id": contextID,
		"continue":   cont,
		"language":   "en",
	}
	data, _ := json.Marshal(msg)
	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		return fmt.Errorf("ws send chunk: %w", err)
	}
	return nil
}

// runReceiver parses each JSON server message. The protocol shape:
//   - {"type":"chunk","data":"<base64>"} — audio chunk
//   - {"type":"done"} — session end (return cleanly)
//   - {"type":"timestamps"|"flush_done"} — ignore
//   - any frame with an "error" string — surface as Stream error
func (c *cartesiaTTSWSEngine) runReceiver(
	ctx context.Context,
	conn *websocket.Conn,
	audioCh chan<- AudioChunk,
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

		if msgType != websocket.TextMessage {
			continue
		}

		var env struct {
			Type    string `json:"type"`
			Data    string `json:"data,omitempty"`
			Error   string `json:"error,omitempty"`
			Message string `json:"message,omitempty"`
		}
		if json.Unmarshal(data, &env) != nil {
			continue
		}

		// Surface vendor errors. Most error envelopes don't carry a
		// "type", so the explicit error check runs alongside the
		// type-switch below.
		if env.Error != "" {
			return fmt.Errorf("cartesia error: %s (%s)", env.Error, env.Message)
		}

		switch env.Type {
		case "chunk":
			if env.Data == "" {
				continue
			}
			pcm, decErr := base64.StdEncoding.DecodeString(env.Data)
			if decErr != nil {
				slog.Warn("cartesia-tts-ws: failed to decode audio chunk",
					"error", decErr,
				)
				continue
			}
			if len(pcm) == 0 {
				continue
			}
			select {
			case audioCh <- AudioChunk{Data: pcm}:
			case <-ctx.Done():
				return nil
			}

		case "done":
			return nil

		case "timestamps", "flush_done":
			// Informational acks — ignore. Stream returns when "done"
			// arrives or the connection closes.
		}
	}
}

// buildCartesiaTTSURL composes the WS URL with the cartesia_version
// query param pinned. Accepts http(s)/ws(s) input; normalises to wss.
func buildCartesiaTTSURL(cfg *CartesiaTTSConfig) (string, error) {
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
	u.Path = "/tts/websocket"
	q := u.Query()
	q.Set("cartesia_version", cartesiaAPIVersion)
	u.RawQuery = q.Encode()
	return u.String(), nil
}
