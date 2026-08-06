package engines

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// inworldTTSWSEngine is the WebSocket variant of Inworld TTS. The
// protocol is documented at
// https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech-websocket
//
// Differences from the other WS engines worth knowing:
//
//  1. Auth is a QUERY PARAMETER (`authorization=Basic <key>`), not an
//     HTTP header — the API server reads it from the URL on the WS
//     handshake.
//  2. Top-level message keys (`create`, `send_text`, `flush_context`,
//     `close_context`) instead of a `type` field. Each message also
//     carries a sibling `contextId` field tying it to the active
//     context.
//  3. Audio chunks are base64 inside `result.audioChunk.audioContent`
//     (same family as ElevenLabs/Cartesia, NOT binary frames like
//     Deepgram).
//  4. Session end is signalled by `result.contextClosed` after we send
//     `close_context` — cleaner than ElevenLabs's isFinal-after-EOS.
//  5. Concurrency caps: 20 connections per account, 5 contexts per
//     connection. v1 uses one connection per Stream() call (= per
//     utterance), so we're capped at 20 concurrent calls. Production
//     peak measured at 37 (issue 037 pre-flight) — the assumption is
//     either Inworld is a low share of TTS volume, or we negotiate
//     special bandwidth with their support (precedent: Deepgram). The
//     HTTP engine is retained as a capacity-pressure fallback;
//     hybrid-mode + connection pool are deferred follow-ups.
//
// Audio format: PCM (no header) at 24 kHz. Wire-compatible with the
// existing HTTP engine and the mediator's downstream
// fade / pcm24kToMulaw8k path.
//
// Provider request id: Inworld has no server-generated request_id; the
// `contextId` we generate per Stream() call IS the correlation handle
// they index server-side logs by, so we expose it via
// ProviderRequestID() — same approach Cartesia uses.
type inworldTTSWSEngine struct {
	cfg InworldTTSConfig

	providerRequestID atomic.Value // string — contextId we generated

	usageMu sync.Mutex
	chars   int
}

// NewInworldTTSWS builds the WebSocket variant of the Inworld TTS engine.
// The factory in cadence.go picks this over the HTTP variant when
// CadenceTTSConfig.Transport == "ws".
func NewInworldTTSWS(cfg InworldTTSConfig) StreamingTTSEngine {
	if cfg.URL == "" {
		cfg.URL = "wss://api.inworld.ai"
	}
	if cfg.Model == "" {
		cfg.Model = "inworld-tts-1.5-max"
	}
	if cfg.Voice == "" {
		cfg.Voice = "Dennis"
	}
	return &inworldTTSWSEngine{cfg: cfg}
}

func (i *inworldTTSWSEngine) Name() string    { return "inworld-tts" }
func (i *inworldTTSWSEngine) SampleRate() int { return 24000 }

func (i *inworldTTSWSEngine) Usage() Usage {
	i.usageMu.Lock()
	defer i.usageMu.Unlock()
	return Usage{CharsIn: i.chars}
}

// Synthesize satisfies TTSEngine for back-compat. Streaming engines
// should be reached via Stream().
func (i *inworldTTSWSEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	return nil, fmt.Errorf("inworld-tts-ws: Synthesize not supported on streaming engine; use Stream")
}

// ProviderRequestID returns the client-generated contextId. Empty
// before Stream opens the WS, stable thereafter.
func (i *inworldTTSWSEngine) ProviderRequestID() string {
	if v := i.providerRequestID.Load(); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// Stream opens an Inworld TTS WebSocket lazily on the first TextChunk,
// emits a `create` for the context (with voice + model + audioConfig),
// forwards each chunk as `send_text` with a `flush_context` so audio
// starts flowing, signals session end with `close_context`, and pushes
// decoded PCM frames to audioCh.
func (i *inworldTTSWSEngine) Stream(
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
		OnTTSStreamEnd(i.Name(), status, time.Since(start).Seconds())
	}()

	// Lazy connect — wait for first chunk OR ctx cancel.
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

	if i.cfg.APIKey == "" {
		return fmt.Errorf("inworld-tts-ws: API key not configured")
	}
	if i.cfg.Voice == "" {
		return fmt.Errorf("inworld-tts-ws: voice id not configured")
	}

	wsURL, err := buildInworldTTSURL(&i.cfg)
	if err != nil {
		return fmt.Errorf("inworld-tts-ws: build URL: %w", err)
	}

	dialer := *websocket.DefaultDialer
	dialer.HandshakeTimeout = 10 * time.Second
	dialer.EnableCompression = false

	// No headers — Inworld carries auth in the URL query string.
	conn, resp, err := dialer.DialContext(ctx, wsURL, nil)
	if err != nil {
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		return fmt.Errorf("inworld-tts-ws: dial: %w", err)
	}
	defer conn.Close()
	if resp != nil {
		resp.Body.Close()
	}

	// Generate the contextId — Inworld's correlation handle. Stamped on
	// ProviderRequestID() for support-ticket lookup, slog'd alongside
	// the engine name for grep correlation.
	contextID := uuid.New().String()
	i.providerRequestID.Store(contextID)
	slog.Info("cadence-tts: external request id captured",
		"engine", i.Name(),
		"provider_request_id", contextID,
	)

	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)

	var sendErr, recvErr error

	go func() {
		defer wg.Done()
		sendErr = i.runSender(streamCtx, conn, contextID, first, textCh)
		if sendErr != nil {
			cancel()
		}
	}()

	go func() {
		defer wg.Done()
		defer cancel()
		recvErr = i.runReceiver(streamCtx, conn, contextID, audioCh)
	}()

	wg.Wait()

	if recvErr != nil {
		return recvErr
	}
	return sendErr
}

// runSender emits the `create` for the context followed by a
// `send_text` for each TextChunk. On graceful end (textCh close) sends
// `close_context` so Inworld flushes any in-flight audio + emits
// `result.contextClosed`. On ctx cancel sends `close_context` too —
// it doubles as cancel since Inworld doesn't expose a separate cancel.
func (i *inworldTTSWSEngine) runSender(
	ctx context.Context,
	conn *websocket.Conn,
	contextID string,
	first TextChunk,
	textCh <-chan TextChunk,
) error {
	if err := i.sendCreate(conn, contextID); err != nil {
		return err
	}
	if err := i.sendTextChunk(conn, contextID, first); err != nil {
		return err
	}

	for {
		select {
		case <-ctx.Done():
			_ = i.sendCloseContext(conn, contextID)
			return nil

		case chunk, ok := <-textCh:
			if !ok {
				_ = i.sendCloseContext(conn, contextID)
				return nil
			}
			if err := i.sendTextChunk(conn, contextID, chunk); err != nil {
				return err
			}
		}
	}
}

// sendCreate opens the synthesis context with voice + model +
// audio_config. autoMode lets Inworld decide when to flush its
// server-side buffer for optimal latency vs. quality, matching how
// fusion uses ElevenLabs's flush. Encoding "PCM" (no header) is the
// same choice as the HTTP engine.
func (i *inworldTTSWSEngine) sendCreate(conn *websocket.Conn, contextID string) error {
	msg := map[string]interface{}{
		"create": map[string]interface{}{
			"voiceId": i.cfg.Voice,
			"modelId": i.cfg.Model,
			"audioConfig": map[string]interface{}{
				"audioEncoding":   "PCM",
				"sampleRateHertz": 24000,
			},
			"autoMode": true,
			"language": "en",
		},
		"contextId": contextID,
	}
	data, _ := json.Marshal(msg)
	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		return fmt.Errorf("ws send create: %w", err)
	}
	return nil
}

// sendTextChunk emits a `send_text` message. Empty / whitespace-only
// chunks are dropped to avoid 1000-char limit math on inputs that
// produce no audio. We DON'T flush manually per chunk — autoMode
// handles flushing. The IsSentenceEnd flag is informational.
func (i *inworldTTSWSEngine) sendTextChunk(conn *websocket.Conn, contextID string, chunk TextChunk) error {
	text := strings.TrimSpace(chunk.Text)
	if text == "" {
		return nil
	}
	// Inworld caps text at 1000 chars per send_text. Sentence-buffer
	// upstream gives us shorter chunks in practice; clamp defensively.
	if len(text) > 1000 {
		text = text[:1000]
	}

	i.usageMu.Lock()
	i.chars += len(text)
	i.usageMu.Unlock()

	msg := map[string]interface{}{
		"send_text": map[string]interface{}{
			"text": text,
		},
		"contextId": contextID,
	}
	data, _ := json.Marshal(msg)
	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		return fmt.Errorf("ws send text: %w", err)
	}
	return nil
}

// sendCloseContext signals end of input. Inworld responds with
// `result.contextClosed` after draining any buffered audio.
func (i *inworldTTSWSEngine) sendCloseContext(conn *websocket.Conn, contextID string) error {
	msg := map[string]interface{}{
		"close_context": map[string]interface{}{},
		"contextId":     contextID,
	}
	data, _ := json.Marshal(msg)
	return conn.WriteMessage(websocket.TextMessage, data)
}

// runReceiver parses each JSON server message. Result envelope shape
// (per docs):
//
//	{"result": {"contextId": "...", "audioChunk": {"audioContent": "<b64>", ...}, "status": {...}}}
//	{"result": {"contextId": "...", "contextCreated": {...}}}
//	{"result": {"contextId": "...", "contextClosed": {...}}}
//	{"result": {"contextId": "...", "flushCompleted": {...}}}
//
// Engine returns on `contextClosed` (our requested close) or on a
// non-zero status.code in any envelope.
func (i *inworldTTSWSEngine) runReceiver(
	ctx context.Context,
	conn *websocket.Conn,
	wantContextID string,
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
			Result *struct {
				ContextID  string `json:"contextId,omitempty"`
				AudioChunk *struct {
					AudioContent string `json:"audioContent,omitempty"`
				} `json:"audioChunk,omitempty"`
				ContextCreated *struct{} `json:"contextCreated,omitempty"`
				ContextClosed  *struct{} `json:"contextClosed,omitempty"`
				FlushCompleted *struct{} `json:"flushCompleted,omitempty"`
				Status         *struct {
					Code    int    `json:"code"`
					Message string `json:"message,omitempty"`
				} `json:"status,omitempty"`
			} `json:"result,omitempty"`
		}
		if json.Unmarshal(data, &env) != nil || env.Result == nil {
			continue
		}
		r := env.Result

		// Surface non-zero status codes as Stream-level errors. code=0
		// is "OK" and arrives on every result envelope.
		if r.Status != nil && r.Status.Code != 0 {
			return fmt.Errorf("inworld error: code=%d message=%s",
				r.Status.Code, r.Status.Message)
		}

		// Decode + emit audio.
		if r.AudioChunk != nil && r.AudioChunk.AudioContent != "" {
			pcm, decErr := base64.StdEncoding.DecodeString(r.AudioChunk.AudioContent)
			if decErr != nil {
				slog.Warn("inworld-tts-ws: failed to decode audio chunk",
					"error", decErr,
				)
				continue
			}
			if len(pcm) > 0 {
				select {
				case audioCh <- AudioChunk{Data: pcm}:
				case <-ctx.Done():
					return nil
				}
			}
		}

		// Session end on contextClosed for OUR context. (If multiple
		// contexts ever shared a connection in a future pool design,
		// the contextID match would be load-bearing.)
		if r.ContextClosed != nil && r.ContextID == wantContextID {
			return nil
		}
	}
}

// buildInworldTTSURL composes the streamBidirectional WS URL with the
// auth query param. Accepts http(s)/ws(s) input; normalises to wss.
func buildInworldTTSURL(cfg *InworldTTSConfig) (string, error) {
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
	u.Path = "/tts/v1/voice:streamBidirectional"
	q := u.Query()
	// Inworld carries the API key in the `authorization` query param
	// rather than a header. The "Basic " prefix is part of the value
	// per their docs.
	q.Set("authorization", "Basic "+cfg.APIKey)
	u.RawQuery = q.Encode()
	return u.String(), nil
}
