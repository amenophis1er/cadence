package engines

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

// CartesiaSTTConfig configures a streaming-STT client against Cartesia's
// /stt/websocket endpoint. Cartesia's "Ink" models accept mu-law 8 kHz
// directly (Twilio telephony format), so we ask for the same encoding —
// no resampling on our end, no quality loss from a forced PCM round-trip.
//
// Cartesia drives its own VAD via the `min_volume` threshold and emits
// is_final=true on each utterance when `max_silence_duration_secs` of
// quiet has elapsed. Local VAD in the mediator stays enabled in
// parallel as a fast barge-in trigger (same pattern as Deepgram STT).
type CartesiaSTTConfig struct {
	URL                    string // base ws/wss URL — defaults to wss://api.cartesia.ai
	APIKey                 string // X-API-Key header
	Model                  string // e.g. ink-whisper
	Language               string // ISO-639-1 (en, es, fr, ...); defaults to en server-side
	MinVolume              float64
	MaxSilenceDurationSecs float64
}

// NewCartesiaSTT builds a streaming STT engine talking to Cartesia.
func NewCartesiaSTT(cfg CartesiaSTTConfig) StreamingSTTEngine {
	if cfg.URL == "" {
		cfg.URL = "wss://api.cartesia.ai"
	}
	if cfg.Model == "" {
		cfg.Model = "ink-whisper"
	}
	if cfg.MaxSilenceDurationSecs == 0 {
		cfg.MaxSilenceDurationSecs = 1.0
	}
	return &cartesiaSTTEngine{cfg: cfg}
}

type cartesiaSTTEngine struct {
	cfg     CartesiaSTTConfig
	session wsSTTSession
}

func (c *cartesiaSTTEngine) Name() string { return "cartesia-stt" }

// Transcribe satisfies STTEngine. The streaming path is the only one
// that makes sense for Cartesia — falling back here is a programmer
// error.
func (c *cartesiaSTTEngine) Transcribe(ctx context.Context, pcm16 PCM16Frame) (string, error) {
	return "", fmt.Errorf("cartesia-stt: batch Transcribe not supported; use Start/PushAudio")
}

// Start dials Cartesia's /stt/websocket and hands ownership of the conn
// to the embedded session, which spawns the reader/writer goroutines.
func (c *cartesiaSTTEngine) Start(ctx context.Context, events chan<- STTEvent) error {
	if c.cfg.APIKey == "" {
		return fmt.Errorf("cartesia-stt: API key not configured")
	}

	wsURL, err := buildCartesiaSTTURL(&c.cfg)
	if err != nil {
		return fmt.Errorf("cartesia-stt: build URL: %w", err)
	}

	headers := http.Header{}
	headers.Set("X-API-Key", c.cfg.APIKey)
	headers.Set("Cartesia-Version", cartesiaAPIVersion)

	conn, err := dialSTT(ctx, wsURL, headers, 10*time.Second)
	if err != nil {
		return fmt.Errorf("cartesia-stt: %w", err)
	}

	c.session.name = "cartesia-stt"
	if err := c.session.start(ctx, conn, events, c.runReader); err != nil {
		_ = conn.Close()
		return err
	}

	slog.Info("cartesia-stt: session started",
		"url", wsURL,
		"model", c.cfg.Model,
		"language", c.cfg.Language,
	)
	return nil
}

func (c *cartesiaSTTEngine) PushAudio(frame []byte) { c.session.pushAudio(frame) }

func (c *cartesiaSTTEngine) Usage() Usage { return c.session.usage() }

// Stop closes the websocket. Cartesia honours a "done" text message to
// flush any pending audio + close cleanly, but a plain WS close
// achieves the same outcome and is safer against partial-frame races
// (the same kind that bit us with WhisperLive).
func (c *cartesiaSTTEngine) Stop() error { return c.session.stop() }

// runReader translates Cartesia's `type:"transcript"` and friends into
// STTEvents. Message shapes that matter:
//
//	{"type":"transcript", "is_final": false, "text": "...", "duration": 0.5, ...}
//	{"type":"transcript", "is_final": true,  "text": "...", ...}
//	{"type":"flush_done", ...}    ← from a finalize control msg, ignored here
//	{"type":"done", ...}          ← session end ack
//	{"type":"error", "title": "...", "message": "...", ...}
//
// We synthesise STTSpeechStarted from the first non-empty partial of
// each utterance, mirroring the Deepgram STT path — Cartesia doesn't
// emit a separate speech-started event. is_final=true commits the
// utterance: emit STTSpeechStopped + STTTranscriptFinal in that order
// so the mediator's userSpeaking gate re-arms before the final text
// kicks off the next LLM turn.
func (c *cartesiaSTTEngine) runReader(ctx context.Context, conn *websocket.Conn, events chan<- STTEvent, stopCh <-chan struct{}) {
	inUtterance := false

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			if !readErrorIsExpected(ctx, stopCh) {
				emitSTT(events, STTEvent{Type: STTError, Err: err})
			}
			return
		}

		var env struct {
			Type    string `json:"type"`
			IsFinal bool   `json:"is_final"`
			Text    string `json:"text"`
			Title   string `json:"title,omitempty"`
			Message string `json:"message,omitempty"`
		}
		if err := json.Unmarshal(msg, &env); err != nil {
			slog.Debug("cartesia-stt: skip non-JSON message", "bytes", len(msg))
			continue
		}

		switch env.Type {
		case "transcript":
			text := strings.TrimSpace(env.Text)
			if text == "" {
				continue
			}
			if !inUtterance {
				inUtterance = true
				slog.Debug("cartesia-stt: synthesised SpeechStarted from first partial")
				emitSTT(events, STTEvent{Type: STTSpeechStarted})
			}
			if env.IsFinal {
				inUtterance = false
				emitSTT(events, STTEvent{Type: STTSpeechStopped})
				emitSTT(events, STTEvent{Type: STTTranscriptFinal, Text: text})
			} else {
				emitSTT(events, STTEvent{Type: STTTranscriptDelta, Text: text})
			}

		case "flush_done", "done", "":
			// Acknowledgments / end-of-session — no STTEvent
			// counterpart. The mediator learns the session is over
			// when the WS closes and the events channel is drained.

		case "error":
			detail := env.Message
			if detail == "" {
				detail = env.Title
			}
			if detail == "" {
				detail = string(msg)
			}
			emitSTT(events, STTEvent{Type: STTError, Err: fmt.Errorf("cartesia-stt: %s", detail)})
			return
		}
	}
}

// buildCartesiaSTTURL constructs the /stt/websocket URL with the
// listening parameters baked in as query string. Cartesia's STT
// configures everything (model, encoding, VAD thresholds) via URL
// params; there's no separate handshake message.
func buildCartesiaSTTURL(cfg *CartesiaSTTConfig) (string, error) {
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
	u.Path = "/stt/websocket"

	q := u.Query()
	q.Set("model", cfg.Model)
	q.Set("encoding", "pcm_mulaw")
	q.Set("sample_rate", "8000")
	q.Set("cartesia_version", cartesiaAPIVersion)
	q.Set("max_silence_duration_secs", strconv.FormatFloat(cfg.MaxSilenceDurationSecs, 'f', -1, 64))
	if cfg.MinVolume > 0 {
		q.Set("min_volume", strconv.FormatFloat(cfg.MinVolume, 'f', -1, 64))
	}
	if cfg.Language != "" {
		q.Set("language", cfg.Language)
	}
	u.RawQuery = q.Encode()
	return u.String(), nil
}
