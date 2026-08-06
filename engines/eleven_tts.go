package engines

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"
)

// ElevenTTSConfig configures ElevenLabs's streaming text-to-speech
// endpoint. Of the cloud TTS engines we ship this is the simplest:
// the response is raw audio bytes via chunked HTTP transfer, no JSON
// envelope to unwrap (Inworld), no header to strip (Deepgram), no
// container hint to pass (Cartesia uses an explicit raw container).
//
// We request `pcm_24000` so the mediator's existing pcm24kToMulaw8k
// path works unchanged. ElevenLabs natively supports `ulaw_8000`,
// which would skip conversion entirely; switching is a worthwhile
// mediator refactor for later (same TODO as Cartesia/Inworld).
type ElevenTTSConfig struct {
	URL     string // base URL — defaults to https://api.elevenlabs.io
	APIKey  string // xi-api-key header
	Model   string // e.g. eleven_flash_v2_5, eleven_multilingual_v2
	Voice   string // voice id (alphanumeric, ~20 chars, e.g. JBFqnCBsd6RMkjVDRZzb)
	Timeout time.Duration
}

// NewElevenTTS builds a TTSEngine talking to ElevenLabs's
// /v1/text-to-speech/{voice_id}/stream endpoint.
func NewElevenTTS(cfg ElevenTTSConfig) TTSEngine {
	if cfg.URL == "" {
		cfg.URL = "https://api.elevenlabs.io"
	}
	if cfg.Model == "" {
		cfg.Model = "eleven_flash_v2_5"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return &elevenTTSEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(cfg.Timeout),
	}
}

type elevenTTSEngine struct {
	cfg    ElevenTTSConfig
	client *http.Client

	usageMu sync.Mutex
	chars   int
}

func (e *elevenTTSEngine) Name() string    { return "eleven-tts" }
func (e *elevenTTSEngine) SampleRate() int { return 24000 }

func (e *elevenTTSEngine) Usage() Usage {
	e.usageMu.Lock()
	defer e.usageMu.Unlock()
	return Usage{CharsIn: e.chars}
}

type elevenTTSRequest struct {
	Text    string `json:"text"`
	ModelID string `json:"model_id,omitempty"`
}

// Synthesize POSTs the text and returns the streaming response body.
// ElevenLabs begins emitting raw PCM bytes as soon as the model
// produces them, so the caller's incremental reads translate directly
// to audio frames going out to the caller.
func (e *elevenTTSEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	if text == "" {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}
	if e.cfg.APIKey == "" {
		return nil, fmt.Errorf("eleven-tts: API key not configured")
	}
	if e.cfg.Voice == "" {
		return nil, fmt.Errorf("eleven-tts: voice id not configured")
	}

	e.usageMu.Lock()
	e.chars += len(text)
	e.usageMu.Unlock()

	body, err := json.Marshal(elevenTTSRequest{
		Text:    text,
		ModelID: e.cfg.Model,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal eleven tts body: %w", err)
	}

	// voice_id goes in the URL path; output_format goes in the query
	// string. url.JoinPath handles a trailing slash on cfg.URL
	// without producing `//v1/...`; it also percent-encodes the
	// voice id (ElevenLabs voices are alphanumeric today, but
	// defending against future shapes is cheap).
	endpoint, err := url.JoinPath(e.cfg.URL, "v1", "text-to-speech", e.cfg.Voice, "stream")
	if err != nil {
		return nil, fmt.Errorf("eleven-tts: build URL: %w", err)
	}
	endpoint += "?output_format=pcm_24000"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("xi-api-key", e.cfg.APIKey)
	req.Header.Set("Accept", "audio/pcm")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("eleven tts request: %w", err)
	}
	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("eleven tts status %d: %s", resp.StatusCode, string(buf))
	}
	return resp.Body, nil
}
