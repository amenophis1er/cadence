package engines

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// cartesiaAPIVersion pins the Cartesia-Version header. Cartesia's API
// versions everything via this header (similar to how Stripe versions
// its REST API), so without it the server defaults to whatever the
// account-level pin is — which can shift under us. Pinning the version
// in code keeps behaviour deterministic across deploys.
const cartesiaAPIVersion = "2026-03-01"

// CartesiaTTSConfig configures Cartesia's TTS /tts/bytes endpoint.
// Cartesia returns raw audio bytes in the requested format directly in
// the HTTP body (no JSON envelope, no SSE), so this engine is a thin
// wrapper around http.Do — almost the same shape as Deepgram TTS.
//
// We request raw PCM16 LE @ 24 kHz to slot into the mediator's
// pcm24kToMulaw8k path unchanged. Cartesia natively supports
// `pcm_mulaw` @ 8 kHz, which would skip conversion entirely, but the
// mediator currently hardcodes the 24k→8k decimation; honouring an
// engine-declared output format is a worthwhile mediator refactor for
// later (same deferred TODO as the Inworld engine).
type CartesiaTTSConfig struct {
	URL     string // base URL — defaults to https://api.cartesia.ai
	APIKey  string // Bearer token (sk_car_...)
	Model   string // e.g. sonic-3, sonic-2, sonic-turbo
	Voice   string // voice id (UUID-ish)
	Timeout time.Duration
}

// NewCartesiaTTS builds a TTSEngine talking to Cartesia's /tts/bytes.
func NewCartesiaTTS(cfg CartesiaTTSConfig) TTSEngine {
	if cfg.URL == "" {
		cfg.URL = "https://api.cartesia.ai"
	}
	if cfg.Model == "" {
		cfg.Model = "sonic-3"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return &cartesiaTTSEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(cfg.Timeout),
	}
}

type cartesiaTTSEngine struct {
	cfg    CartesiaTTSConfig
	client *http.Client

	usageMu sync.Mutex
	chars   int
}

func (c *cartesiaTTSEngine) Name() string    { return "cartesia-tts" }
func (c *cartesiaTTSEngine) SampleRate() int { return 24000 }

func (c *cartesiaTTSEngine) Usage() Usage {
	c.usageMu.Lock()
	defer c.usageMu.Unlock()
	return Usage{CharsIn: c.chars}
}

type cartesiaTTSRequest struct {
	ModelID      string                  `json:"model_id"`
	Transcript   string                  `json:"transcript"`
	Voice        cartesiaVoiceRef        `json:"voice"`
	OutputFormat cartesiaTTSOutputFormat `json:"output_format"`
	Language     string                  `json:"language,omitempty"`
}

type cartesiaVoiceRef struct {
	Mode string `json:"mode"` // "id"
	ID   string `json:"id"`
}

type cartesiaTTSOutputFormat struct {
	Container  string `json:"container"`   // "raw"
	Encoding   string `json:"encoding"`    // "pcm_s16le"
	SampleRate int    `json:"sample_rate"` // 24000
}

// Synthesize POSTs the transcript and returns the streaming response
// body. Cartesia begins emitting raw PCM bytes as soon as the model
// produces them, so the caller's incremental reads translate directly
// to audio frames going out to the caller.
func (c *cartesiaTTSEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	if text == "" {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}
	if c.cfg.APIKey == "" {
		return nil, fmt.Errorf("cartesia-tts: API key not configured")
	}
	if c.cfg.Voice == "" {
		return nil, fmt.Errorf("cartesia-tts: voice id not configured")
	}

	c.usageMu.Lock()
	c.chars += len(text)
	c.usageMu.Unlock()

	body, err := json.Marshal(cartesiaTTSRequest{
		ModelID:    c.cfg.Model,
		Transcript: text,
		Voice:      cartesiaVoiceRef{Mode: "id", ID: c.cfg.Voice},
		OutputFormat: cartesiaTTSOutputFormat{
			Container:  "raw",
			Encoding:   "pcm_s16le",
			SampleRate: 24000,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("marshal cartesia tts body: %w", err)
	}

	endpoint := c.cfg.URL + "/tts/bytes"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Cartesia-Version", cartesiaAPIVersion)
	req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cartesia tts request: %w", err)
	}
	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("cartesia tts status %d: %s", resp.StatusCode, string(buf))
	}
	return resp.Body, nil
}
