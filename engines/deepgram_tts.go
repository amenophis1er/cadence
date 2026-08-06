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

// DeepgramTTSConfig configures a Deepgram /v1/speak HTTP-streaming TTS
// engine. Deepgram returns audio as a chunked HTTP body, which is exactly
// what TTSEngine.Synthesize wants (io.ReadCloser, no buffering needed).
//
// We request linear16 PCM @ 24 kHz mono so the mediator's existing
// pcm24kToMulaw8k conversion path works unchanged. Switching to mu-law
// 8 kHz would skip the conversion step but require the mediator to
// handle multiple input formats — tradeoff deferred for now since both
// Kokoro and Deepgram emit 24 kHz PCM under this default.
type DeepgramTTSConfig struct {
	URL     string // base URL — defaults to https://api.deepgram.com
	APIKey  string // Token-prefixed Authorization header
	Model   string // Aura voice id, e.g. aura-asteria-en, aura-orion-en
	Timeout time.Duration
}

// NewDeepgramTTS builds a TTSEngine talking to Deepgram's /v1/speak.
func NewDeepgramTTS(cfg DeepgramTTSConfig) TTSEngine {
	if cfg.URL == "" {
		cfg.URL = "https://api.deepgram.com"
	}
	if cfg.Model == "" {
		cfg.Model = "aura-asteria-en"
	}
	if cfg.Timeout == 0 {
		// Long enough to cover the start of generation; the body itself
		// streams, so we don't time-bound the chunked transfer.
		cfg.Timeout = 60 * time.Second
	}
	return &deepgramTTSEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(cfg.Timeout),
	}
}

type deepgramTTSEngine struct {
	cfg    DeepgramTTSConfig
	client *http.Client

	usageMu sync.Mutex
	chars   int
}

func (d *deepgramTTSEngine) Name() string    { return "deepgram-tts" }
func (d *deepgramTTSEngine) SampleRate() int { return 24000 }

func (d *deepgramTTSEngine) Usage() Usage {
	d.usageMu.Lock()
	defer d.usageMu.Unlock()
	return Usage{CharsIn: d.chars}
}

// Synthesize POSTs the text to Deepgram and returns the streaming
// response body. Deepgram begins emitting PCM bytes as soon as the model
// produces them, so the caller's incremental reads translate directly to
// audio frames going out to the caller.
func (d *deepgramTTSEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	if text == "" {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}
	if d.cfg.APIKey == "" {
		return nil, fmt.Errorf("deepgram-tts: API key not configured")
	}

	d.usageMu.Lock()
	d.chars += len(text)
	d.usageMu.Unlock()

	// container=none strips the 44-byte WAV/RIFF header Deepgram
	// otherwise prepends to linear16 responses. The mediator's downstream
	// PCM→μ-law converter would otherwise interpret those header bytes
	// as audio samples and emit a brief burst of noise at the start of
	// every sentence — audible as a "click" between AI utterances since
	// the sentence buffer makes one TTS request per sentence flush.
	endpoint := d.cfg.URL + "/v1/speak?model=" + d.cfg.Model + "&encoding=linear16&sample_rate=24000&container=none"

	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return nil, fmt.Errorf("marshal deepgram tts body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Token "+d.cfg.APIKey)

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("deepgram tts request: %w", err)
	}
	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("deepgram tts status %d: %s", resp.StatusCode, string(buf))
	}
	return resp.Body, nil
}
