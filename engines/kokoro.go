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

// KokoroConfig configures a speaches-compatible Kokoro TTS engine.
// Speaches exposes OpenAI's /v1/audio/speech endpoint and accepts
// response_format=pcm to return raw 24 kHz PCM16 mono.
type KokoroConfig struct {
	URL     string // e.g. http://cadence-speaches:8000/v1/audio/speech
	APIKey  string // optional bearer
	Model   string // e.g. speaches-ai/Kokoro-82M-v1.0-ONNX
	Voice   string // e.g. af_heart
	Timeout time.Duration
}

type kokoroEngine struct {
	cfg    KokoroConfig
	client *http.Client

	usageMu sync.Mutex
	chars   int
}

// NewKokoro builds a Kokoro TTS engine talking to a speaches-compatible
// OpenAI endpoint. The default 60 s timeout covers cold-start model loads
// plus synthesis of long agent utterances on CPU.
func NewKokoro(cfg KokoroConfig) TTSEngine {
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return &kokoroEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(cfg.Timeout),
	}
}

func (k *kokoroEngine) Name() string    { return "kokoro" }
func (k *kokoroEngine) SampleRate() int { return 24000 }

func (k *kokoroEngine) Usage() Usage {
	k.usageMu.Lock()
	defer k.usageMu.Unlock()
	return Usage{CharsIn: k.chars}
}

// Synthesize hands back the raw PCM stream from speaches without buffering
// it locally. The caller (mediator.synthesizeAndStream) reads the body
// incrementally, resamples 24 kHz → mu-law 8 kHz on the fly, and pushes
// 20 ms frames into the outbox as they arrive — collapsing time-to-first-
// audio from "full sentence synthesis" to "first chunk on the wire".
//
// io.ReadAll'ing here would defeat the streaming win even if speaches sends
// chunked transfer encoding: nothing leaves the engine until generation is
// done.
func (k *kokoroEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	if text == "" {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}

	k.usageMu.Lock()
	k.chars += len(text)
	k.usageMu.Unlock()

	payload := map[string]interface{}{
		"model":           k.cfg.Model,
		"input":           text,
		"voice":           k.cfg.Voice,
		"response_format": "pcm", // raw PCM16 little-endian mono @ 24 kHz
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal kokoro payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, k.cfg.URL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if k.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+k.cfg.APIKey)
	}

	resp, err := k.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("kokoro request: %w", err)
	}

	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("kokoro status %d: %s", resp.StatusCode, string(buf))
	}

	return resp.Body, nil
}
