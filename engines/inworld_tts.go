package engines

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// InworldTTSConfig configures Inworld's streaming /tts/v1/voice:stream
// endpoint. The API returns a stream of JSON envelopes, each carrying a
// base64-encoded audioContent chunk; this engine unwraps that protocol
// and presents the raw PCM bytes through TTSEngine.Synthesize so the
// mediator's existing pcm24kToMulaw8k path works unchanged.
//
// We request audioEncoding="PCM" (NOT LINEAR16). Per Inworld's docs,
// LINEAR16 streams "include the WAV header in every audio chunk" — those
// 44 header bytes would otherwise be interpreted as PCM samples by the
// mediator and emit a rhythmic click at every chunk boundary (same
// failure mode as Deepgram TTS without container=none). PCM gives us
// raw 16-bit signed little-endian samples with no header.
//
// Inworld natively supports MULAW @ 8 kHz (which would skip the 24k→8k
// step entirely), but the mediator currently hardcodes the 24 kHz PCM16
// → mu-law 8 kHz conversion. Requesting PCM @ 24 kHz keeps this engine
// drop-in with Kokoro / Deepgram TTS; switching to native mu-law is a
// worthwhile mediator refactor for later.
type InworldTTSConfig struct {
	URL     string // base URL — defaults to https://api.inworld.ai
	APIKey  string // value goes verbatim after `Authorization: Basic `
	Model   string // e.g. inworld-tts-1.5-max, inworld-tts-1.5-mini
	Voice   string // voice id, e.g. Dennis, Ashley, Alex
	Timeout time.Duration
}

// NewInworldTTS builds a TTSEngine talking to Inworld's streaming TTS.
func NewInworldTTS(cfg InworldTTSConfig) TTSEngine {
	if cfg.URL == "" {
		cfg.URL = "https://api.inworld.ai"
	}
	if cfg.Model == "" {
		cfg.Model = "inworld-tts-1.5-max"
	}
	if cfg.Voice == "" {
		cfg.Voice = "Dennis"
	}
	if cfg.Timeout == 0 {
		// Generous: covers connect + first byte. The body itself
		// streams JSON envelopes so we don't time-bound the chunked
		// transfer.
		cfg.Timeout = 60 * time.Second
	}
	return &inworldTTSEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(cfg.Timeout),
	}
}

type inworldTTSEngine struct {
	cfg    InworldTTSConfig
	client *http.Client

	usageMu sync.Mutex
	chars   int
}

func (i *inworldTTSEngine) Name() string    { return "inworld-tts" }
func (i *inworldTTSEngine) SampleRate() int { return 24000 }

func (i *inworldTTSEngine) Usage() Usage {
	i.usageMu.Lock()
	defer i.usageMu.Unlock()
	return Usage{CharsIn: i.chars}
}

// inworldStreamRequest mirrors the JSON body Inworld's streaming TTS
// endpoint expects. We only set the fields we actually drive; the rest
// (temperature, timestampType, normalization) accept their server-side
// defaults.
type inworldStreamRequest struct {
	Text        string             `json:"text"`
	VoiceID     string             `json:"voiceId"`
	ModelID     string             `json:"modelId"`
	AudioConfig inworldAudioConfig `json:"audioConfig"`
}

type inworldAudioConfig struct {
	AudioEncoding   string `json:"audioEncoding"`
	SampleRateHertz int    `json:"sampleRateHertz"`
}

// inworldStreamEnvelope is the per-chunk JSON object Inworld emits over
// the streaming response. Audio chunks land in result.audioContent;
// terminal chunks carry usage metadata; failures land in error.
type inworldStreamEnvelope struct {
	Result *struct {
		AudioContent string `json:"audioContent"`
	} `json:"result,omitempty"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Synthesize POSTs the text and returns a pipe reader that yields raw
// PCM16 bytes as the JSON envelopes arrive. A goroutine drives the JSON
// decode + base64 unwrap so the caller's incremental Read calls map
// directly to incoming audio chunks.
func (i *inworldTTSEngine) Synthesize(ctx context.Context, text string) (io.ReadCloser, error) {
	if text == "" {
		return io.NopCloser(bytes.NewReader(nil)), nil
	}
	if i.cfg.APIKey == "" {
		return nil, fmt.Errorf("inworld-tts: API key not configured")
	}

	i.usageMu.Lock()
	i.chars += len(text)
	i.usageMu.Unlock()

	body, err := json.Marshal(inworldStreamRequest{
		Text:    text,
		VoiceID: i.cfg.Voice,
		ModelID: i.cfg.Model,
		AudioConfig: inworldAudioConfig{
			// PCM, not LINEAR16: see file-level docstring. LINEAR16
			// prepends a WAV header to every chunk; PCM is raw.
			AudioEncoding:   "PCM",
			SampleRateHertz: 24000,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("marshal inworld tts body: %w", err)
	}

	endpoint := i.cfg.URL + "/tts/v1/voice:stream"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	// Inworld uses `Authorization: Basic <key>` — the API key goes in
	// directly, NOT base64(user:pass). Their convention.
	req.Header.Set("Authorization", "Basic "+i.cfg.APIKey)

	resp, err := i.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("inworld tts request: %w", err)
	}
	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("inworld tts status %d: %s", resp.StatusCode, string(buf))
	}

	pr, pw := io.Pipe()
	// Cancellation contract: this goroutine blocks on dec.Decode
	// reading from resp.Body. Closing the pipe reader (pr.Close) on
	// the consumer side is NOT enough to unblock the read — the
	// read only returns when (a) the next Inworld chunk arrives and
	// the subsequent pw.Write fails, or (b) the request ctx is
	// cancelled, which closes the underlying connection. Callers MUST
	// cancel `ctx` (not just close the returned reader) for prompt
	// goroutine teardown. The mediator's synthesizeAndStream does
	// this via its parent turnCtx defer cancel(); other callers must
	// follow the same pattern.
	go func() {
		defer resp.Body.Close()
		dec := json.NewDecoder(resp.Body)
		// Sequential json.Decoder.Decode calls walk a stream of
		// whitespace-separated JSON values (the format Inworld's
		// gRPC-transcoded streaming endpoint emits) one envelope at
		// a time.
		for {
			var env inworldStreamEnvelope
			if err := dec.Decode(&env); err != nil {
				if err == io.EOF {
					pw.Close()
					return
				}
				pw.CloseWithError(fmt.Errorf("inworld tts decode: %w", err))
				return
			}
			if env.Error != nil {
				pw.CloseWithError(fmt.Errorf("inworld tts: %s (code %d)", env.Error.Message, env.Error.Code))
				return
			}
			if env.Result == nil || env.Result.AudioContent == "" {
				continue
			}
			pcm, err := base64.StdEncoding.DecodeString(env.Result.AudioContent)
			if err != nil {
				pw.CloseWithError(fmt.Errorf("inworld tts base64: %w", err))
				return
			}
			if _, err := pw.Write(pcm); err != nil {
				// Reader closed (barge-in / cancel) — stop draining
				// the upstream and let the deferred body.Close run.
				return
			}
		}
	}()
	return pr, nil
}
