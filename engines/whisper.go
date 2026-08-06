package engines

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"sync"
	"time"
)

// WhisperConfig configures a speaches-compatible Whisper STT engine.
// Speaches exposes OpenAI's /v1/audio/transcriptions endpoint.
type WhisperConfig struct {
	URL      string // e.g. http://cadence-speaches:8000/v1/audio/transcriptions
	APIKey   string // optional bearer
	Model    string // e.g. Systran/faster-distil-whisper-large-v3
	Language string // BCP-47, optional
	Timeout  time.Duration
}

type whisperEngine struct {
	cfg    WhisperConfig
	client *http.Client

	usageMu      sync.Mutex
	audioSeconds float64
}

// NewWhisper builds a Whisper STT engine talking to a speaches-compatible
// OpenAI endpoint. The default 60 s timeout is generous enough to cover the
// first cold-start request, when speaches lazy-loads the model.
func NewWhisper(cfg WhisperConfig) STTEngine {
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return &whisperEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(cfg.Timeout),
	}
}

func (w *whisperEngine) Name() string { return "whisper" }

func (w *whisperEngine) Usage() Usage {
	w.usageMu.Lock()
	defer w.usageMu.Unlock()
	return Usage{AudioSeconds: w.audioSeconds}
}

func (w *whisperEngine) Transcribe(ctx context.Context, pcm16 PCM16Frame) (string, error) {
	if len(pcm16) == 0 {
		return "", nil
	}

	w.usageMu.Lock()
	w.audioSeconds += float64(len(pcm16)) / 2 / 16000
	w.usageMu.Unlock()

	wav := pcm16ToWAV(pcm16, 16000, 1)

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)

	fw, err := mw.CreateFormFile("file", "utterance.wav")
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}
	if _, err := fw.Write(wav); err != nil {
		return "", fmt.Errorf("write wav: %w", err)
	}

	if w.cfg.Model != "" {
		_ = mw.WriteField("model", w.cfg.Model)
	}
	if w.cfg.Language != "" {
		_ = mw.WriteField("language", w.cfg.Language)
	}
	_ = mw.WriteField("response_format", "json")

	if err := mw.Close(); err != nil {
		return "", fmt.Errorf("close multipart: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.cfg.URL, body)
	if err != nil {
		return "", fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	if w.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+w.cfg.APIKey)
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("whisper request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("whisper status %d: %s", resp.StatusCode, string(buf))
	}

	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode whisper response: %w", err)
	}
	return out.Text, nil
}

// pcm16ToWAV wraps raw PCM16 little-endian mono samples in a RIFF/WAV header.
// Whisper accepts raw audio formats only via WAV; we encode in-memory.
func pcm16ToWAV(pcm []byte, sampleRate, channels int) []byte {
	const bitsPerSample = 16
	byteRate := sampleRate * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8

	buf := &bytes.Buffer{}
	buf.WriteString("RIFF")
	binary.Write(buf, binary.LittleEndian, uint32(36+len(pcm)))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	binary.Write(buf, binary.LittleEndian, uint32(16)) // PCM chunk size
	binary.Write(buf, binary.LittleEndian, uint16(1))  // PCM format
	binary.Write(buf, binary.LittleEndian, uint16(channels))
	binary.Write(buf, binary.LittleEndian, uint32(sampleRate))
	binary.Write(buf, binary.LittleEndian, uint32(byteRate))
	binary.Write(buf, binary.LittleEndian, uint16(blockAlign))
	binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))
	buf.WriteString("data")
	binary.Write(buf, binary.LittleEndian, uint32(len(pcm)))
	buf.Write(pcm)
	return buf.Bytes()
}
