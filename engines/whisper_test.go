package engines

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestWhisper_HappyPath verifies the multipart transcription request —
// the WAV-wrapped audio under the "file" part, the model / language /
// response_format fields, Bearer auth — and that the JSON transcript is
// parsed out of the response.
func TestWhisper_HappyPath(t *testing.T) {
	pcm := make([]byte, 3200) // 100 ms @ 16 kHz PCM16 mono
	for i := range pcm {
		pcm[i] = byte(i % 251)
	}
	wantWAV := pcm16ToWAV(pcm, 16000, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer ws-key" {
			t.Errorf("Authorization = %q, want Bearer ws-key", got)
		}
		if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "multipart/form-data") {
			t.Errorf("Content-Type = %q, want multipart/form-data", ct)
		}
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			t.Fatalf("parse multipart: %v", err)
		}
		if got := r.FormValue("model"); got != "Systran/faster-distil-whisper-large-v3" {
			t.Errorf("model = %q", got)
		}
		if got := r.FormValue("language"); got != "en" {
			t.Errorf("language = %q, want en", got)
		}
		if got := r.FormValue("response_format"); got != "json" {
			t.Errorf("response_format = %q, want json", got)
		}
		f, hdr, err := r.FormFile("file")
		if err != nil {
			t.Fatalf("form file: %v", err)
		}
		defer f.Close()
		if hdr.Filename != "utterance.wav" {
			t.Errorf("filename = %q, want utterance.wav", hdr.Filename)
		}
		gotWAV, err := io.ReadAll(f)
		if err != nil {
			t.Fatalf("read file part: %v", err)
		}
		if string(gotWAV) != string(wantWAV) {
			t.Errorf("file part is not the expected WAV wrapping of the PCM (%d vs %d bytes)", len(gotWAV), len(wantWAV))
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"text":"hello there"}`)
	}))
	defer srv.Close()

	eng := NewWhisper(WhisperConfig{
		URL:      srv.URL,
		APIKey:   "ws-key",
		Model:    "Systran/faster-distil-whisper-large-v3",
		Language: "en",
	})

	if eng.Name() != "whisper" {
		t.Errorf("Name = %q, want whisper", eng.Name())
	}

	got, err := eng.Transcribe(context.Background(), pcm)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if got != "hello there" {
		t.Errorf("transcript = %q, want hello there", got)
	}

	wantSec := float64(len(pcm)) / 2 / 16000
	if u := eng.(*whisperEngine).Usage(); u.AudioSeconds != wantSec {
		t.Errorf("AudioSeconds = %v, want %v", u.AudioSeconds, wantSec)
	}
}

// TestWhisper_OptionalFieldsOmitted confirms model/language fields are
// absent from the form when unset, and no Authorization header is sent
// without an API key.
func TestWhisper_OptionalFieldsOmitted(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization = %q, want unset", got)
		}
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			t.Fatalf("parse multipart: %v", err)
		}
		if _, ok := r.MultipartForm.Value["model"]; ok {
			t.Error("model field should be omitted when unset")
		}
		if _, ok := r.MultipartForm.Value["language"]; ok {
			t.Error("language field should be omitted when unset")
		}
		io.WriteString(w, `{"text":"ok"}`)
	}))
	defer srv.Close()

	eng := NewWhisper(WhisperConfig{URL: srv.URL})
	got, err := eng.Transcribe(context.Background(), []byte{0x00, 0x01})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if got != "ok" {
		t.Errorf("transcript = %q, want ok", got)
	}
}

// TestWhisper_EmptyAudioSkipsRequest confirms zero-length audio returns
// an empty transcript without touching the network.
func TestWhisper_EmptyAudioSkipsRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be hit for empty audio")
	}))
	defer srv.Close()

	eng := NewWhisper(WhisperConfig{URL: srv.URL})
	got, err := eng.Transcribe(context.Background(), nil)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if got != "" {
		t.Errorf("transcript = %q, want empty", got)
	}
}

// TestWhisper_ErrorStatus verifies a non-200 surfaces the status and body
// in the error.
func TestWhisper_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		io.WriteString(w, `{"detail":"model load failed"}`)
	}))
	defer srv.Close()

	eng := NewWhisper(WhisperConfig{URL: srv.URL})
	_, err := eng.Transcribe(context.Background(), []byte{0x00, 0x01})
	if err == nil {
		t.Fatal("expected error on 500")
	}
	if !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "model load failed") {
		t.Errorf("error should carry status and body, got: %v", err)
	}
}

// TestWhisper_MalformedJSONResponse verifies a garbage 200 body surfaces
// as a decode error instead of an empty transcript.
func TestWhisper_MalformedJSONResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `not json at all`)
	}))
	defer srv.Close()

	eng := NewWhisper(WhisperConfig{URL: srv.URL})
	_, err := eng.Transcribe(context.Background(), []byte{0x00, 0x01})
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Errorf("expected decode error, got: %v", err)
	}
}

// TestWhisper_ContextCancellation confirms a cancelled context aborts a
// stalled transcription promptly.
func TestWhisper_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Drain the body first: the server only watches for client
		// disconnect (and cancels r.Context()) once the request body is
		// consumed. Without this the handler — and the deferred
		// srv.Close() — would hang forever.
		io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	defer srv.Close()

	eng := NewWhisper(WhisperConfig{URL: srv.URL})

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := eng.Transcribe(ctx, []byte{0x00, 0x01})
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected error from cancelled context")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Transcribe did not return promptly after context cancellation")
	}
}

// TestWhisper_WAVHeader sanity-checks the in-memory RIFF encoder used to
// wrap raw PCM for the multipart upload.
func TestWhisper_WAVHeader(t *testing.T) {
	pcm := []byte{0x01, 0x02, 0x03, 0x04}
	wav := pcm16ToWAV(pcm, 16000, 1)

	if len(wav) != 44+len(pcm) {
		t.Fatalf("wav length = %d, want %d", len(wav), 44+len(pcm))
	}
	if string(wav[0:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		t.Errorf("bad RIFF/WAVE magic: %q %q", wav[0:4], wav[8:12])
	}
	if string(wav[36:40]) != "data" {
		t.Errorf("bad data chunk id: %q", wav[36:40])
	}
	if string(wav[44:]) != string(pcm) {
		t.Errorf("payload mismatch")
	}
}
