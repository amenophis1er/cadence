package engines

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestElevenHTTP_HappyPath verifies the request shape against ElevenLabs's
// streaming endpoint — voice id in the path, output_format in the query,
// xi-api-key header — and that the response body bytes come back verbatim
// through the returned reader.
func TestElevenHTTP_HappyPath(t *testing.T) {
	audio := []byte{0x01, 0x02, 0x03, 0x04, 0xfe, 0xff}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/text-to-speech/voice123/stream" {
			t.Errorf("path = %s, want /v1/text-to-speech/voice123/stream", r.URL.Path)
		}
		if got := r.URL.Query().Get("output_format"); got != "pcm_24000" {
			t.Errorf("output_format = %q, want pcm_24000", got)
		}
		if got := r.Header.Get("xi-api-key"); got != "test-key" {
			t.Errorf("xi-api-key = %q, want test-key", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", got)
		}
		var body struct {
			Text    string `json:"text"`
			ModelID string `json:"model_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		if body.Text != "hello world" {
			t.Errorf("text = %q, want hello world", body.Text)
		}
		if body.ModelID != "eleven_flash_v2_5" {
			t.Errorf("model_id = %q, want eleven_flash_v2_5", body.ModelID)
		}
		w.Write(audio)
	}))
	defer srv.Close()

	eng := NewElevenTTS(ElevenTTSConfig{
		URL:    srv.URL,
		APIKey: "test-key",
		Voice:  "voice123",
	})

	if eng.Name() != "eleven-tts" {
		t.Errorf("Name = %q, want eleven-tts", eng.Name())
	}
	if eng.SampleRate() != 24000 {
		t.Errorf("SampleRate = %d, want 24000", eng.SampleRate())
	}

	rc, err := eng.Synthesize(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer rc.Close()

	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if string(got) != string(audio) {
		t.Errorf("audio = %v, want %v", got, audio)
	}

	if u := eng.(*elevenTTSEngine).Usage(); u.CharsIn != len("hello world") {
		t.Errorf("CharsIn = %d, want %d", u.CharsIn, len("hello world"))
	}
}

// TestElevenHTTP_EmptyTextSkipsRequest confirms empty input short-circuits
// to an empty reader without touching the network.
func TestElevenHTTP_EmptyTextSkipsRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be hit for empty text")
	}))
	defer srv.Close()

	eng := NewElevenTTS(ElevenTTSConfig{URL: srv.URL, APIKey: "k", Voice: "v"})
	rc, err := eng.Synthesize(context.Background(), "")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer rc.Close()
	got, _ := io.ReadAll(rc)
	if len(got) != 0 {
		t.Errorf("expected empty audio, got %d bytes", len(got))
	}
}

// TestElevenHTTP_MissingConfigErrors covers the guard clauses: no API key
// and no voice id both fail fast without a request.
func TestElevenHTTP_MissingConfigErrors(t *testing.T) {
	eng := NewElevenTTS(ElevenTTSConfig{URL: "http://unused", Voice: "v"})
	if _, err := eng.Synthesize(context.Background(), "hi"); err == nil {
		t.Error("expected error for missing API key")
	}

	eng = NewElevenTTS(ElevenTTSConfig{URL: "http://unused", APIKey: "k"})
	if _, err := eng.Synthesize(context.Background(), "hi"); err == nil {
		t.Error("expected error for missing voice")
	}
}

// TestElevenHTTP_ErrorStatus verifies a non-200 response surfaces the
// status code and response body in the error.
func TestElevenHTTP_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, `{"detail":"invalid api key"}`)
	}))
	defer srv.Close()

	eng := NewElevenTTS(ElevenTTSConfig{URL: srv.URL, APIKey: "bad", Voice: "v"})
	_, err := eng.Synthesize(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error on 401")
	}
	if !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("error should carry status and body, got: %v", err)
	}
}

// TestElevenHTTP_ContextCancellation confirms a cancelled context aborts
// a stalled request promptly instead of hanging until the client timeout.
func TestElevenHTTP_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Drain the body first: the server only watches for client
		// disconnect (and cancels r.Context()) once the request body is
		// consumed. Without this the handler — and the deferred
		// srv.Close() — would hang forever.
		io.Copy(io.Discard, r.Body)
		<-r.Context().Done() // hold until the client gives up
	}))
	defer srv.Close()

	eng := NewElevenTTS(ElevenTTSConfig{URL: srv.URL, APIKey: "k", Voice: "v"})

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := eng.Synthesize(ctx, "hi")
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected error from cancelled context")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Synthesize did not return promptly after context cancellation")
	}
}
