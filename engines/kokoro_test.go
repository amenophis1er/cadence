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

// TestKokoro_HappyPath verifies the OpenAI /v1/audio/speech request shape
// — model/input/voice/response_format=pcm JSON body, Bearer auth — and
// that the streamed PCM bytes come back verbatim. Unlike the cloud
// engines, cfg.URL is the full endpoint, not a base URL.
func TestKokoro_HappyPath(t *testing.T) {
	audio := []byte{0x11, 0x22, 0x33, 0x44, 0x55}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/audio/speech" {
			t.Errorf("path = %s, want /v1/audio/speech", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer kk-key" {
			t.Errorf("Authorization = %q, want Bearer kk-key", got)
		}
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		if body["model"] != "speaches-ai/Kokoro-82M-v1.0-ONNX" {
			t.Errorf("model = %q", body["model"])
		}
		if body["input"] != "hello kokoro" {
			t.Errorf("input = %q, want hello kokoro", body["input"])
		}
		if body["voice"] != "af_heart" {
			t.Errorf("voice = %q, want af_heart", body["voice"])
		}
		if body["response_format"] != "pcm" {
			t.Errorf("response_format = %q, want pcm", body["response_format"])
		}
		w.Write(audio)
	}))
	defer srv.Close()

	eng := NewKokoro(KokoroConfig{
		URL:    srv.URL + "/v1/audio/speech",
		APIKey: "kk-key",
		Model:  "speaches-ai/Kokoro-82M-v1.0-ONNX",
		Voice:  "af_heart",
	})

	if eng.Name() != "kokoro" {
		t.Errorf("Name = %q, want kokoro", eng.Name())
	}
	if eng.SampleRate() != 24000 {
		t.Errorf("SampleRate = %d, want 24000", eng.SampleRate())
	}

	rc, err := eng.Synthesize(context.Background(), "hello kokoro")
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

	if u := eng.(*kokoroEngine).Usage(); u.CharsIn != len("hello kokoro") {
		t.Errorf("CharsIn = %d, want %d", u.CharsIn, len("hello kokoro"))
	}
}

// TestKokoro_NoAuthHeaderWhenKeyEmpty confirms the optional bearer is
// omitted entirely for keyless self-hosted deployments.
func TestKokoro_NoAuthHeaderWhenKeyEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization = %q, want unset", got)
		}
		w.Write([]byte{0x00})
	}))
	defer srv.Close()

	eng := NewKokoro(KokoroConfig{URL: srv.URL})
	rc, err := eng.Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	rc.Close()
}

// TestKokoro_EmptyTextSkipsRequest confirms empty input never hits the
// network.
func TestKokoro_EmptyTextSkipsRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be hit for empty text")
	}))
	defer srv.Close()

	eng := NewKokoro(KokoroConfig{URL: srv.URL})
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

// TestKokoro_ErrorStatus verifies a non-200 surfaces the status and body
// in the error.
func TestKokoro_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		io.WriteString(w, `{"detail":"model not loaded"}`)
	}))
	defer srv.Close()

	eng := NewKokoro(KokoroConfig{URL: srv.URL})
	_, err := eng.Synthesize(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error on 500")
	}
	if !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "model not loaded") {
		t.Errorf("error should carry status and body, got: %v", err)
	}
}

// TestKokoro_ContextCancellation confirms a cancelled context aborts a
// stalled request promptly (cold-start model loads can hang for a while;
// barge-in must not wait on them).
func TestKokoro_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Drain the body first: the server only watches for client
		// disconnect (and cancels r.Context()) once the request body is
		// consumed. Without this the handler — and the deferred
		// srv.Close() — would hang forever.
		io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	defer srv.Close()

	eng := NewKokoro(KokoroConfig{URL: srv.URL})

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
