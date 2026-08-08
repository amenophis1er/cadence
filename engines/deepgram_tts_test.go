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

// TestDeepgramTTSHTTP_HappyPath verifies the /v1/speak request shape —
// model/encoding/sample_rate/container query params, Token auth header,
// {"text": ...} JSON body — and that the streamed audio bytes come back
// verbatim.
func TestDeepgramTTSHTTP_HappyPath(t *testing.T) {
	audio := []byte{0x10, 0x20, 0x30, 0x40}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/speak" {
			t.Errorf("path = %s, want /v1/speak", r.URL.Path)
		}
		q := r.URL.Query()
		if got := q.Get("model"); got != "aura-orion-en" {
			t.Errorf("model = %q, want aura-orion-en", got)
		}
		if got := q.Get("encoding"); got != "linear16" {
			t.Errorf("encoding = %q, want linear16", got)
		}
		if got := q.Get("sample_rate"); got != "24000" {
			t.Errorf("sample_rate = %q, want 24000", got)
		}
		// container=none is load-bearing: it strips the WAV header that
		// would otherwise click at the start of every sentence.
		if got := q.Get("container"); got != "none" {
			t.Errorf("container = %q, want none", got)
		}
		if got := r.Header.Get("Authorization"); got != "Token dg-key" {
			t.Errorf("Authorization = %q, want Token dg-key", got)
		}
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		if body["text"] != "hello deepgram" {
			t.Errorf("text = %q, want hello deepgram", body["text"])
		}
		w.Write(audio)
	}))
	defer srv.Close()

	eng := NewDeepgramTTS(DeepgramTTSConfig{
		URL:    srv.URL,
		APIKey: "dg-key",
		Model:  "aura-orion-en",
	})

	if eng.Name() != "deepgram-tts" {
		t.Errorf("Name = %q, want deepgram-tts", eng.Name())
	}
	if eng.SampleRate() != 24000 {
		t.Errorf("SampleRate = %d, want 24000", eng.SampleRate())
	}

	rc, err := eng.Synthesize(context.Background(), "hello deepgram")
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

	if u := eng.(*deepgramTTSEngine).Usage(); u.CharsIn != len("hello deepgram") {
		t.Errorf("CharsIn = %d, want %d", u.CharsIn, len("hello deepgram"))
	}
}

// TestDeepgramTTSHTTP_EmptyTextSkipsRequest confirms empty input never
// hits the network.
func TestDeepgramTTSHTTP_EmptyTextSkipsRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be hit for empty text")
	}))
	defer srv.Close()

	eng := NewDeepgramTTS(DeepgramTTSConfig{URL: srv.URL, APIKey: "k"})
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

// TestDeepgramTTSHTTP_MissingAPIKeyErrors covers the fail-fast guard.
func TestDeepgramTTSHTTP_MissingAPIKeyErrors(t *testing.T) {
	eng := NewDeepgramTTS(DeepgramTTSConfig{URL: "http://unused"})
	if _, err := eng.Synthesize(context.Background(), "hi"); err == nil {
		t.Error("expected error for missing API key")
	}
}

// TestDeepgramTTSHTTP_ErrorStatus verifies a non-200 surfaces the status
// and body in the error.
func TestDeepgramTTSHTTP_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		io.WriteString(w, `{"err_msg":"model overloaded"}`)
	}))
	defer srv.Close()

	eng := NewDeepgramTTS(DeepgramTTSConfig{URL: srv.URL, APIKey: "k"})
	_, err := eng.Synthesize(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error on 500")
	}
	if !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "model overloaded") {
		t.Errorf("error should carry status and body, got: %v", err)
	}
}

// TestDeepgramTTSHTTP_ContextCancellation confirms a cancelled context
// aborts a stalled request promptly.
func TestDeepgramTTSHTTP_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Drain the body first: the server only watches for client
		// disconnect (and cancels r.Context()) once the request body is
		// consumed. Without this the handler — and the deferred
		// srv.Close() — would hang forever.
		io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	defer srv.Close()

	eng := NewDeepgramTTS(DeepgramTTSConfig{URL: srv.URL, APIKey: "k"})

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
