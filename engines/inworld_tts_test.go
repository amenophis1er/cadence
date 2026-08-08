package engines

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestInworldTTSHTTP_HappyPath verifies the /tts/v1/voice:stream request
// shape — Basic auth with the raw key, PCM @ 24 kHz audioConfig — and that
// a stream of JSON envelopes carrying base64 audioContent is unwrapped
// into the concatenated raw PCM bytes. Envelopes without a result (usage
// metadata) and with empty audioContent must be skipped silently.
func TestInworldTTSHTTP_HappyPath(t *testing.T) {
	chunk1 := []byte{0x01, 0x02, 0x03}
	chunk2 := []byte{0x04, 0x05, 0x06, 0x07}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/tts/v1/voice:stream" {
			t.Errorf("path = %s, want /tts/v1/voice:stream", r.URL.Path)
		}
		// Inworld convention: raw key after "Basic ", not base64(user:pass).
		if got := r.Header.Get("Authorization"); got != "Basic iw-key" {
			t.Errorf("Authorization = %q, want Basic iw-key", got)
		}
		var body inworldStreamRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		if body.Text != "hello inworld" {
			t.Errorf("text = %q, want hello inworld", body.Text)
		}
		if body.VoiceID != "Ashley" {
			t.Errorf("voiceId = %q, want Ashley", body.VoiceID)
		}
		if body.ModelID != "inworld-tts-1.5-max" {
			t.Errorf("modelId = %q, want inworld-tts-1.5-max", body.ModelID)
		}
		if body.AudioConfig.AudioEncoding != "PCM" || body.AudioConfig.SampleRateHertz != 24000 {
			t.Errorf("audioConfig = %+v, want PCM/24000", body.AudioConfig)
		}

		// Whitespace-separated JSON values, the way Inworld's
		// gRPC-transcoded streaming endpoint emits them.
		fmt.Fprintf(w, `{"result":{"audioContent":%q}}`+"\n", base64.StdEncoding.EncodeToString(chunk1))
		fmt.Fprint(w, `{"result":{"audioContent":""}}`+"\n") // skipped
		fmt.Fprint(w, `{"usage":{"characters":13}}`+"\n")    // no result — skipped
		fmt.Fprintf(w, `{"result":{"audioContent":%q}}`+"\n", base64.StdEncoding.EncodeToString(chunk2))
	}))
	defer srv.Close()

	eng := NewInworldTTS(InworldTTSConfig{
		URL:    srv.URL,
		APIKey: "iw-key",
		Voice:  "Ashley",
	})

	if eng.Name() != "inworld-tts" {
		t.Errorf("Name = %q, want inworld-tts", eng.Name())
	}
	if eng.SampleRate() != 24000 {
		t.Errorf("SampleRate = %d, want 24000", eng.SampleRate())
	}

	rc, err := eng.Synthesize(context.Background(), "hello inworld")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer rc.Close()

	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	want := append(append([]byte{}, chunk1...), chunk2...)
	if string(got) != string(want) {
		t.Errorf("audio = %v, want %v", got, want)
	}

	if u := eng.(*inworldTTSEngine).Usage(); u.CharsIn != len("hello inworld") {
		t.Errorf("CharsIn = %d, want %d", u.CharsIn, len("hello inworld"))
	}
}

// TestInworldTTSHTTP_EmptyTextSkipsRequest confirms empty input never
// hits the network.
func TestInworldTTSHTTP_EmptyTextSkipsRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be hit for empty text")
	}))
	defer srv.Close()

	eng := NewInworldTTS(InworldTTSConfig{URL: srv.URL, APIKey: "k"})
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

// TestInworldTTSHTTP_MissingAPIKeyErrors covers the fail-fast guard.
func TestInworldTTSHTTP_MissingAPIKeyErrors(t *testing.T) {
	eng := NewInworldTTS(InworldTTSConfig{URL: "http://unused"})
	if _, err := eng.Synthesize(context.Background(), "hi"); err == nil {
		t.Error("expected error for missing API key")
	}
}

// TestInworldTTSHTTP_ErrorStatus verifies a non-200 surfaces the status
// and body from Synthesize itself (before any stream is returned).
func TestInworldTTSHTTP_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, `{"message":"unauthorized"}`)
	}))
	defer srv.Close()

	eng := NewInworldTTS(InworldTTSConfig{URL: srv.URL, APIKey: "bad"})
	_, err := eng.Synthesize(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error on 401")
	}
	if !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "unauthorized") {
		t.Errorf("error should carry status and body, got: %v", err)
	}
}

// TestInworldTTSHTTP_ErrorEnvelope verifies an in-stream error envelope
// (Inworld reports mid-stream failures inside a 200 body) surfaces as a
// read error carrying the vendor message and code.
func TestInworldTTSHTTP_ErrorEnvelope(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"result":{"audioContent":%q}}`+"\n", base64.StdEncoding.EncodeToString([]byte{0x01}))
		fmt.Fprint(w, `{"error":{"code":8,"message":"quota exceeded"}}`+"\n")
	}))
	defer srv.Close()

	eng := NewInworldTTS(InworldTTSConfig{URL: srv.URL, APIKey: "k"})
	rc, err := eng.Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer rc.Close()

	_, err = io.ReadAll(rc)
	if err == nil {
		t.Fatal("expected error from error envelope")
	}
	if !strings.Contains(err.Error(), "quota exceeded") || !strings.Contains(err.Error(), "code 8") {
		t.Errorf("error should carry vendor message and code, got: %v", err)
	}
}

// TestInworldTTSHTTP_BadBase64 verifies invalid base64 in audioContent
// surfaces as a read error rather than corrupt audio.
func TestInworldTTSHTTP_BadBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"result":{"audioContent":"!!!not-base64!!!"}}`+"\n")
	}))
	defer srv.Close()

	eng := NewInworldTTS(InworldTTSConfig{URL: srv.URL, APIKey: "k"})
	rc, err := eng.Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer rc.Close()

	if _, err := io.ReadAll(rc); err == nil || !strings.Contains(err.Error(), "base64") {
		t.Errorf("expected base64 decode error, got: %v", err)
	}
}

// TestInworldTTSHTTP_MalformedJSON verifies a truncated / invalid JSON
// stream surfaces as a decode error on the reader.
func TestInworldTTSHTTP_MalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"result":{"audioContent":`) // cut off mid-envelope
	}))
	defer srv.Close()

	eng := NewInworldTTS(InworldTTSConfig{URL: srv.URL, APIKey: "k"})
	rc, err := eng.Synthesize(context.Background(), "hi")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer rc.Close()

	if _, err := io.ReadAll(rc); err == nil || !strings.Contains(err.Error(), "decode") {
		t.Errorf("expected decode error, got: %v", err)
	}
}

// TestInworldTTSHTTP_ContextCancellation covers both cancellation points:
// the initial request (server never responds) and mid-stream (documented
// contract: callers must cancel ctx, not just close the reader, to
// unblock the decode goroutine).
func TestInworldTTSHTTP_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Drain the body so the server watches for client disconnect
		// (it only cancels r.Context() once the request body is
		// consumed); otherwise the deferred srv.Close() hangs forever.
		io.Copy(io.Discard, r.Body)
		// One envelope, then stall until the client goes away.
		fmt.Fprintf(w, `{"result":{"audioContent":%q}}`+"\n", base64.StdEncoding.EncodeToString([]byte{0x01, 0x02}))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()

	eng := NewInworldTTS(InworldTTSConfig{URL: srv.URL, APIKey: "k"})

	ctx, cancel := context.WithCancel(context.Background())
	rc, err := eng.Synthesize(ctx, "hi")
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer rc.Close()

	// First chunk must arrive.
	buf := make([]byte, 2)
	if _, err := io.ReadFull(rc, buf); err != nil {
		t.Fatalf("read first chunk: %v", err)
	}

	cancel()

	done := make(chan error, 1)
	go func() {
		_, err := io.ReadAll(rc)
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected error after context cancellation")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("reader did not unblock promptly after context cancellation")
	}
}
