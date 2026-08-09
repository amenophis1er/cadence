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

// TestCartesiaTTSHTTP_HappyPath verifies the /tts/bytes request shape —
// pinned Cartesia-Version header, Bearer auth, and the nested JSON body
// (voice mode/id, raw pcm_s16le 24 kHz output format) — and that the
// response bytes come back verbatim.
func TestCartesiaTTSHTTP_HappyPath(t *testing.T) {
	audio := []byte{0xaa, 0xbb, 0xcc, 0xdd, 0xee}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/tts/bytes" {
			t.Errorf("path = %s, want /tts/bytes", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk_car_test" {
			t.Errorf("Authorization = %q, want Bearer sk_car_test", got)
		}
		if got := r.Header.Get("Cartesia-Version"); got != cartesiaAPIVersion {
			t.Errorf("Cartesia-Version = %q, want %q", got, cartesiaAPIVersion)
		}
		var body cartesiaTTSRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		if body.ModelID != "sonic-3" {
			t.Errorf("model_id = %q, want sonic-3", body.ModelID)
		}
		if body.Transcript != "hello cartesia" {
			t.Errorf("transcript = %q, want hello cartesia", body.Transcript)
		}
		if body.Voice.Mode != "id" || body.Voice.ID != "voice-uuid" {
			t.Errorf("voice = %+v, want {id voice-uuid}", body.Voice)
		}
		of := body.OutputFormat
		if of.Container != "raw" || of.Encoding != "pcm_s16le" || of.SampleRate != 24000 {
			t.Errorf("output_format = %+v, want raw/pcm_s16le/24000", of)
		}
		w.Write(audio)
	}))
	defer srv.Close()

	eng := NewCartesiaTTS(CartesiaTTSConfig{
		URL:    srv.URL,
		APIKey: "sk_car_test",
		Voice:  "voice-uuid",
	})

	if eng.Name() != "cartesia-tts" {
		t.Errorf("Name = %q, want cartesia-tts", eng.Name())
	}
	if eng.SampleRate() != 24000 {
		t.Errorf("SampleRate = %d, want 24000", eng.SampleRate())
	}

	rc, err := eng.Synthesize(context.Background(), "hello cartesia")
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

	if u := eng.(*cartesiaTTSEngine).Usage(); u.CharsIn != len("hello cartesia") {
		t.Errorf("CharsIn = %d, want %d", u.CharsIn, len("hello cartesia"))
	}
}

// TestCartesiaTTSHTTP_EmptyTextSkipsRequest confirms empty input never
// hits the network.
func TestCartesiaTTSHTTP_EmptyTextSkipsRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be hit for empty text")
	}))
	defer srv.Close()

	eng := NewCartesiaTTS(CartesiaTTSConfig{URL: srv.URL, APIKey: "k", Voice: "v"})
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

// TestCartesiaTTSHTTP_MissingConfigErrors covers the guard clauses.
func TestCartesiaTTSHTTP_MissingConfigErrors(t *testing.T) {
	eng := NewCartesiaTTS(CartesiaTTSConfig{URL: "http://unused", Voice: "v"})
	if _, err := eng.Synthesize(context.Background(), "hi"); err == nil {
		t.Error("expected error for missing API key")
	}

	eng = NewCartesiaTTS(CartesiaTTSConfig{URL: "http://unused", APIKey: "k"})
	if _, err := eng.Synthesize(context.Background(), "hi"); err == nil {
		t.Error("expected error for missing voice")
	}
}

// TestCartesiaTTSHTTP_ErrorStatus verifies a non-200 surfaces the status
// and body in the error.
func TestCartesiaTTSHTTP_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, `{"error":"invalid API key"}`)
	}))
	defer srv.Close()

	eng := NewCartesiaTTS(CartesiaTTSConfig{URL: srv.URL, APIKey: "bad", Voice: "v"})
	_, err := eng.Synthesize(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error on 401")
	}
	if !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "invalid API key") {
		t.Errorf("error should carry status and body, got: %v", err)
	}
}

// TestCartesiaTTSHTTP_ContextCancellation confirms a cancelled context
// aborts a stalled request promptly.
func TestCartesiaTTSHTTP_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Drain the body first: the server only watches for client
		// disconnect (and cancels r.Context()) once the request body is
		// consumed. Without this the handler — and the deferred
		// srv.Close() — would hang forever.
		io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	defer srv.Close()

	eng := NewCartesiaTTS(CartesiaTTSConfig{URL: srv.URL, APIKey: "k", Voice: "v"})

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
