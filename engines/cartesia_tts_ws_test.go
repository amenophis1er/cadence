package engines

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// mockCartesiaTTSServer spins up an httptest WS server for the
// Cartesia /tts/websocket contract.
func mockCartesiaTTSServer(t *testing.T, handler func(t *testing.T, ws *websocket.Conn)) (wsURL string, cleanup func()) {
	t.Helper()
	upgrader := websocket.Upgrader{
		ReadBufferSize:  4096,
		WriteBufferSize: 4096,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ws, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer ws.Close()
		handler(t, ws)
	}))
	wsURL = "ws" + strings.TrimPrefix(srv.URL, "http")
	return wsURL, srv.Close
}

func newTestCartesiaWSEngine(wsURL string) *cartesiaTTSWSEngine {
	return &cartesiaTTSWSEngine{
		cfg: CartesiaTTSConfig{
			URL:    wsURL,
			APIKey: "test-key",
			Model:  "sonic-3",
			Voice:  "694f9389-aac1-45b6-b726-9d9369183238",
		},
	}
}

// Happy path: caller pushes one chunk + closes textCh; server emits
// audio chunks then a "done"; engine pushes decoded PCM, returns nil.
func TestCartesiaTTSWS_HappyPath(t *testing.T) {
	expected := "Thank you for calling."
	pcm1 := []byte{0x10, 0x20}
	pcm2 := []byte{0x30, 0x40}

	var receivedRequest map[string]interface{}
	var sawClose bool

	wsURL, cleanup := mockCartesiaTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		// 1. Read first request (continue:true)
		_, data, err := ws.ReadMessage()
		if err != nil {
			return
		}
		_ = json.Unmarshal(data, &receivedRequest)

		// Reply with two audio chunks
		_ = ws.WriteJSON(map[string]interface{}{
			"type": "chunk",
			"data": base64.StdEncoding.EncodeToString(pcm1),
		})
		_ = ws.WriteJSON(map[string]interface{}{
			"type": "chunk",
			"data": base64.StdEncoding.EncodeToString(pcm2),
		})

		// 2. Read close request (continue:false)
		_, data, err = ws.ReadMessage()
		if err != nil {
			return
		}
		var closeMsg map[string]interface{}
		_ = json.Unmarshal(data, &closeMsg)
		if cont, ok := closeMsg["continue"].(bool); ok && !cont {
			sawClose = true
		}

		// Send done
		_ = ws.WriteJSON(map[string]interface{}{"type": "done"})
	})
	defer cleanup()

	eng := newTestCartesiaWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 4)
	textCh <- TextChunk{Text: expected, IsSentenceEnd: true}
	close(textCh)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	streamErr := make(chan error, 1)
	go func() { streamErr <- eng.Stream(ctx, textCh, audioCh) }()

	var got [][]byte
	for chunk := range audioCh {
		got = append(got, chunk.Data)
	}

	if err := <-streamErr; err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	if receivedRequest["transcript"] != expected {
		t.Errorf("transcript = %v, want %q", receivedRequest["transcript"], expected)
	}
	if cont, ok := receivedRequest["continue"].(bool); !ok || !cont {
		t.Errorf("first request continue should be true, got %v", receivedRequest["continue"])
	}
	if voice, ok := receivedRequest["voice"].(map[string]interface{}); !ok || voice["id"] != "694f9389-aac1-45b6-b726-9d9369183238" {
		t.Errorf("voice = %v, want id 694f...", receivedRequest["voice"])
	}
	if !sawClose {
		t.Error("server did not see closing message (continue:false)")
	}

	if len(got) != 2 {
		t.Fatalf("got %d audio chunks, want 2", len(got))
	}
	if string(got[0]) != string(pcm1) || string(got[1]) != string(pcm2) {
		t.Errorf("audio mismatch: got %v / %v, want %v / %v", got[0], got[1], pcm1, pcm2)
	}

	if eng.Usage().CharsIn != len(expected) {
		t.Errorf("Usage.CharsIn = %d, want %d", eng.Usage().CharsIn, len(expected))
	}
	if eng.ProviderRequestID() == "" {
		t.Error("ProviderRequestID empty (should be the client-generated context_id)")
	}
}

// Server "done" frame should make the receiver return cleanly.
func TestCartesiaTTSWS_DoneSignalsEnd(t *testing.T) {
	pcm := []byte{0xff}
	wsURL, cleanup := mockCartesiaTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		_, _, _ = ws.ReadMessage()
		_ = ws.WriteJSON(map[string]interface{}{
			"type": "chunk",
			"data": base64.StdEncoding.EncodeToString(pcm),
		})
		_ = ws.WriteJSON(map[string]interface{}{"type": "done"})
		// Hold the conn open after done to verify the receiver returns
		// on the "done" message rather than a transport close.
		time.Sleep(500 * time.Millisecond)
	})
	defer cleanup()

	eng := newTestCartesiaWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 2)
	textCh <- TextChunk{Text: "hi"}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	streamErr := make(chan error, 1)
	go func() { streamErr <- eng.Stream(ctx, textCh, audioCh) }()

	gotAudio := false
	for chunk := range audioCh {
		if string(chunk.Data) == string(pcm) {
			gotAudio = true
		}
	}

	close(textCh)
	if err := <-streamErr; err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	if !gotAudio {
		t.Error("did not receive audio chunk before done")
	}
}

// Cancel mid-stream sends a `cancel:true` message if any text was sent.
func TestCartesiaTTSWS_CancellationSendsCancel(t *testing.T) {
	gotCancel := make(chan struct{}, 1)
	wsURL, cleanup := mockCartesiaTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env map[string]interface{}
			if json.Unmarshal(data, &env) == nil {
				if c, ok := env["cancel"].(bool); ok && c {
					select {
					case gotCancel <- struct{}{}:
					default:
					}
					return
				}
			}
		}
	})
	defer cleanup()

	eng := newTestCartesiaWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 2)
	textCh <- TextChunk{Text: "hello"}

	ctx, cancel := context.WithCancel(context.Background())
	streamErr := make(chan error, 1)
	go func() { streamErr <- eng.Stream(ctx, textCh, audioCh) }()

	time.Sleep(100 * time.Millisecond)
	cancel()

	for range audioCh {
	}
	if err := <-streamErr; err != nil {
		t.Fatalf("Stream after cancel returned error: %v", err)
	}

	select {
	case <-gotCancel:
	case <-time.After(500 * time.Millisecond):
		t.Error("server did not receive cancel message after ctx cancel")
	}
}

// Vendor error in server payload should propagate as a Stream error.
func TestCartesiaTTSWS_VendorErrorPropagates(t *testing.T) {
	wsURL, cleanup := mockCartesiaTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		_, _, _ = ws.ReadMessage()
		_ = ws.WriteJSON(map[string]interface{}{
			"error":   "voice_not_found",
			"message": "voice id 694f... is not valid",
		})
	})
	defer cleanup()

	eng := newTestCartesiaWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 2)
	textCh <- TextChunk{Text: "hi"}
	close(textCh)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	streamErr := make(chan error, 1)
	go func() { streamErr <- eng.Stream(ctx, textCh, audioCh) }()

	for range audioCh {
	}
	err := <-streamErr
	if err == nil {
		t.Fatal("expected error for vendor error envelope, got nil")
	}
	if !strings.Contains(err.Error(), "voice_not_found") {
		t.Errorf("error did not contain vendor error string: %v", err)
	}
}

// Whitespace-only chunks should be skipped on the continue=true path
// (Cartesia 400s on blank transcripts).
func TestCartesiaTTSWS_BlankChunksSkipped(t *testing.T) {
	receivedTexts := []string{}
	wsURL, cleanup := mockCartesiaTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env map[string]interface{}
			if json.Unmarshal(data, &env) != nil {
				continue
			}
			if cont, ok := env["continue"].(bool); ok {
				txt, _ := env["transcript"].(string)
				if cont {
					receivedTexts = append(receivedTexts, txt)
				} else {
					_ = ws.WriteJSON(map[string]interface{}{"type": "done"})
					return
				}
			}
		}
	})
	defer cleanup()

	eng := newTestCartesiaWSEngine(wsURL)
	textCh := make(chan TextChunk, 4)
	audioCh := make(chan AudioChunk, 2)
	textCh <- TextChunk{Text: "Real text."}
	textCh <- TextChunk{Text: "   "} // whitespace — must be skipped
	textCh <- TextChunk{Text: "More real text."}
	close(textCh)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	streamErr := make(chan error, 1)
	go func() { streamErr <- eng.Stream(ctx, textCh, audioCh) }()

	for range audioCh {
	}
	if err := <-streamErr; err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	if len(receivedTexts) != 2 {
		t.Fatalf("got %d non-blank chunks delivered, want 2: %v", len(receivedTexts), receivedTexts)
	}
	if receivedTexts[0] != "Real text." || receivedTexts[1] != "More real text." {
		t.Errorf("delivered chunks mismatch: %v", receivedTexts)
	}
}

// Empty input: closing textCh with no chunks must not dial WS.
func TestCartesiaTTSWS_EmptyInputNoConnect(t *testing.T) {
	var dialed bool
	wsURL, cleanup := mockCartesiaTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		dialed = true
	})
	defer cleanup()

	eng := newTestCartesiaWSEngine(wsURL)
	textCh := make(chan TextChunk)
	audioCh := make(chan AudioChunk, 1)
	close(textCh)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	if err := eng.Stream(ctx, textCh, audioCh); err != nil {
		t.Fatalf("Stream on empty input returned error: %v", err)
	}
	if dialed {
		t.Error("engine dialed WS even though textCh closed before any chunk")
	}
}

func TestCartesiaTTSWS_SynthesizeReturnsError(t *testing.T) {
	eng := newTestCartesiaWSEngine("ws://nowhere")
	body, err := eng.Synthesize(context.Background(), "hello")
	if err == nil {
		t.Fatal("Synthesize unexpectedly returned nil error")
	}
	if body != nil {
		t.Error("Synthesize returned a non-nil body alongside the error")
	}
}

func TestBuildCartesiaTTSURL(t *testing.T) {
	cases := []struct {
		in       string
		wantPref string
	}{
		{"https://api.cartesia.ai", "wss://api.cartesia.ai/tts/websocket"},
		{"http://localhost:9100", "ws://localhost:9100/tts/websocket"},
		{"wss://api.cartesia.ai", "wss://api.cartesia.ai/tts/websocket"},
	}
	for _, c := range cases {
		got, err := buildCartesiaTTSURL(&CartesiaTTSConfig{URL: c.in})
		if err != nil {
			t.Fatalf("buildCartesiaTTSURL(%q): %v", c.in, err)
		}
		if !strings.HasPrefix(got, c.wantPref) {
			t.Errorf("URL = %q, want prefix %q", got, c.wantPref)
		}
		if !strings.Contains(got, "cartesia_version=") {
			t.Errorf("URL %q missing cartesia_version", got)
		}
	}
}
