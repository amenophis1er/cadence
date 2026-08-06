package engines

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// mockDeepgramTTSServer spins up an httptest.Server that upgrades to a
// WebSocket and runs `handler` on the upgraded connection. It also stamps
// the `dg-request-id` header on the upgrade response so request-id capture
// can be exercised.
func mockDeepgramTTSServer(t *testing.T, requestID string, handler func(t *testing.T, ws *websocket.Conn)) (wsURL string, cleanup func()) {
	t.Helper()
	upgrader := websocket.Upgrader{
		ReadBufferSize:  4096,
		WriteBufferSize: 4096,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// gorilla.Upgrade strips custom headers set via w.Header().Set;
		// the only way to inject headers into the 101 handshake response
		// is to pass them as the third arg.
		respHeader := http.Header{}
		if requestID != "" {
			respHeader.Set("dg-request-id", requestID)
		}
		ws, err := upgrader.Upgrade(w, r, respHeader)
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

func newTestEngine(wsURL string) *deepgramTTSWSEngine {
	return &deepgramTTSWSEngine{
		cfg: DeepgramTTSConfig{
			URL:    wsURL,
			APIKey: "test-key",
			Model:  "aura-asteria-en",
		},
	}
}

// Happy path: caller pushes one sentence-end TextChunk + closes textCh;
// server replies with a couple of binary audio frames + a Flushed ack;
// engine pushes audio bytes to audioCh, returns nil on graceful end.
func TestDeepgramTTSWS_HappyPath(t *testing.T) {
	expectedSpeak := "Thank you for calling."
	requestID := "req-12345"
	audioFrame1 := []byte{0x01, 0x02, 0x03, 0x04}
	audioFrame2 := []byte{0x05, 0x06, 0x07, 0x08}

	var sawSpeak, sawFlush, sawClose bool
	wsURL, cleanup := mockDeepgramTTSServer(t, requestID, func(t *testing.T, ws *websocket.Conn) {
		// Read the client's messages and stage server replies.
		// 1) expect Speak with the right text
		// 2) expect Flush
		// 3) write two binary audio frames
		// 4) write a Flushed ack
		// 5) expect Close
		for i := 0; i < 3; i++ {
			mt, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			if mt != websocket.TextMessage {
				t.Errorf("expected text message, got %d", mt)
				continue
			}
			var env map[string]interface{}
			if err := json.Unmarshal(data, &env); err != nil {
				t.Errorf("unmarshal client msg: %v", err)
				continue
			}
			switch env["type"] {
			case "Speak":
				sawSpeak = true
				if env["text"] != expectedSpeak {
					t.Errorf("Speak.text = %q, want %q", env["text"], expectedSpeak)
				}
				// Reply with binary audio frames after Speak.
				_ = ws.WriteMessage(websocket.BinaryMessage, audioFrame1)
				_ = ws.WriteMessage(websocket.BinaryMessage, audioFrame2)
			case "Flush":
				sawFlush = true
				_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Flushed"}`))
			case "Close":
				sawClose = true
				// Real Deepgram closes with a proper close frame after
				// receiving Close — not a bare TCP shutdown. Mirror that
				// so the receiver doesn't see "abnormal closure".
				_ = ws.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
				return
			}
		}
	})
	defer cleanup()

	eng := newTestEngine(wsURL)

	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 8)

	textCh <- TextChunk{Text: expectedSpeak, IsSentenceEnd: true}
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

	if !sawSpeak {
		t.Error("server did not see Speak message")
	}
	if !sawFlush {
		t.Error("server did not see Flush message")
	}
	if !sawClose {
		t.Error("server did not see Close message")
	}

	if len(got) != 2 {
		t.Fatalf("got %d audio chunks, want 2", len(got))
	}
	if string(got[0]) != string(audioFrame1) {
		t.Errorf("chunk 0 = %v, want %v", got[0], audioFrame1)
	}
	if string(got[1]) != string(audioFrame2) {
		t.Errorf("chunk 1 = %v, want %v", got[1], audioFrame2)
	}

	if eng.ProviderRequestID() != requestID {
		t.Errorf("ProviderRequestID() = %q, want %q", eng.ProviderRequestID(), requestID)
	}

	if eng.Usage().CharsIn != len(expectedSpeak) {
		t.Errorf("Usage.CharsIn = %d, want %d", eng.Usage().CharsIn, len(expectedSpeak))
	}
}

// Cancellation: engine should return nil (not an error) when ctx is
// cancelled, audioCh should be closed by Stream's defer, and the server
// should see a Close sentinel.
func TestDeepgramTTSWS_Cancellation(t *testing.T) {
	var serverGotClose bool
	gotCloseCh := make(chan struct{}, 1)

	wsURL, cleanup := mockDeepgramTTSServer(t, "", func(t *testing.T, ws *websocket.Conn) {
		// Read until we see Close or the conn dies.
		for {
			mt, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			if mt == websocket.TextMessage {
				if strings.Contains(string(data), `"Close"`) {
					serverGotClose = true
					select {
					case gotCloseCh <- struct{}{}:
					default:
					}
					return
				}
			}
		}
	})
	defer cleanup()

	eng := newTestEngine(wsURL)

	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 8)

	// Push one chunk so the engine opens the WS, then cancel mid-stream.
	textCh <- TextChunk{Text: "hello", IsSentenceEnd: false}

	ctx, cancel := context.WithCancel(context.Background())

	streamErr := make(chan error, 1)
	go func() { streamErr <- eng.Stream(ctx, textCh, audioCh) }()

	// Give the engine a moment to dial and send the first Speak.
	time.Sleep(100 * time.Millisecond)
	cancel()

	// Drain audioCh to completion.
	for range audioCh {
	}

	select {
	case err := <-streamErr:
		if err != nil {
			t.Fatalf("Stream after cancel returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Stream did not return within 2s of cancel")
	}

	// Wait briefly for the server's read loop to observe Close.
	select {
	case <-gotCloseCh:
	case <-time.After(500 * time.Millisecond):
	}
	if !serverGotClose {
		t.Error("server did not receive Close sentinel after cancel")
	}
}

// Empty input: closing textCh without any chunks should make Stream
// return immediately without dialing the WS.
func TestDeepgramTTSWS_EmptyInputNoConnect(t *testing.T) {
	var dialed bool
	wsURL, cleanup := mockDeepgramTTSServer(t, "", func(t *testing.T, ws *websocket.Conn) {
		dialed = true
	})
	defer cleanup()

	eng := newTestEngine(wsURL)
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
	if eng.ProviderRequestID() != "" {
		t.Errorf("ProviderRequestID = %q, want empty (no handshake)", eng.ProviderRequestID())
	}
}

// Server-initiated close mid-stream should return cleanly (not an error)
// — we get a normal-closure WS frame, not a transport failure.
func TestDeepgramTTSWS_ServerNormalClose(t *testing.T) {
	wsURL, cleanup := mockDeepgramTTSServer(t, "", func(t *testing.T, ws *websocket.Conn) {
		// Read one Speak, send a frame, then close normally.
		_, _, _ = ws.ReadMessage()
		_ = ws.WriteMessage(websocket.BinaryMessage, []byte{0xaa, 0xbb})
		_ = ws.WriteMessage(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
	})
	defer cleanup()

	eng := newTestEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 4)
	textCh <- TextChunk{Text: "hi", IsSentenceEnd: false}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	streamErr := make(chan error, 1)
	var receivedAudio [][]byte
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for chunk := range audioCh {
			receivedAudio = append(receivedAudio, chunk.Data)
		}
	}()
	go func() { streamErr <- eng.Stream(ctx, textCh, audioCh) }()

	// Close the text channel so the sender can finish gracefully if the
	// server didn't initiate close fast enough — Stream returns when both
	// goroutines exit.
	close(textCh)

	if err := <-streamErr; err != nil {
		t.Fatalf("Stream after server normal close returned error: %v", err)
	}
	wg.Wait()

	if len(receivedAudio) == 0 {
		t.Error("expected at least one audio chunk before server closed")
	}
}

// Synthesize must error — the WS engine doesn't support the legacy path.
func TestDeepgramTTSWS_SynthesizeReturnsError(t *testing.T) {
	eng := newTestEngine("ws://nowhere")
	body, err := eng.Synthesize(context.Background(), "hello")
	if err == nil {
		t.Fatal("Synthesize unexpectedly returned nil error")
	}
	if body != nil {
		t.Error("Synthesize returned a non-nil body alongside the error")
	}
}

// URL builder: http(s) inputs should normalize to ws(s) and carry the
// expected query params for our preferred encoding.
func TestBuildDeepgramTTSURL(t *testing.T) {
	cases := []struct {
		in       string
		wantPref string
	}{
		{"https://api.deepgram.com", "wss://api.deepgram.com/v1/speak"},
		{"http://localhost:8000", "ws://localhost:8000/v1/speak"},
		{"wss://api.deepgram.com", "wss://api.deepgram.com/v1/speak"},
	}
	for _, c := range cases {
		got, err := buildDeepgramTTSURL(&DeepgramTTSConfig{URL: c.in, Model: "aura-asteria-en"})
		if err != nil {
			t.Fatalf("buildDeepgramTTSURL(%q): %v", c.in, err)
		}
		if !strings.HasPrefix(got, c.wantPref) {
			t.Errorf("URL = %q, want prefix %q", got, c.wantPref)
		}
		if !strings.Contains(got, "encoding=linear16") {
			t.Errorf("URL %q missing encoding=linear16", got)
		}
		if !strings.Contains(got, "sample_rate=24000") {
			t.Errorf("URL %q missing sample_rate=24000", got)
		}
		if !strings.Contains(got, "model=aura-asteria-en") {
			t.Errorf("URL %q missing model=aura-asteria-en", got)
		}
	}
}
