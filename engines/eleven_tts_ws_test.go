package engines

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// mockElevenTTSServer spins up an httptest WS server for the
// ElevenLabs stream-input contract. The handler runs in the upgraded
// connection.
func mockElevenTTSServer(t *testing.T, handler func(t *testing.T, ws *websocket.Conn)) (wsURL string, cleanup func()) {
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

func newTestElevenWSEngine(wsURL string) *elevenTTSWSEngine {
	return &elevenTTSWSEngine{
		cfg: ElevenTTSConfig{
			URL:    wsURL,
			APIKey: "test-key",
			Model:  "eleven_flash_v2_5",
			Voice:  "JBFqnCBsd6RMkjVDRZzb",
		},
	}
}

// b64 helper.
func b64(b []byte) string { return base64.StdEncoding.EncodeToString(b) }

// Happy path: caller pushes one TextChunk + closes textCh; server
// echoes a base64-encoded audio payload and a real isFinal AFTER the
// client's EOS; engine pushes decoded PCM to audioCh, returns nil.
func TestElevenTTSWS_HappyPath(t *testing.T) {
	expectedText := "Thank you for calling."
	pcm := []byte{0x10, 0x20, 0x30, 0x40}

	// Captures from the WS handler goroutine; protected by mu so the
	// test goroutine's reads after Stream() returns don't race the
	// handler goroutine's writes during its run.
	var (
		mu      sync.Mutex
		sawBOS  bool
		sawEOS  bool
		bosText string
	)

	wsURL, cleanup := mockElevenTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		// 1. Read BOS (text + voice_settings + flush=true)
		mt, data, err := ws.ReadMessage()
		if err != nil || mt != websocket.TextMessage {
			t.Errorf("BOS read: mt=%d err=%v", mt, err)
			return
		}
		var bos map[string]interface{}
		_ = json.Unmarshal(data, &bos)
		if _, ok := bos["voice_settings"].(map[string]interface{}); !ok {
			t.Errorf("BOS missing voice_settings, got %v", bos)
		}
		text, _ := bos["text"].(string)
		mu.Lock()
		bosText = text
		sawBOS = true
		mu.Unlock()

		// Reply with one audio chunk + a per-flush isFinal=true (which
		// the engine MUST ignore because EOS hasn't been sent yet).
		_ = ws.WriteJSON(map[string]interface{}{
			"audio":   b64(pcm),
			"isFinal": true,
		})

		// 2. Read EOS ({"text": ""})
		_, data, err = ws.ReadMessage()
		if err != nil {
			return
		}
		var eos map[string]interface{}
		_ = json.Unmarshal(data, &eos)
		if eos["text"] != "" {
			t.Errorf("expected EOS with empty text, got %v", eos)
		}
		mu.Lock()
		sawEOS = true
		mu.Unlock()

		// Send a real session-final isFinal — engine should now exit.
		_ = ws.WriteJSON(map[string]interface{}{"isFinal": true})
	})
	defer cleanup()

	eng := newTestElevenWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 4)
	textCh <- TextChunk{Text: expectedText, IsSentenceEnd: true}
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
	mu.Lock()
	gotSawBOS, gotSawEOS, gotBosText := sawBOS, sawEOS, bosText
	mu.Unlock()
	if !gotSawBOS {
		t.Error("server did not see BOS")
	}
	if gotBosText != expectedText {
		t.Errorf("BOS text = %q, want %q", gotBosText, expectedText)
	}
	if !gotSawEOS {
		t.Error("server did not see EOS")
	}
	if len(got) != 1 {
		t.Fatalf("got %d audio chunks, want 1", len(got))
	}
	if string(got[0]) != string(pcm) {
		t.Errorf("decoded audio = %v, want %v", got[0], pcm)
	}
	if eng.Usage().CharsIn != len(expectedText) {
		t.Errorf("Usage.CharsIn = %d, want %d", eng.Usage().CharsIn, len(expectedText))
	}
	// ElevenLabs WS doesn't expose a published correlation handle —
	// ProviderRequestID stays empty by design.
	if eng.ProviderRequestID() != "" {
		t.Errorf("ProviderRequestID = %q, want empty (eleven has no published handle)", eng.ProviderRequestID())
	}
}

// Pre-EOS isFinal is per-flush bookkeeping the engine must IGNORE.
// If we honoured it as session-end, audio sent after the first flush
// would be lost. This test sends three flushes' worth of audio with
// isFinal interleaved, then closes the input.
func TestElevenTTSWS_PreEOSIsFinalIgnored(t *testing.T) {
	pcms := [][]byte{{1}, {2}, {3}}

	wsURL, cleanup := mockElevenTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		// BOS
		_, _, _ = ws.ReadMessage()
		// Real ElevenLabs sends audio and per-flush isFinal in
		// SEPARATE messages — three audio frames then one bare
		// isFinal=true ack. Modelling them in one message conflated
		// the two and missed the actual protocol shape.
		for _, pcm := range pcms {
			_ = ws.WriteJSON(map[string]interface{}{"audio": b64(pcm)})
		}
		_ = ws.WriteJSON(map[string]interface{}{"isFinal": true})
		// Now read EOS
		_, _, _ = ws.ReadMessage()
		// Real session-end
		_ = ws.WriteJSON(map[string]interface{}{"isFinal": true})
	})
	defer cleanup()

	eng := newTestElevenWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 8)
	textCh <- TextChunk{Text: "hi"}
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

	if len(got) != len(pcms) {
		t.Fatalf("got %d chunks, want %d (pre-EOS isFinal must NOT abort the receiver)", len(got), len(pcms))
	}
	for i, p := range pcms {
		if string(got[i]) != string(p) {
			t.Errorf("chunk %d = %v, want %v", i, got[i], p)
		}
	}
}

// Cancellation: ctx-cancel mid-stream returns nil, server sees EOS,
// audioCh is closed by the engine.
func TestElevenTTSWS_Cancellation(t *testing.T) {
	gotEOS := make(chan struct{}, 1)
	wsURL, cleanup := mockElevenTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env map[string]interface{}
			if json.Unmarshal(data, &env) == nil {
				if t, ok := env["text"].(string); ok && t == "" && env["voice_settings"] == nil {
					select {
					case gotEOS <- struct{}{}:
					default:
					}
					_ = ws.WriteJSON(map[string]interface{}{"isFinal": true})
					return
				}
			}
		}
	})
	defer cleanup()

	eng := newTestElevenWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 4)
	textCh <- TextChunk{Text: "hello"}

	ctx, cancel := context.WithCancel(context.Background())
	streamErr := make(chan error, 1)
	go func() { streamErr <- eng.Stream(ctx, textCh, audioCh) }()

	time.Sleep(100 * time.Millisecond)
	cancel()

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

	select {
	case <-gotEOS:
	case <-time.After(500 * time.Millisecond):
		t.Error("server did not receive EOS after cancel")
	}
}

// Empty input: closing textCh with no chunks must not dial WS.
func TestElevenTTSWS_EmptyInputNoConnect(t *testing.T) {
	var dialed bool
	wsURL, cleanup := mockElevenTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		dialed = true
	})
	defer cleanup()

	eng := newTestElevenWSEngine(wsURL)
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

// API error in server payload should be logged but not crash; engine
// continues until isFinal-after-EOS or transport close.
func TestElevenTTSWS_APIErrorIsLoggedNotFatal(t *testing.T) {
	wsURL, cleanup := mockElevenTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		_, _, _ = ws.ReadMessage() // BOS
		_ = ws.WriteJSON(map[string]interface{}{
			"error":   "voice_not_found",
			"message": "voice id JBFqn... is not valid",
		})
		_, _, _ = ws.ReadMessage() // EOS
		_ = ws.WriteJSON(map[string]interface{}{"isFinal": true})
	})
	defer cleanup()

	eng := newTestElevenWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 4)
	textCh <- TextChunk{Text: "hi"}
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
}

func TestElevenTTSWS_SynthesizeReturnsError(t *testing.T) {
	eng := newTestElevenWSEngine("ws://nowhere")
	body, err := eng.Synthesize(context.Background(), "hello")
	if err == nil {
		t.Fatal("Synthesize unexpectedly returned nil error")
	}
	if body != nil {
		t.Error("Synthesize returned a non-nil body alongside the error")
	}
}

func TestBuildElevenTTSURL(t *testing.T) {
	cases := []struct {
		in       string
		wantPref string
	}{
		{"https://api.elevenlabs.io", "wss://api.elevenlabs.io/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb/stream-input"},
		{"http://localhost:9000", "ws://localhost:9000/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb/stream-input"},
		{"wss://api.elevenlabs.io", "wss://api.elevenlabs.io/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb/stream-input"},
	}
	for _, c := range cases {
		got, err := buildElevenTTSURL(&ElevenTTSConfig{
			URL:   c.in,
			Model: "eleven_flash_v2_5",
			Voice: "JBFqnCBsd6RMkjVDRZzb",
		})
		if err != nil {
			t.Fatalf("buildElevenTTSURL(%q): %v", c.in, err)
		}
		if !strings.HasPrefix(got, c.wantPref) {
			t.Errorf("URL = %q, want prefix %q", got, c.wantPref)
		}
		if !strings.Contains(got, "model_id=eleven_flash_v2_5") {
			t.Errorf("URL %q missing model_id=eleven_flash_v2_5", got)
		}
		if !strings.Contains(got, "output_format=pcm_24000") {
			t.Errorf("URL %q missing output_format=pcm_24000", got)
		}
	}
}
