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

// mockInworldTTSServer spins up an httptest WS server for the
// /tts/v1/voice:streamBidirectional contract.
func mockInworldTTSServer(t *testing.T, handler func(t *testing.T, ws *websocket.Conn, authQueryValue string)) (wsURL string, cleanup func()) {
	t.Helper()
	upgrader := websocket.Upgrader{
		ReadBufferSize:  4096,
		WriteBufferSize: 4096,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.URL.Query().Get("authorization")
		ws, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer ws.Close()
		handler(t, ws, auth)
	}))
	wsURL = "ws" + strings.TrimPrefix(srv.URL, "http")
	return wsURL, srv.Close
}

func newTestInworldWSEngine(wsURL string) *inworldTTSWSEngine {
	return &inworldTTSWSEngine{
		cfg: InworldTTSConfig{
			URL:    wsURL,
			APIKey: "test-key",
			Model:  "inworld-tts-1.5-max",
			Voice:  "Dennis",
		},
	}
}

// Happy path: caller pushes one chunk + closes textCh; server emits
// audioChunk + contextClosed; engine pushes decoded PCM, returns nil.
// Also verifies auth lands as a query param ("Basic test-key").
func TestInworldTTSWS_HappyPath(t *testing.T) {
	expected := "Thank you for calling."
	pcm := []byte{0x10, 0x20, 0x30, 0x40}

	var (
		mu              sync.Mutex
		gotAuth         string
		createMsg       map[string]interface{}
		sendTextMsg     map[string]interface{}
		closeContextMsg map[string]interface{}
		ctxIDOnCreate   string
	)

	wsURL, cleanup := mockInworldTTSServer(t, func(t *testing.T, ws *websocket.Conn, auth string) {
		mu.Lock()
		gotAuth = auth
		mu.Unlock()

		// 1. Read create
		_, data, err := ws.ReadMessage()
		if err != nil {
			return
		}
		var m map[string]interface{}
		_ = json.Unmarshal(data, &m)
		mu.Lock()
		createMsg = m
		if id, ok := m["contextId"].(string); ok {
			ctxIDOnCreate = id
		}
		mu.Unlock()

		// 2. Read send_text
		_, data, err = ws.ReadMessage()
		if err != nil {
			return
		}
		_ = json.Unmarshal(data, &m)
		mu.Lock()
		sendTextMsg = m
		mu.Unlock()

		// Emit one audioChunk
		_ = ws.WriteJSON(map[string]interface{}{
			"result": map[string]interface{}{
				"contextId": ctxIDOnCreate,
				"audioChunk": map[string]interface{}{
					"audioContent": base64.StdEncoding.EncodeToString(pcm),
				},
				"status": map[string]interface{}{"code": 0},
			},
		})

		// 3. Read close_context
		_, data, err = ws.ReadMessage()
		if err != nil {
			return
		}
		_ = json.Unmarshal(data, &m)
		mu.Lock()
		closeContextMsg = m
		mu.Unlock()

		// Reply with contextClosed for the right contextId
		_ = ws.WriteJSON(map[string]interface{}{
			"result": map[string]interface{}{
				"contextId":     ctxIDOnCreate,
				"contextClosed": map[string]interface{}{},
				"status":        map[string]interface{}{"code": 0},
			},
		})
	})
	defer cleanup()

	eng := newTestInworldWSEngine(wsURL)
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

	mu.Lock()
	defer mu.Unlock()

	if gotAuth != "Basic test-key" {
		t.Errorf("auth query param = %q, want %q", gotAuth, "Basic test-key")
	}

	createBlock, ok := createMsg["create"].(map[string]interface{})
	if !ok {
		t.Fatalf("create message did not have create block: %v", createMsg)
	}
	if createBlock["voiceId"] != "Dennis" {
		t.Errorf("voiceId = %v, want Dennis", createBlock["voiceId"])
	}
	if createBlock["modelId"] != "inworld-tts-1.5-max" {
		t.Errorf("modelId = %v, want inworld-tts-1.5-max", createBlock["modelId"])
	}
	ac, _ := createBlock["audioConfig"].(map[string]interface{})
	if ac["audioEncoding"] != "PCM" {
		t.Errorf("audioEncoding = %v, want PCM", ac["audioEncoding"])
	}
	if rate, _ := ac["sampleRateHertz"].(float64); int(rate) != 24000 {
		t.Errorf("sampleRateHertz = %v, want 24000", ac["sampleRateHertz"])
	}

	stBlock, ok := sendTextMsg["send_text"].(map[string]interface{})
	if !ok {
		t.Fatalf("send_text message missing send_text block: %v", sendTextMsg)
	}
	if stBlock["text"] != expected {
		t.Errorf("send_text.text = %q, want %q", stBlock["text"], expected)
	}

	if _, ok := closeContextMsg["close_context"]; !ok {
		t.Errorf("close_context message did not have close_context block: %v", closeContextMsg)
	}

	if len(got) != 1 || string(got[0]) != string(pcm) {
		t.Fatalf("got %d audio chunks (want 1, %v)", len(got), got)
	}
	if eng.Usage().CharsIn != len(expected) {
		t.Errorf("Usage.CharsIn = %d, want %d", eng.Usage().CharsIn, len(expected))
	}
	if eng.ProviderRequestID() == "" {
		t.Error("ProviderRequestID empty (should be the client-generated contextId)")
	}
}

// Cancellation: ctx-cancel triggers close_context and a clean return.
func TestInworldTTSWS_CancellationSendsCloseContext(t *testing.T) {
	gotClose := make(chan struct{}, 1)
	wsURL, cleanup := mockInworldTTSServer(t, func(t *testing.T, ws *websocket.Conn, _ string) {
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var m map[string]interface{}
			if json.Unmarshal(data, &m) == nil {
				if _, ok := m["close_context"]; ok {
					select {
					case gotClose <- struct{}{}:
					default:
					}
					// Reply with contextClosed so engine returns
					if id, _ := m["contextId"].(string); id != "" {
						_ = ws.WriteJSON(map[string]interface{}{
							"result": map[string]interface{}{
								"contextId":     id,
								"contextClosed": map[string]interface{}{},
								"status":        map[string]interface{}{"code": 0},
							},
						})
					}
					return
				}
			}
		}
	})
	defer cleanup()

	eng := newTestInworldWSEngine(wsURL)
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
	if err := <-streamErr; err != nil {
		t.Fatalf("Stream after cancel returned error: %v", err)
	}

	select {
	case <-gotClose:
	case <-time.After(500 * time.Millisecond):
		t.Error("server did not receive close_context after ctx cancel")
	}
}

// Non-zero status code in a server result envelope must surface as a
// Stream error. status.code=0 (the OK case in the happy-path test) is
// the no-op; non-zero is a vendor error.
func TestInworldTTSWS_NonZeroStatusPropagates(t *testing.T) {
	wsURL, cleanup := mockInworldTTSServer(t, func(t *testing.T, ws *websocket.Conn, _ string) {
		_, _, _ = ws.ReadMessage()     // create
		_, data, _ := ws.ReadMessage() // send_text
		var m map[string]interface{}
		_ = json.Unmarshal(data, &m)
		ctxID, _ := m["contextId"].(string)
		_ = ws.WriteJSON(map[string]interface{}{
			"result": map[string]interface{}{
				"contextId": ctxID,
				"status":    map[string]interface{}{"code": 7, "message": "voice not found"},
			},
		})
	})
	defer cleanup()

	eng := newTestInworldWSEngine(wsURL)
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
	err := <-streamErr
	if err == nil {
		t.Fatal("expected error for non-zero status, got nil")
	}
	if !strings.Contains(err.Error(), "voice not found") {
		t.Errorf("error did not contain vendor message: %v", err)
	}
}

// Whitespace-only chunks must be skipped — Inworld would reject empty
// send_text payloads.
func TestInworldTTSWS_BlankChunksSkipped(t *testing.T) {
	receivedTexts := []string{}
	wsURL, cleanup := mockInworldTTSServer(t, func(t *testing.T, ws *websocket.Conn, _ string) {
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var m map[string]interface{}
			if json.Unmarshal(data, &m) != nil {
				continue
			}
			if st, ok := m["send_text"].(map[string]interface{}); ok {
				if t, ok := st["text"].(string); ok {
					receivedTexts = append(receivedTexts, t)
				}
			}
			if _, ok := m["close_context"]; ok {
				if id, _ := m["contextId"].(string); id != "" {
					_ = ws.WriteJSON(map[string]interface{}{
						"result": map[string]interface{}{
							"contextId":     id,
							"contextClosed": map[string]interface{}{},
							"status":        map[string]interface{}{"code": 0},
						},
					})
				}
				return
			}
		}
	})
	defer cleanup()

	eng := newTestInworldWSEngine(wsURL)
	textCh := make(chan TextChunk, 4)
	audioCh := make(chan AudioChunk, 4)
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
		t.Fatalf("got %d non-blank texts, want 2: %v", len(receivedTexts), receivedTexts)
	}
	if receivedTexts[0] != "Real text." || receivedTexts[1] != "More real text." {
		t.Errorf("delivered texts mismatch: %v", receivedTexts)
	}
}

// Long text gets truncated to 1000 chars (Inworld's per-message cap).
func TestInworldTTSWS_TextLengthClamp(t *testing.T) {
	gotLen := 0
	wsURL, cleanup := mockInworldTTSServer(t, func(t *testing.T, ws *websocket.Conn, _ string) {
		_, _, _ = ws.ReadMessage()     // create
		_, data, _ := ws.ReadMessage() // send_text
		var m map[string]interface{}
		_ = json.Unmarshal(data, &m)
		if st, ok := m["send_text"].(map[string]interface{}); ok {
			if txt, ok := st["text"].(string); ok {
				gotLen = len(txt)
			}
		}
		// Close cleanly so the engine doesn't get stuck.
		_, _, _ = ws.ReadMessage() // close_context
		ctxID, _ := m["contextId"].(string)
		_ = ws.WriteJSON(map[string]interface{}{
			"result": map[string]interface{}{
				"contextId":     ctxID,
				"contextClosed": map[string]interface{}{},
				"status":        map[string]interface{}{"code": 0},
			},
		})
	})
	defer cleanup()

	eng := newTestInworldWSEngine(wsURL)
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 4)
	textCh <- TextChunk{Text: strings.Repeat("a", 1500)} // 1500 chars
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

	if gotLen != 1000 {
		t.Errorf("server received %d chars, want 1000 (clamped)", gotLen)
	}
}

// Empty input: closing textCh before any chunk must not dial WS.
func TestInworldTTSWS_EmptyInputNoConnect(t *testing.T) {
	var dialed bool
	wsURL, cleanup := mockInworldTTSServer(t, func(t *testing.T, ws *websocket.Conn, _ string) {
		dialed = true
	})
	defer cleanup()

	eng := newTestInworldWSEngine(wsURL)
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

func TestInworldTTSWS_SynthesizeReturnsError(t *testing.T) {
	eng := newTestInworldWSEngine("ws://nowhere")
	body, err := eng.Synthesize(context.Background(), "hello")
	if err == nil {
		t.Fatal("Synthesize unexpectedly returned nil error")
	}
	if body != nil {
		t.Error("Synthesize returned a non-nil body alongside the error")
	}
}

func TestBuildInworldTTSURL(t *testing.T) {
	cases := []struct {
		in       string
		wantPref string
	}{
		{"https://api.inworld.ai", "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"},
		{"http://localhost:8000", "ws://localhost:8000/tts/v1/voice:streamBidirectional"},
		{"wss://api.inworld.ai", "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"},
	}
	for _, c := range cases {
		got, err := buildInworldTTSURL(&InworldTTSConfig{URL: c.in, APIKey: "abc"})
		if err != nil {
			t.Fatalf("buildInworldTTSURL(%q): %v", c.in, err)
		}
		if !strings.HasPrefix(got, c.wantPref) {
			t.Errorf("URL = %q, want prefix %q", got, c.wantPref)
		}
		if !strings.Contains(got, "authorization=Basic+abc") && !strings.Contains(got, "authorization=Basic%20abc") {
			t.Errorf("URL %q missing authorization=Basic abc", got)
		}
	}
}
