package engines

import (
	"context"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestBuildDeepgramURL(t *testing.T) {
	tests := []struct {
		name      string
		cfg       DeepgramSTTConfig
		wantErr   string
		wantHost  string
		wantQuery map[string]string // subset of expected query params
		noQuery   []string          // params that must be absent
	}{
		{
			name: "wss default shape",
			cfg: DeepgramSTTConfig{
				URL:           "wss://api.deepgram.com",
				Model:         "nova-3",
				EndpointingMs: 300,
			},
			wantHost: "api.deepgram.com",
			wantQuery: map[string]string{
				"model":           "nova-3",
				"encoding":        "mulaw",
				"sample_rate":     "8000",
				"channels":        "1",
				"punctuate":       "true",
				"interim_results": "true",
				"endpointing":     "300",
			},
			noQuery: []string{"language"},
		},
		{
			name: "https upgraded to wss",
			cfg: DeepgramSTTConfig{
				URL:           "https://api.deepgram.com",
				Model:         "nova-2",
				EndpointingMs: 500,
			},
			wantHost: "api.deepgram.com",
			wantQuery: map[string]string{
				"model":       "nova-2",
				"endpointing": "500",
			},
		},
		{
			name: "http upgraded to ws with language",
			cfg: DeepgramSTTConfig{
				URL:           "http://127.0.0.1:9999",
				Model:         "nova-2-phonecall",
				Language:      "en-US",
				EndpointingMs: 300,
			},
			wantHost: "127.0.0.1:9999",
			wantQuery: map[string]string{
				"model":    "nova-2-phonecall",
				"language": "en-US",
			},
		},
		{
			name:    "unsupported scheme",
			cfg:     DeepgramSTTConfig{URL: "ftp://api.deepgram.com", Model: "nova-3"},
			wantErr: "unsupported scheme",
		},
		{
			name:    "missing scheme",
			cfg:     DeepgramSTTConfig{URL: "api.deepgram.com", Model: "nova-3"},
			wantErr: "unsupported scheme",
		},
		{
			name:    "unparseable URL",
			cfg:     DeepgramSTTConfig{URL: "wss://bad host/", Model: "nova-3"},
			wantErr: "invalid character",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := buildDeepgramURL(&tc.cfg)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("buildDeepgramURL = %q, want error containing %q", got, tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("buildDeepgramURL: %v", err)
			}
			u, err := url.Parse(got)
			if err != nil {
				t.Fatalf("result unparseable: %v", err)
			}
			wantScheme := "wss"
			if strings.HasPrefix(tc.cfg.URL, "http://") || strings.HasPrefix(tc.cfg.URL, "ws://") {
				wantScheme = "ws"
			}
			if u.Scheme != wantScheme {
				t.Errorf("scheme = %q, want %q", u.Scheme, wantScheme)
			}
			if u.Host != tc.wantHost {
				t.Errorf("host = %q, want %q", u.Host, tc.wantHost)
			}
			if u.Path != "/v1/listen" {
				t.Errorf("path = %q, want /v1/listen", u.Path)
			}
			q := u.Query()
			for k, want := range tc.wantQuery {
				if got := q.Get(k); got != want {
					t.Errorf("query %s = %q, want %q", k, got, want)
				}
			}
			for _, k := range tc.noQuery {
				if _, ok := q[k]; ok {
					t.Errorf("query %s present (%q), want absent", k, q.Get(k))
				}
			}
		})
	}
}

func newTestDeepgramEngine(wsURL string, mutate func(*DeepgramSTTConfig)) *deepgramSTTEngine {
	cfg := DeepgramSTTConfig{URL: wsURL, APIKey: "dg-test-key"}
	eng := NewDeepgramSTT(cfg).(*deepgramSTTEngine)
	if mutate != nil {
		mutate(&eng.cfg)
	}
	return eng
}

// deepgramIgnoreAudio reads (and discards) inbound messages until the
// conn dies, keeping the server side alive for the whole test.
func deepgramIgnoreAudio(ws *websocket.Conn) {
	for {
		if _, _, err := ws.ReadMessage(); err != nil {
			return
		}
	}
}

func TestDeepgramSTT_HappyPath(t *testing.T) {
	wsURL, capture, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msgs := []string{
			`{"type":"Metadata","request_id":"abc"}`,
			`{"type":"SpeechStarted","timestamp":0.1}`,
			`{"type":"Results","channel":{"alternatives":[{"transcript":"hello","confidence":0.9}]},"is_final":false,"speech_final":false}`,
			`{"type":"Results","channel":{"alternatives":[{"transcript":"","confidence":0}]},"is_final":false}`,
			`{"type":"Results","channel":{"alternatives":[{"transcript":"hello there.","confidence":0.98}]},"is_final":true,"speech_final":true}`,
			`{"type":"UtteranceEnd","last_word_end":1.2}`,
		}
		for _, m := range msgs {
			if err := ws.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				return
			}
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, nil)
	eng.SetCallSID("CA-test-happy")
	events := make(chan STTEvent, 32)
	ctx := context.Background()
	if err := eng.Start(ctx, events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	expectSTTEvent(t, events, STTTranscriptDelta, "hello")
	expectSTTEvent(t, events, STTTranscriptFinal, "hello there.")

	// Handshake assertions: auth header + core listen query params.
	path, query, header := capture.get()
	if path != "/v1/listen" {
		t.Errorf("dial path = %q, want /v1/listen", path)
	}
	if got := header.Get("Authorization"); got != "Token dg-test-key" {
		t.Errorf("Authorization = %q, want %q", got, "Token dg-test-key")
	}
	for k, want := range map[string]string{
		"model":           "nova-3",
		"encoding":        "mulaw",
		"sample_rate":     "8000",
		"channels":        "1",
		"interim_results": "true",
		"endpointing":     "300",
	} {
		if got := query.Get(k); got != want {
			t.Errorf("dial query %s = %q, want %q", k, got, want)
		}
	}

	if err := eng.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	extra := drainUntilClosed(t, events, 2*time.Second)
	for _, ev := range extra {
		if ev.Type == STTError {
			t.Errorf("unexpected STTError during clean stop: %v", ev.Err)
		}
	}

	// Counters exercised by the envelope mix above.
	if got := eng.speechStartedRecv.Load(); got != 1 {
		t.Errorf("speechStartedRecv = %d, want 1", got)
	}
	if got := eng.utteranceEndRecv.Load(); got != 1 {
		t.Errorf("utteranceEndRecv = %d, want 1", got)
	}
	if got := eng.resultsSpeechFinal.Load(); got != 1 {
		t.Errorf("resultsSpeechFinal = %d, want 1", got)
	}
	if got := eng.finalsEmitted.Load(); got != 1 {
		t.Errorf("finalsEmitted = %d, want 1", got)
	}
}

// Multiple is_final segments inside one utterance merge into one final
// when speech_final eventually fires.
func TestDeepgramSTT_IsFinalSegmentsMerge(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msgs := []string{
			`{"type":"Results","channel":{"alternatives":[{"transcript":"six"}]},"is_final":false}`,
			`{"type":"Results","channel":{"alternatives":[{"transcript":"six seven eight"}]},"is_final":true,"speech_final":false}`,
			`{"type":"Results","channel":{"alternatives":[{"transcript":"four seven two"}]},"is_final":true,"speech_final":true}`,
		}
		for _, m := range msgs {
			if err := ws.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				return
			}
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, nil)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	expectSTTEvent(t, events, STTTranscriptDelta, "six")
	expectSTTEvent(t, events, STTTranscriptFinal, "six seven eight four seven two")
}

// When speech_final never fires, the debounce timer flushes accumulated
// is_final text after FlushTimeoutMs.
func TestDeepgramSTT_DebounceFlushWithoutSpeechFinal(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msg := `{"type":"Results","channel":{"alternatives":[{"transcript":"okay"}]},"is_final":true,"speech_final":false}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(msg)); err != nil {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 80
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	// Final must arrive via the debounce path, well after the 80 ms
	// window but within the receive timeout.
	expectSTTEvent(t, events, STTTranscriptFinal, "okay")
}

func TestDeepgramSTT_AudioPathAndCloseStream(t *testing.T) {
	type wsMsg struct {
		mt   int
		data []byte
	}
	received := make(chan wsMsg, 16)
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for {
			mt, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			received <- wsMsg{mt: mt, data: data}
		}
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, nil)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	frames := [][]byte{{0xaa, 0xbb, 0xcc}, {0x00, 0x11}, {0x7f}}
	for _, f := range frames {
		eng.PushAudio(f)
	}
	for i, want := range frames {
		select {
		case got := <-received:
			if got.mt != websocket.BinaryMessage {
				t.Fatalf("frame %d type = %d, want BinaryMessage", i, got.mt)
			}
			if string(got.data) != string(want) {
				t.Fatalf("frame %d = %v, want %v", i, got.data, want)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timeout waiting for audio frame %d at server", i)
		}
	}

	// Stop must send the CloseStream sentinel as a text message before
	// tearing down the conn.
	if err := eng.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	select {
	case got := <-received:
		if got.mt != websocket.TextMessage {
			t.Fatalf("post-audio message type = %d, want TextMessage (CloseStream)", got.mt)
		}
		if !strings.Contains(string(got.data), "CloseStream") {
			t.Fatalf("post-audio message = %q, want CloseStream sentinel", got.data)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timeout waiting for CloseStream sentinel")
	}

	drainUntilClosed(t, events, 2*time.Second)
	if got := eng.Usage().AudioSeconds; got != 6.0/8000 {
		t.Errorf("Usage().AudioSeconds = %v, want %v", got, 6.0/8000)
	}
}

func TestDeepgramSTT_StopSemantics(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, deepgramIgnoreAudio)
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, nil)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}

	if err := eng.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	extra := drainUntilClosed(t, events, 2*time.Second)
	for _, ev := range extra {
		if ev.Type == STTError {
			t.Errorf("unexpected STTError on clean Stop: %v", ev.Err)
		}
	}

	// Second Stop is a no-op.
	done := make(chan error, 1)
	go func() { done <- eng.Stop() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("second Stop: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("second Stop blocked")
	}

	// PushAudio after Stop must not block or panic.
	pushed := make(chan struct{})
	go func() {
		for i := 0; i < 300; i++ {
			eng.PushAudio(make([]byte, 160))
		}
		close(pushed)
	}()
	select {
	case <-pushed:
	case <-time.After(2 * time.Second):
		t.Fatalf("PushAudio after Stop blocked")
	}
}

func TestDeepgramSTT_ServerErrorEnvelope(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Error","description":"quota exceeded"}`))
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, nil)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	ev := recvSTTEvent(t, events, 2*time.Second)
	if ev.Type != STTError {
		t.Fatalf("event = %s, want Error", sttTypeName(ev.Type))
	}
	if ev.Err == nil || !strings.Contains(ev.Err.Error(), "quota exceeded") {
		t.Fatalf("error = %v, want it to contain 'quota exceeded'", ev.Err)
	}
	// The reader exits after an Error envelope; the events channel closes.
	drainUntilClosed(t, events, 2*time.Second)
}

func TestDeepgramSTT_ServerClosesUnexpectedly(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		// Emit one buffered is_final so the abnormal close also exercises
		// the last-resort flush, then die without a close handshake.
		_ = ws.WriteMessage(websocket.TextMessage,
			[]byte(`{"type":"Results","channel":{"alternatives":[{"transcript":"partial thou"}]},"is_final":true,"speech_final":false}`))
		_ = ws.Close()
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, nil)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	// Last-resort flush of the buffered is_final segment, then the error.
	expectSTTEvent(t, events, STTTranscriptFinal, "partial thou")

	sawError := false
	for _, ev := range drainUntilClosed(t, events, 2*time.Second) {
		if ev.Type == STTError {
			sawError = true
		}
	}
	if !sawError {
		t.Errorf("expected an STTError after unexpected server close")
	}
}

func TestDeepgramSTT_ContextCancellation(t *testing.T) {
	proceed := make(chan struct{})
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		<-proceed
		_ = ws.Close()
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, nil)
	events := make(chan STTEvent, 32)
	ctx, cancel := context.WithCancel(context.Background())
	if err := eng.Start(ctx, events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	cancel()
	close(proceed) // server tears down the conn after cancellation

	// Cancellation is graceful: channel closes without an STTError.
	for _, ev := range drainUntilClosed(t, events, 2*time.Second) {
		if ev.Type == STTError {
			t.Errorf("unexpected STTError after ctx cancel: %v", ev.Err)
		}
	}
}

func TestDeepgramSTT_DialFailure(t *testing.T) {
	srvURL, srvCleanup := newNonUpgradingServer()
	defer srvCleanup()

	eng := newTestDeepgramEngine(srvURL, nil)
	events := make(chan STTEvent, 8)
	err := eng.Start(context.Background(), events)
	if err == nil {
		eng.Stop()
		t.Fatalf("Start succeeded against 401 server, want error")
	}
	if !strings.Contains(err.Error(), "deepgram-stt") || !strings.Contains(err.Error(), "401") {
		t.Errorf("Start error = %v, want engine name and status 401", err)
	}
	// The engine must not have closed or written the events channel.
	select {
	case ev, ok := <-events:
		t.Fatalf("unexpected activity on events channel after dial failure: %v ok=%v", ev, ok)
	default:
	}
}

func TestDeepgramSTT_MissingAPIKey(t *testing.T) {
	eng := NewDeepgramSTT(DeepgramSTTConfig{URL: "ws://127.0.0.1:1"})
	err := eng.Start(context.Background(), make(chan STTEvent, 1))
	if err == nil || !strings.Contains(err.Error(), "API key") {
		t.Fatalf("Start without key = %v, want API key error", err)
	}
}

func TestDeepgramSTT_TranscribeUnsupported(t *testing.T) {
	eng := NewDeepgramSTT(DeepgramSTTConfig{APIKey: "k"})
	if _, err := eng.Transcribe(context.Background(), nil); err == nil {
		t.Fatalf("Transcribe should be unsupported for deepgram-stt")
	}
	if eng.Name() != "deepgram-stt" {
		t.Errorf("Name = %q, want deepgram-stt", eng.Name())
	}
}
