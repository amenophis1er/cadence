package engines

import (
	"context"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestBuildCartesiaSTTURL(t *testing.T) {
	tests := []struct {
		name      string
		cfg       CartesiaSTTConfig
		wantErr   string
		wantHost  string
		wantQuery map[string]string
		noQuery   []string
	}{
		{
			name: "wss default shape",
			cfg: CartesiaSTTConfig{
				URL:                    "wss://api.cartesia.ai",
				Model:                  "ink-whisper",
				MaxSilenceDurationSecs: 1.0,
			},
			wantHost: "api.cartesia.ai",
			wantQuery: map[string]string{
				"model":                     "ink-whisper",
				"encoding":                  "pcm_mulaw",
				"sample_rate":               "8000",
				"cartesia_version":          cartesiaAPIVersion,
				"max_silence_duration_secs": "1",
			},
			noQuery: []string{"min_volume", "language"},
		},
		{
			name: "https upgraded to wss with vad and language",
			cfg: CartesiaSTTConfig{
				URL:                    "https://api.cartesia.ai",
				Model:                  "ink-whisper",
				Language:               "fr",
				MinVolume:              0.15,
				MaxSilenceDurationSecs: 0.4,
			},
			wantHost: "api.cartesia.ai",
			wantQuery: map[string]string{
				"min_volume":                "0.15",
				"language":                  "fr",
				"max_silence_duration_secs": "0.4",
			},
		},
		{
			name: "http upgraded to ws",
			cfg: CartesiaSTTConfig{
				URL:                    "http://127.0.0.1:8080",
				Model:                  "ink-whisper",
				MaxSilenceDurationSecs: 1.0,
			},
			wantHost:  "127.0.0.1:8080",
			wantQuery: map[string]string{"model": "ink-whisper"},
		},
		{
			name:    "unsupported scheme",
			cfg:     CartesiaSTTConfig{URL: "ftp://api.cartesia.ai", Model: "ink-whisper"},
			wantErr: "unsupported scheme",
		},
		{
			name:    "unparseable URL",
			cfg:     CartesiaSTTConfig{URL: "wss://bad host/", Model: "ink-whisper"},
			wantErr: "invalid character",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := buildCartesiaSTTURL(&tc.cfg)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("buildCartesiaSTTURL = %q, want error containing %q", got, tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("buildCartesiaSTTURL: %v", err)
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
			if u.Path != "/stt/websocket" {
				t.Errorf("path = %q, want /stt/websocket", u.Path)
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

func newTestCartesiaEngine(wsURL string) *cartesiaSTTEngine {
	return NewCartesiaSTT(CartesiaSTTConfig{URL: wsURL, APIKey: "ct-test-key"}).(*cartesiaSTTEngine)
}

// cartesiaIgnoreAudio drains inbound messages until the conn dies.
func cartesiaIgnoreAudio(ws *websocket.Conn) {
	for {
		if _, _, err := ws.ReadMessage(); err != nil {
			return
		}
	}
}

func TestCartesiaSTT_HappyPath(t *testing.T) {
	wsURL, capture, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msgs := []string{
			`{"type":"flush_done"}`,
			`{"type":"transcript","is_final":false,"text":"good ","duration":0.4}`,
			`{"type":"transcript","is_final":false,"text":"  "}`,
			`{"type":"transcript","is_final":true,"text":"good morning","duration":1.1}`,
			`{"type":"done"}`,
		}
		for _, m := range msgs {
			if err := ws.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				return
			}
		}
		cartesiaIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestCartesiaEngine(wsURL)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	expectSTTEvent(t, events, STTTranscriptDelta, "good")
	// is_final commits: SpeechStopped first so the mediator's gate
	// re-arms, then the final text.
	expectSTTEvent(t, events, STTSpeechStopped, "")
	expectSTTEvent(t, events, STTTranscriptFinal, "good morning")

	path, query, header := capture.get()
	if path != "/stt/websocket" {
		t.Errorf("dial path = %q, want /stt/websocket", path)
	}
	if got := header.Get("X-API-Key"); got != "ct-test-key" {
		t.Errorf("X-API-Key = %q, want %q", got, "ct-test-key")
	}
	if got := header.Get("Cartesia-Version"); got != cartesiaAPIVersion {
		t.Errorf("Cartesia-Version = %q, want %q", got, cartesiaAPIVersion)
	}
	for k, want := range map[string]string{
		"model":                     "ink-whisper",
		"encoding":                  "pcm_mulaw",
		"sample_rate":               "8000",
		"cartesia_version":          cartesiaAPIVersion,
		"max_silence_duration_secs": "1",
	} {
		if got := query.Get(k); got != want {
			t.Errorf("dial query %s = %q, want %q", k, got, want)
		}
	}

	if err := eng.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	for _, ev := range drainUntilClosed(t, events, 2*time.Second) {
		if ev.Type == STTError {
			t.Errorf("unexpected STTError on clean stop: %v", ev.Err)
		}
	}
}

// A second utterance after a final re-emits SpeechStarted.
func TestCartesiaSTT_SpeechStartedPerUtterance(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msgs := []string{
			`{"type":"transcript","is_final":true,"text":"yes"}`,
			`{"type":"transcript","is_final":false,"text":"and also"}`,
		}
		for _, m := range msgs {
			if err := ws.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				return
			}
		}
		cartesiaIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestCartesiaEngine(wsURL)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	// First utterance arrives already-final: still synthesises the
	// started/stopped bracket around it.
	expectSTTEvent(t, events, STTSpeechStarted, "")
	expectSTTEvent(t, events, STTSpeechStopped, "")
	expectSTTEvent(t, events, STTTranscriptFinal, "yes")
	// Next partial opens a fresh utterance.
	expectSTTEvent(t, events, STTSpeechStarted, "")
	expectSTTEvent(t, events, STTTranscriptDelta, "and also")
}

func TestCartesiaSTT_AudioPath(t *testing.T) {
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

	eng := newTestCartesiaEngine(wsURL)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	frames := [][]byte{{0x01}, {0x02, 0x03}, {0x04, 0x05, 0x06}}
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
	if got := eng.Usage().AudioSeconds; got != 6.0/8000 {
		t.Errorf("Usage().AudioSeconds = %v, want %v", got, 6.0/8000)
	}
}

func TestCartesiaSTT_StopSemantics(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, cartesiaIgnoreAudio)
	defer cleanup()

	eng := newTestCartesiaEngine(wsURL)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}

	if err := eng.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	for _, ev := range drainUntilClosed(t, events, 2*time.Second) {
		if ev.Type == STTError {
			t.Errorf("unexpected STTError on clean Stop: %v", ev.Err)
		}
	}

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

func TestCartesiaSTT_ErrorEnvelope(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"error","title":"Bad Request","message":"unsupported encoding"}`))
		cartesiaIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestCartesiaEngine(wsURL)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	ev := recvSTTEvent(t, events, 2*time.Second)
	if ev.Type != STTError {
		t.Fatalf("event = %s, want Error", sttTypeName(ev.Type))
	}
	if ev.Err == nil || !strings.Contains(ev.Err.Error(), "unsupported encoding") {
		t.Fatalf("error = %v, want it to contain the vendor message", ev.Err)
	}
	drainUntilClosed(t, events, 2*time.Second)
}

func TestCartesiaSTT_ServerClosesUnexpectedly(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		_ = ws.Close()
	})
	defer cleanup()

	eng := newTestCartesiaEngine(wsURL)
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

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

func TestCartesiaSTT_ContextCancellation(t *testing.T) {
	proceed := make(chan struct{})
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		<-proceed
		_ = ws.Close()
	})
	defer cleanup()

	eng := newTestCartesiaEngine(wsURL)
	events := make(chan STTEvent, 32)
	ctx, cancel := context.WithCancel(context.Background())
	if err := eng.Start(ctx, events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	cancel()
	close(proceed)

	for _, ev := range drainUntilClosed(t, events, 2*time.Second) {
		if ev.Type == STTError {
			t.Errorf("unexpected STTError after ctx cancel: %v", ev.Err)
		}
	}
}

func TestCartesiaSTT_DialFailure(t *testing.T) {
	srvURL, srvCleanup := newNonUpgradingServer()
	defer srvCleanup()

	eng := newTestCartesiaEngine(srvURL)
	err := eng.Start(context.Background(), make(chan STTEvent, 8))
	if err == nil {
		eng.Stop()
		t.Fatalf("Start succeeded against 401 server, want error")
	}
	if !strings.Contains(err.Error(), "cartesia-stt") || !strings.Contains(err.Error(), "401") {
		t.Errorf("Start error = %v, want engine name and status 401", err)
	}
}

func TestCartesiaSTT_MissingAPIKey(t *testing.T) {
	eng := NewCartesiaSTT(CartesiaSTTConfig{URL: "ws://127.0.0.1:1"})
	err := eng.Start(context.Background(), make(chan STTEvent, 1))
	if err == nil || !strings.Contains(err.Error(), "API key") {
		t.Fatalf("Start without key = %v, want API key error", err)
	}
}

func TestCartesiaSTT_TranscribeUnsupported(t *testing.T) {
	eng := NewCartesiaSTT(CartesiaSTTConfig{APIKey: "k"})
	if _, err := eng.Transcribe(context.Background(), nil); err == nil {
		t.Fatalf("Transcribe should be unsupported for cartesia-stt")
	}
	if eng.Name() != "cartesia-stt" {
		t.Errorf("Name = %q, want cartesia-stt", eng.Name())
	}
}
