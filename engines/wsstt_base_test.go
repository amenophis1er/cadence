package engines

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// mockSTTWSServer spins up an httptest WS server. The handler runs on
// the upgraded connection; capture records the handshake request's
// query params and headers so tests can assert auth/config without
// touching *testing.T from the handler goroutine.
type sttHandshakeCapture struct {
	mu     sync.Mutex
	path   string
	query  url.Values
	header http.Header
}

func (c *sttHandshakeCapture) get() (string, url.Values, http.Header) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.path, c.query, c.header
}

func mockSTTWSServer(t *testing.T, handler func(ws *websocket.Conn)) (wsURL string, capture *sttHandshakeCapture, cleanup func()) {
	t.Helper()
	capture = &sttHandshakeCapture{}
	upgrader := websocket.Upgrader{
		ReadBufferSize:  4096,
		WriteBufferSize: 4096,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capture.mu.Lock()
		capture.path = r.URL.Path
		capture.query = r.URL.Query()
		capture.header = r.Header.Clone()
		capture.mu.Unlock()
		ws, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer ws.Close()
		handler(ws)
	}))
	wsURL = "ws" + strings.TrimPrefix(srv.URL, "http")
	return wsURL, capture, srv.Close
}

func sttTypeName(tp STTEventType) string {
	switch tp {
	case STTSpeechStarted:
		return "SpeechStarted"
	case STTSpeechStopped:
		return "SpeechStopped"
	case STTTranscriptDelta:
		return "TranscriptDelta"
	case STTTranscriptFinal:
		return "TranscriptFinal"
	case STTError:
		return "Error"
	default:
		return "unknown"
	}
}

// recvSTTEvent receives one event or fails on close/timeout.
func recvSTTEvent(t *testing.T, ch <-chan STTEvent, timeout time.Duration) STTEvent {
	t.Helper()
	select {
	case ev, ok := <-ch:
		if !ok {
			t.Fatalf("events channel closed while waiting for an event")
		}
		return ev
	case <-time.After(timeout):
		t.Fatalf("timeout waiting for STT event")
	}
	panic("unreachable")
}

// drainUntilClosed drains remaining events until the channel closes,
// failing the test if it doesn't close within the timeout. Returns the
// drained events.
func drainUntilClosed(t *testing.T, ch <-chan STTEvent, timeout time.Duration) []STTEvent {
	t.Helper()
	deadline := time.After(timeout)
	var drained []STTEvent
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				return drained
			}
			drained = append(drained, ev)
		case <-deadline:
			t.Fatalf("timeout waiting for events channel to close (drained %d events)", len(drained))
		}
	}
}

func expectSTTEvent(t *testing.T, ch <-chan STTEvent, wantType STTEventType, wantText string) {
	t.Helper()
	ev := recvSTTEvent(t, ch, 2*time.Second)
	if ev.Type != wantType {
		t.Fatalf("event type = %s, want %s (text=%q err=%v)", sttTypeName(ev.Type), sttTypeName(wantType), ev.Text, ev.Err)
	}
	if ev.Text != wantText {
		t.Fatalf("%s text = %q, want %q", sttTypeName(wantType), ev.Text, wantText)
	}
}

// blockingReader is a trivial sttReaderFn that parks until stop.
func blockingReader(ctx context.Context, conn *websocket.Conn, events chan<- STTEvent, stopCh <-chan struct{}) {
	select {
	case <-ctx.Done():
	case <-stopCh:
	}
}

func TestWSSTTSession_PushAudioDropOnOverrun(t *testing.T) {
	// No reader/writer running: audioIn is never drained, so every push
	// past the buffer capacity must drop instead of blocking.
	s := &wsSTTSession{name: "test-stt"}
	s.audioIn = make(chan []byte, 4)
	s.stopCh = make(chan struct{})

	const total = 500
	done := make(chan struct{})
	go func() {
		defer close(done)
		frame := make([]byte, 160)
		for i := 0; i < total; i++ {
			s.pushAudio(frame)
		}
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatalf("pushAudio blocked on overrun; expected non-blocking drop")
	}

	if got := s.framesReceived.Load(); got != total {
		t.Errorf("framesReceived = %d, want %d", got, total)
	}
	if got := s.framesDropped.Load(); got != total-4 {
		t.Errorf("framesDropped = %d, want %d", got, total-4)
	}
}

func TestWSSTTSession_PushAudioOnZeroValueSession(t *testing.T) {
	// Never started: audioIn is nil. pushAudio must not block or panic
	// (send on a nil channel is never ready, so the default arm drops).
	var s wsSTTSession
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.pushAudio(make([]byte, 160))
		s.pushAudio(make([]byte, 160))
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatalf("pushAudio blocked on a never-started session")
	}
	if got := s.framesDropped.Load(); got != 2 {
		t.Errorf("framesDropped = %d, want 2", got)
	}
}

// waitForwarded polls until the writer has forwarded want frames.
func waitForwarded(t *testing.T, s *wsSTTSession, want int64) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if s.framesForwarded.Load() >= want {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("framesForwarded = %d, want >= %d", s.framesForwarded.Load(), want)
}

func TestWSSTTSession_UsageCountsOnlyForwardedAudio(t *testing.T) {
	// usage tracks audio actually delivered to the vendor — what the
	// vendor could bill — not every frame offered to pushAudio.
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				return
			}
		}
	})
	defer cleanup()

	s := &wsSTTSession{name: "test-stt"}
	conn, err := dialSTT(context.Background(), wsURL, nil, 2*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	events := make(chan STTEvent, 8)
	if err := s.start(context.Background(), conn, events, blockingReader); err != nil {
		t.Fatalf("start: %v", err)
	}

	// mu-law 8 kHz: 8000 bytes == 1 second.
	s.pushAudio(make([]byte, 8000))
	s.pushAudio(make([]byte, 4000))
	waitForwarded(t, s, 2)
	if got := s.usage().AudioSeconds; got != 1.5 {
		t.Errorf("AudioSeconds = %v, want 1.5", got)
	}

	// Post-stop pushes are never forwarded and must not accrue.
	if err := s.stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}
	s.pushAudio(make([]byte, 8000))
	if got := s.usage().AudioSeconds; got != 1.5 {
		t.Errorf("AudioSeconds after post-stop push = %v, want 1.5", got)
	}
	drainUntilClosed(t, events, 2*time.Second)
}

func TestWSSTTSession_UsageIgnoresDroppedFrames(t *testing.T) {
	// Never started: every push drops, so no audio time accrues.
	var s wsSTTSession
	s.pushAudio(make([]byte, 8000))
	s.pushAudio(make([]byte, 4000))
	if got := s.usage().AudioSeconds; got != 0 {
		t.Errorf("AudioSeconds for dropped-only frames = %v, want 0", got)
	}
}

func TestWSSTTSession_CtxCancelUnblocksReader(t *testing.T) {
	// Cancelling ctx alone — no Stop call, server keeps the conn open —
	// must close the conn so a reader parked in ReadMessage unblocks
	// and the events channel closes, with no STTError (graceful path).
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				return
			}
		}
	})
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s := &wsSTTSession{name: "test-stt"}
	conn, err := dialSTT(ctx, wsURL, nil, 2*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	events := make(chan STTEvent, 8)
	reader := func(ctx context.Context, conn *websocket.Conn, events chan<- STTEvent, stopCh <-chan struct{}) {
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				if !readErrorIsExpected(ctx, stopCh) {
					emitSTT(events, STTEvent{Type: STTError, Err: err})
				}
				return
			}
		}
	}
	if err := s.start(ctx, conn, events, reader); err != nil {
		t.Fatalf("start: %v", err)
	}

	cancel()
	for _, ev := range drainUntilClosed(t, events, 2*time.Second) {
		if ev.Type == STTError {
			t.Errorf("STTError on ctx cancel: %v", ev.Err)
		}
	}
	if err := s.stop(); err != nil {
		t.Fatalf("stop after cancel: %v", err)
	}
}

func TestWSSTTSession_StopBeforeStartAndIdempotent(t *testing.T) {
	var s wsSTTSession
	if err := s.stop(); err != nil {
		t.Fatalf("stop before start: %v", err)
	}
	if err := s.stop(); err != nil {
		t.Fatalf("second stop before start: %v", err)
	}
}

func TestWSSTTSession_StartTwiceErrors(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				return
			}
		}
	})
	defer cleanup()

	conn, err := dialSTT(context.Background(), wsURL, nil, 5*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}

	s := &wsSTTSession{name: "test-stt"}
	events := make(chan STTEvent, 8)
	if err := s.start(context.Background(), conn, events, blockingReader); err != nil {
		t.Fatalf("start: %v", err)
	}
	defer s.stop()

	if err := s.start(context.Background(), conn, make(chan STTEvent, 1), blockingReader); err == nil {
		t.Fatalf("second start succeeded, want 'already started' error")
	} else if !strings.Contains(err.Error(), "already started") {
		t.Fatalf("second start error = %v, want 'already started'", err)
	}
}

func TestWSSTTSession_StopClosesEventsAndRestarts(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				return
			}
		}
	})
	defer cleanup()

	s := &wsSTTSession{name: "test-stt"}

	for cycle := 0; cycle < 2; cycle++ {
		conn, err := dialSTT(context.Background(), wsURL, nil, 5*time.Second)
		if err != nil {
			t.Fatalf("cycle %d dial: %v", cycle, err)
		}
		events := make(chan STTEvent, 8)
		if err := s.start(context.Background(), conn, events, blockingReader); err != nil {
			t.Fatalf("cycle %d start: %v", cycle, err)
		}
		if err := s.stop(); err != nil {
			t.Fatalf("cycle %d stop: %v", cycle, err)
		}
		drainUntilClosed(t, events, 2*time.Second)

		// Repeated stop is a no-op.
		if err := s.stop(); err != nil {
			t.Fatalf("cycle %d repeated stop: %v", cycle, err)
		}
		// pushAudio after stop must not block or panic.
		pushed := make(chan struct{})
		go func() {
			s.pushAudio(make([]byte, 160))
			close(pushed)
		}()
		select {
		case <-pushed:
		case <-time.After(time.Second):
			t.Fatalf("cycle %d: pushAudio after stop blocked", cycle)
		}
	}
}

func TestWSSTTSession_WriteMessageAfterStop(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				return
			}
		}
	})
	defer cleanup()

	conn, err := dialSTT(context.Background(), wsURL, nil, 5*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	s := &wsSTTSession{name: "test-stt"}
	events := make(chan STTEvent, 8)
	if err := s.start(context.Background(), conn, events, blockingReader); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := s.stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if err := s.writeMessage(websocket.TextMessage, []byte("x")); err != errSessionClosed {
		t.Fatalf("writeMessage after stop = %v, want errSessionClosed", err)
	}
}

func TestWSSTTSession_WriterForwardsFramesAsBinary(t *testing.T) {
	type binMsg struct {
		mt   int
		data []byte
	}
	received := make(chan binMsg, 16)
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for {
			mt, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			received <- binMsg{mt: mt, data: data}
		}
	})
	defer cleanup()

	conn, err := dialSTT(context.Background(), wsURL, nil, 5*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	s := &wsSTTSession{name: "test-stt"}
	events := make(chan STTEvent, 8)
	if err := s.start(context.Background(), conn, events, blockingReader); err != nil {
		t.Fatalf("start: %v", err)
	}
	defer s.stop()

	frames := [][]byte{{0x01, 0x02}, {0x03, 0x04, 0x05}, {0xff}}
	for _, f := range frames {
		s.pushAudio(f)
	}

	for i, want := range frames {
		select {
		case got := <-received:
			if got.mt != websocket.BinaryMessage {
				t.Fatalf("frame %d message type = %d, want BinaryMessage", i, got.mt)
			}
			if string(got.data) != string(want) {
				t.Fatalf("frame %d = %v, want %v", i, got.data, want)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timeout waiting for forwarded frame %d", i)
		}
	}

	if got := s.framesForwarded.Load(); got != int64(len(frames)) {
		t.Errorf("framesForwarded = %d, want %d", got, len(frames))
	}
}

// TestWSSTTSession_WriterHandlesWriteFailure kills the transport out
// from under a running session (without stopping it) and verifies the
// writer takes the error branch — no forward count, no hang, and stop
// still completes cleanly afterwards.
func TestWSSTTSession_WriterHandlesWriteFailure(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				return
			}
		}
	})
	defer cleanup()

	conn, err := dialSTT(context.Background(), wsURL, nil, 5*time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	s := &wsSTTSession{name: "test-stt"}
	events := make(chan STTEvent, 8)
	if err := s.start(context.Background(), conn, events, blockingReader); err != nil {
		t.Fatalf("start: %v", err)
	}

	// Kill the conn directly — the session still believes it is live, so
	// the next writeMessage returns a real transport error (not
	// errSessionClosed).
	_ = conn.Close()
	s.pushAudio([]byte{0x01, 0x02, 0x03})

	// Wait for the writer to consume the frame and hit the error path:
	// the frame leaves audioIn but framesForwarded never increments.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(s.audioIn) == 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if len(s.audioIn) != 0 {
		t.Fatalf("writer never consumed the frame after transport failure")
	}
	if got := s.framesForwarded.Load(); got != 0 {
		t.Errorf("framesForwarded = %d, want 0 after write failure", got)
	}

	done := make(chan error, 1)
	go func() { done <- s.stop() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("stop after write failure: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("stop hung after writer error")
	}
	drainUntilClosed(t, events, 2*time.Second)
}

// newNonUpgradingServer serves 401 to every request — simulates an
// auth-rejected websocket dial (no 101 upgrade).
func newNonUpgradingServer() (wsURL string, cleanup func()) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "invalid api key", http.StatusUnauthorized)
	}))
	return "ws" + strings.TrimPrefix(srv.URL, "http"), srv.Close
}

func TestWSSTTDialSTT_NonUpgradeStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "invalid api key", http.StatusUnauthorized)
	}))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	conn, err := dialSTT(context.Background(), wsURL, nil, 5*time.Second)
	if err == nil {
		conn.Close()
		t.Fatalf("dialSTT succeeded against non-upgrading server, want error")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("dial error = %v, want it to mention status 401", err)
	}
	if !strings.Contains(err.Error(), wsURL) {
		t.Errorf("dial error = %v, want it to mention the URL", err)
	}
}

func TestWSSTTDialSTT_Unreachable(t *testing.T) {
	// Grab a port that is closed by binding then immediately releasing it.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")
	srv.Close()

	conn, err := dialSTT(context.Background(), wsURL, nil, 2*time.Second)
	if err == nil {
		conn.Close()
		t.Fatalf("dialSTT to closed port succeeded, want error")
	}
	if !strings.Contains(err.Error(), "dial") {
		t.Errorf("dial error = %v, want it to mention dial failure", err)
	}
}

func TestReadErrorIsExpected(t *testing.T) {
	stopCh := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if readErrorIsExpected(ctx, stopCh) {
		t.Errorf("expected false with live ctx and open stopCh")
	}
	close(stopCh)
	if !readErrorIsExpected(ctx, stopCh) {
		t.Errorf("expected true after stopCh close")
	}

	stopCh2 := make(chan struct{})
	cancel()
	if !readErrorIsExpected(ctx, stopCh2) {
		t.Errorf("expected true after ctx cancel")
	}
}

func TestEmitSTTNonBlocking(t *testing.T) {
	full := make(chan STTEvent) // unbuffered, no receiver
	done := make(chan struct{})
	go func() {
		emitSTT(full, STTEvent{Type: STTTranscriptDelta, Text: "dropped"})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatalf("emitSTT blocked on a full channel")
	}
}
