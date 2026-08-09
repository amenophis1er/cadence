package engines

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// wsSTTSession owns the WebSocket lifecycle shared by streaming STT
// engines (Deepgram, Cartesia, future Speechmatics/AssemblyAI/etc.).
// Each engine wraps a session and supplies its own dial logic + reader
// function; this type handles the rest of the boilerplate (audioIn
// channel, runWriter, Stop, mutex, error-during-shutdown suppression).
//
// Lifecycle: each engine creates a fresh session per Start call (or
// reuses one across Stop/Start cycles — Stop resets the closed flag so
// the session can be re-started). Concurrent Start on the same session
// returns an error.
type wsSTTSession struct {
	// name is the engine name used in error messages and log lines
	// (e.g. "deepgram-stt", "cartesia-stt").
	name string

	// callSID is the Twilio CallSid for this session, included in every
	// diagnostic slog line so per-call triage works when multiple calls
	// are concurrent on a single instance. Set by the embedding engine
	// before start; empty string if the engine has not been wired up
	// with the call's SID (legacy code path / unit tests).
	callSID string

	mu     sync.Mutex
	conn   *websocket.Conn
	closed bool

	// writeMu serializes WriteMessage calls on conn. gorilla/websocket
	// permits one concurrent reader and one concurrent writer; without
	// this lock, runWriter (audio frames) and engine-level shutdown
	// senders (Deepgram's CloseStream sentinel) could interleave and
	// corrupt the WS frame stream. All writes go through writeMessage.
	writeMu sync.Mutex

	audioIn chan []byte
	stopCh  chan struct{}
	wg      sync.WaitGroup

	// audioSeconds accumulates the duration of audio actually forwarded to
	// the vendor (accrued by runWriter after a successful write), for cost
	// attribution. Frames dropped on overrun or pushed after Stop are NOT
	// counted — usage tracks what the vendor could have billed, not what
	// the caller offered. Assumes mu-law 8 kHz (the documented contract
	// for Streaming STT engines, see PushAudio doc on StreamingSTTEngine).
	audioSeconds float64

	// Per-session counters for the end-of-call summary line. Atomic so the
	// pushAudio (caller goroutine) and runWriter (writer goroutine) can
	// update without holding mu.
	framesReceived  atomic.Int64
	framesForwarded atomic.Int64
	framesDropped   atomic.Int64
}

// sttReaderFn is the engine-specific message reader. It blocks on
// `conn.ReadMessage` and translates inbound envelopes into STTEvents
// emitted on `events`. It must NOT call `close(events)` itself — the
// session wrapper handles that. On read error, callers should consult
// `ctx.Done()` and `stopCh` to distinguish graceful shutdown (no
// STTError emit) from a real failure (emit STTError before returning).
type sttReaderFn func(ctx context.Context, conn *websocket.Conn, events chan<- STTEvent, stopCh <-chan struct{})

// start spawns the reader and writer goroutines on the supplied conn.
// The caller is responsible for the dial (URL/auth differ per vendor)
// and must close the conn on dial failure before calling start.
//
// On error return, the caller still owns the conn and must close it.
// On success, the session takes ownership — Stop will close.
func (s *wsSTTSession) start(ctx context.Context, conn *websocket.Conn, events chan<- STTEvent, reader sttReaderFn) error {
	s.mu.Lock()
	if s.conn != nil {
		s.mu.Unlock()
		return fmt.Errorf("%s: already started", s.name)
	}
	s.audioIn = make(chan []byte, 256)
	s.stopCh = make(chan struct{})
	s.conn = conn
	// Reset closed so the session can be re-started after a prior
	// Stop. Without this reset, a Stop/Start cycle would leave
	// closed=true and the next Stop would short-circuit, leaking the
	// re-started goroutines.
	s.closed = false
	s.mu.Unlock()

	s.wg.Add(2)
	go func() {
		defer s.wg.Done()
		defer close(events)
		reader(ctx, conn, events, s.stopCh)
	}()
	go s.runWriter(ctx)

	// Watch for context cancellation and close the conn so the reader's
	// blocked ReadMessage unblocks immediately. Without this, cancelling
	// ctx only stopped the writer — the reader stayed parked until the
	// peer closed the socket or Stop was called, so "barge-in is context
	// cancellation" didn't hold for STT. readErrorIsExpected sees the
	// cancelled ctx, so teardown stays graceful (no STTError emitted).
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		select {
		case <-ctx.Done():
			s.mu.Lock()
			conn := s.conn
			s.mu.Unlock()
			if conn != nil {
				_ = conn.Close()
			}
		case <-s.stopCh:
		}
	}()
	return nil
}

// stop closes the websocket and waits for the reader+writer to exit.
// Idempotent — repeated calls are no-ops, and calling stop before
// start is safe (it short-circuits, since stopCh is only initialised
// in start).
func (s *wsSTTSession) stop() error {
	s.mu.Lock()
	if s.closed || s.stopCh == nil {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	close(s.stopCh)
	conn := s.conn
	s.conn = nil
	s.mu.Unlock()

	if conn != nil {
		_ = conn.Close()
	}
	s.wg.Wait()
	return nil
}

// errSessionClosed is returned by writeMessage when the session has
// already been stopped or never started. Callers (writer goroutine,
// engine shutdown senders) treat it as a clean stop signal.
var errSessionClosed = fmt.Errorf("wsSTTSession: closed")

// writeMessage sends one WS message under writeMu so audio writes from
// runWriter and engine-level control sends (Deepgram's CloseStream,
// future vendor sentinels) can't interleave. Returns errSessionClosed
// if the session has been stopped — caller decides whether that is a
// graceful exit or an error.
//
// We re-read s.conn under s.mu inside the locked section: between a
// caller's check and the write, stop() may have nilled the conn. This
// keeps the contract consistent without exposing the raw conn.
func (s *wsSTTSession) writeMessage(messageType int, data []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	s.mu.Lock()
	conn := s.conn
	closed := s.closed
	s.mu.Unlock()
	if closed || conn == nil {
		return errSessionClosed
	}
	return conn.WriteMessage(messageType, data)
}

// pushAudio enqueues one audio frame for forwarding. Drops on overrun
// rather than blocking, preserving real-time behaviour over completeness.
func (s *wsSTTSession) pushAudio(frame []byte) {
	s.framesReceived.Add(1)
	select {
	case s.audioIn <- frame:
	default:
		s.framesDropped.Add(1)
	}
}

// usage returns cumulative audio time forwarded to the vendor.
func (s *wsSTTSession) usage() Usage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return Usage{AudioSeconds: s.audioSeconds}
}

// runWriter forwards audio frames as binary WebSocket messages. The
// engine is responsible for ensuring the frames are in the format the
// server expects (encoding/sample-rate are configured at dial time via
// query params or handshake). On WriteMessage failure we close the
// conn so the reader's blocked ReadMessage unblocks and surfaces an
// STTError — without this the writer would die silently and the engine
// would sit idle until TCP keepalive timed out.
func (s *wsSTTSession) runWriter(ctx context.Context) {
	defer s.wg.Done()
	var firstFrameOnce sync.Once
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.stopCh:
			return
		case frame, ok := <-s.audioIn:
			if !ok {
				return
			}
			err := s.writeMessage(websocket.BinaryMessage, frame)
			if err == errSessionClosed {
				return
			}
			if err != nil {
				// Surface the failure — silent close here is what made
				// "silent call" bugs (no STT events for the whole call)
				// invisible: the writer would die, the reader would
				// eventually see the closed conn, but neither path
				// logged anything specific to the underlying transport
				// error. Gate on readErrorIsExpected so graceful
				// shutdown (stop() closes the conn while we're mid-Write)
				// doesn't WARN on every call hangup — match the reader's
				// own suppression policy.
				if !readErrorIsExpected(ctx, s.stopCh) {
					slog.Warn("cadence: STT writer WriteMessage failed; closing conn",
						"engine", s.name,
						"call_sid", s.callSID,
						"error", err,
						"frame_bytes", len(frame))
				}
				// Take the same conn pointer used by writeMessage and
				// close it to unblock the reader's ReadMessage.
				s.mu.Lock()
				conn := s.conn
				s.mu.Unlock()
				if conn != nil {
					_ = conn.Close()
				}
				return
			}
			s.framesForwarded.Add(1)
			s.mu.Lock()
			s.audioSeconds += float64(len(frame)) / 8000
			s.mu.Unlock()
			firstFrameOnce.Do(func() {
				slog.Info("cadence: STT writer forwarded first audio frame",
					"engine", s.name,
					"call_sid", s.callSID,
					"frame_bytes", len(frame))
			})
		}
	}
}

// readErrorIsExpected reports whether a WebSocket read error was caused
// by graceful shutdown (ctx cancelled or Stop called) rather than a
// real failure. Engine readers use this to suppress STTError emission
// during clean teardown.
func readErrorIsExpected(ctx context.Context, stopCh <-chan struct{}) bool {
	select {
	case <-ctx.Done():
		return true
	case <-stopCh:
		return true
	default:
		return false
	}
}

// emitSTT is a non-blocking send helper. Drops events rather than
// blocking the read loop when the consumer is slow — same contract as
// the per-engine `emit` methods we used to maintain individually.
func emitSTT(events chan<- STTEvent, ev STTEvent) {
	select {
	case events <- ev:
	default:
	}
}

// dialSTT wraps websocket.Dialer.DialContext and ensures the failed
// HTTP upgrade response (which gorilla returns on dial error) gets
// closed. Without the explicit Close, every failed dial leaks one HTTP
// connection — observable on auth-rejected keys, regional outages, or
// 429 floods.
func dialSTT(ctx context.Context, wsURL string, headers http.Header, handshakeTimeout time.Duration) (*websocket.Conn, error) {
	dialer := websocket.Dialer{HandshakeTimeout: handshakeTimeout}
	conn, resp, err := dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		if resp != nil {
			defer resp.Body.Close()
			return nil, fmt.Errorf("dial %s: %w (status %s)", wsURL, err, resp.Status)
		}
		return nil, fmt.Errorf("dial %s: %w", wsURL, err)
	}
	return conn, nil
}
