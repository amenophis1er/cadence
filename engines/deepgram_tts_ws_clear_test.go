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

// The engine must satisfy the optional interrupt capability.
func TestDeepgramTTSWS_ImplementsInterruptible(t *testing.T) {
	var eng any = newTestEngine("ws://example.invalid")
	if _, ok := eng.(InterruptibleTTSEngine); !ok {
		t.Fatal("deepgram TTS WS engine must implement InterruptibleTTSEngine")
	}
}

// Barge-in: Clear reaches the wire, audio synthesised before the
// interruption is dropped by the engine, the session stays open, and text
// pushed afterwards is spoken on the SAME connection.
func TestDeepgramTTSWS_ClearDropsStaleAudioAndKeepsSession(t *testing.T) {
	cleared := make(chan struct{})
	secondSpeak := make(chan string, 1)

	wsURL, cleanup := mockDeepgramTTSServer(t, "", func(t *testing.T, ws *websocket.Conn) {
		// Audio for the first sentence, then wait for the Clear.
		ws.WriteMessage(websocket.BinaryMessage, []byte("stale-1"))

		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if json.Unmarshal(data, &env) != nil {
				continue
			}
			switch env.Type {
			case "Clear":
				// A real server may still emit audio queued before the
				// Clear; the engine must swallow it.
				ws.WriteMessage(websocket.BinaryMessage, []byte("stale-2"))
				ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Cleared"}`))
				close(cleared)
			case "Speak":
				if env.Text != "first sentence" {
					select {
					case secondSpeak <- env.Text:
					default:
					}
					// Post-interruption audio must reach the consumer.
					ws.WriteMessage(websocket.BinaryMessage, []byte("fresh"))
				}
			case "Close":
				// Deepgram closes on the Close sentinel; mirror it so the
				// engine's receiver unblocks and Stream returns.
				_ = ws.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
				return
			}
		}
	})
	defer cleanup()

	eng := newTestEngine(wsURL)
	textCh := make(chan TextChunk, 4)
	audioCh := make(chan AudioChunk, 16)

	done := make(chan error, 1)
	go func() { done <- eng.Stream(context.Background(), textCh, audioCh) }()

	textCh <- TextChunk{Text: "first sentence", IsSentenceEnd: true}

	// Wait until the first (stale) frame has been produced, so the test
	// exercises the real race: audio already in flight when Clear lands.
	select {
	case <-audioCh:
	case <-time.After(3 * time.Second):
		t.Fatal("no initial audio")
	}

	if err := eng.Clear(); err != nil {
		t.Fatalf("Clear: %v", err)
	}
	select {
	case <-cleared:
	case <-time.After(3 * time.Second):
		t.Fatal("server never saw the Clear control message")
	}

	// Same session: speak again without re-dialling.
	textCh <- TextChunk{Text: "after barge-in", IsSentenceEnd: true}
	select {
	case got := <-secondSpeak:
		if got != "after barge-in" {
			t.Fatalf("second Speak = %q", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("post-Clear text was never spoken — session did not survive")
	}

	// Everything the consumer receives from here must be post-interruption
	// audio; the pre-Clear frame must have been swallowed by the engine.
	deadline := time.After(3 * time.Second)
	for {
		select {
		case chunk := <-audioCh:
			if string(chunk.Data) == "stale-2" {
				t.Fatal("stale audio from before the interruption reached the consumer")
			}
			if string(chunk.Data) == "fresh" {
				close(textCh)
				<-done
				return
			}
		case <-deadline:
			t.Fatal("post-Clear audio never arrived")
		}
	}
}

// Two rapid barge-ins mean two outstanding Clears. The FIRST Cleared ack
// must not lower the discard gate while the second is still pending —
// audio arriving between the two acks predates an interruption.
func TestDeepgramTTSWS_DoubleClearKeepsGateUntilLastAck(t *testing.T) {
	sawBothClears := make(chan struct{})

	wsURL, cleanup := mockDeepgramTTSServer(t, "", func(t *testing.T, ws *websocket.Conn) {
		clears := 0
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env struct {
				Type string `json:"type"`
			}
			if json.Unmarshal(data, &env) != nil {
				continue
			}
			switch env.Type {
			case "Speak":
				// Lets the test observe that the session is live before
				// it fires the barge-ins (Clear pre-connect is a no-op).
				ws.WriteMessage(websocket.BinaryMessage, []byte("warmup"))
			case "Clear":
				clears++
				if clears == 2 {
					// Ack the first Clear, then emit audio that belongs
					// to the window before the SECOND Clear's ack, then
					// ack the second and emit genuinely fresh audio.
					ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Cleared"}`))
					ws.WriteMessage(websocket.BinaryMessage, []byte("stale-between-acks"))
					ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Cleared"}`))
					ws.WriteMessage(websocket.BinaryMessage, []byte("fresh"))
					close(sawBothClears)
				}
			case "Close":
				_ = ws.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
				return
			}
		}
	})
	defer cleanup()

	eng := newTestEngine(wsURL)
	textCh := make(chan TextChunk, 4)
	audioCh := make(chan AudioChunk, 16)
	done := make(chan error, 1)
	go func() { done <- eng.Stream(context.Background(), textCh, audioCh) }()

	// First chunk triggers the lazy connect; wait for the warmup frame so
	// the connection is provably live before barging in.
	textCh <- TextChunk{Text: "hello", IsSentenceEnd: true}
	select {
	case <-audioCh:
	case <-time.After(3 * time.Second):
		t.Fatal("session never came up")
	}

	if err := eng.Clear(); err != nil {
		t.Fatalf("first Clear: %v", err)
	}
	if err := eng.Clear(); err != nil {
		t.Fatalf("second Clear: %v", err)
	}
	select {
	case <-sawBothClears:
	case <-time.After(3 * time.Second):
		t.Fatal("server never saw both Clear messages")
	}

	deadline := time.After(3 * time.Second)
	for {
		select {
		case chunk := <-audioCh:
			if string(chunk.Data) == "stale-between-acks" {
				t.Fatal("audio from between the two Cleared acks reached the consumer")
			}
			if string(chunk.Data) == "fresh" {
				close(textCh)
				<-done
				return
			}
		case <-deadline:
			t.Fatal("post-ack audio never arrived — gate stuck raised")
		}
	}
}

// A frame that passed the discard gate but is blocked waiting for room in a
// full audioCh predates any Clear that lands meanwhile — it must be aborted,
// not delivered whenever the consumer next drains.
func TestDeepgramTTSWS_ClearAbortsBlockedForward(t *testing.T) {
	sawClear := make(chan struct{})

	wsURL, cleanup := mockDeepgramTTSServer(t, "", func(t *testing.T, ws *websocket.Conn) {
		// Emit the stale frame immediately; the consumer isn't draining,
		// so the engine's forward will block on the unbuffered channel.
		ws.WriteMessage(websocket.BinaryMessage, []byte("stale-blocked"))
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env struct {
				Type string `json:"type"`
			}
			if json.Unmarshal(data, &env) != nil {
				continue
			}
			switch env.Type {
			case "Clear":
				ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Cleared"}`))
				ws.WriteMessage(websocket.BinaryMessage, []byte("fresh"))
				close(sawClear)
			case "Close":
				_ = ws.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
				return
			}
		}
	})
	defer cleanup()

	eng := newTestEngine(wsURL)
	textCh := make(chan TextChunk, 4)
	audioCh := make(chan AudioChunk) // unbuffered: the forward must block
	done := make(chan error, 1)
	go func() { done <- eng.Stream(context.Background(), textCh, audioCh) }()

	textCh <- TextChunk{Text: "hello", IsSentenceEnd: true}

	// Give the receiver time to pick up the stale frame and block on the
	// send; then barge in while it is stuck there.
	time.Sleep(200 * time.Millisecond)
	if err := eng.Clear(); err != nil {
		t.Fatalf("Clear: %v", err)
	}
	select {
	case <-sawClear:
	case <-time.After(3 * time.Second):
		t.Fatal("server never saw the Clear")
	}

	// Only NOW does the consumer start draining. The blocked stale frame
	// must have been aborted; the first thing out must be post-Clear audio.
	select {
	case chunk := <-audioCh:
		if string(chunk.Data) != "fresh" {
			t.Fatalf("first drained chunk = %q, want the post-Clear frame", chunk.Data)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("post-Clear audio never arrived")
	}
	close(textCh)
	<-done
}

// A Clear that lands while the lazy connect is still dialling has no conn to
// write to — but the engine is already holding a dequeued text chunk, and
// speaking it once the connection comes up would voice pre-interruption text
// after a successful Clear. The interruption epoch must drop it.
func TestDeepgramTTSWS_ClearDuringDialDropsHeldText(t *testing.T) {
	upgrader := websocket.Upgrader{}
	dialGate := make(chan struct{})
	spoke := make(chan string, 4)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-dialGate // hold the handshake so the Clear lands mid-dial
		ws, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer ws.Close()
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if json.Unmarshal(data, &env) != nil {
				continue
			}
			switch env.Type {
			case "Speak":
				spoke <- env.Text
			case "Close":
				_ = ws.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
				return
			}
		}
	}))
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	eng := newTestEngine(wsURL)
	textCh := make(chan TextChunk, 4)
	audioCh := make(chan AudioChunk, 16)
	done := make(chan error, 1)
	go func() { done <- eng.Stream(context.Background(), textCh, audioCh) }()

	textCh <- TextChunk{Text: "held across dial", IsSentenceEnd: true}
	// Let Stream dequeue the chunk and block in the (gated) handshake.
	time.Sleep(100 * time.Millisecond)
	if err := eng.Clear(); err != nil {
		t.Fatalf("Clear during dial: %v", err)
	}
	close(dialGate)

	textCh <- TextChunk{Text: "after barge-in", IsSentenceEnd: true}
	select {
	case got := <-spoke:
		if got != "after barge-in" {
			t.Fatalf("engine spoke %q — text held across the interruption was voiced", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("post-Clear text never spoken")
	}
	close(textCh)
	<-done
}

// A Clear racing session teardown must never produce two concurrent
// websocket writers: the sender's Close sentinel and Clear's control
// message share connMu, and gorilla panics (not errors) on concurrent
// writes. Repeats the race enough times to make an unguarded write fail
// reliably under -race.
func TestDeepgramTTSWS_ClearRacesTeardown(t *testing.T) {
	for i := 0; i < 20; i++ {
		wsURL, cleanup := mockDeepgramTTSServer(t, "", func(t *testing.T, ws *websocket.Conn) {
			for {
				_, data, err := ws.ReadMessage()
				if err != nil {
					return
				}
				var env struct {
					Type string `json:"type"`
				}
				if json.Unmarshal(data, &env) == nil && env.Type == "Close" {
					_ = ws.WriteMessage(websocket.CloseMessage,
						websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
					return
				}
			}
		})

		eng := newTestEngine(wsURL)
		textCh := make(chan TextChunk, 1)
		audioCh := make(chan AudioChunk, 16)
		done := make(chan error, 1)
		go func() { done <- eng.Stream(context.Background(), textCh, audioCh) }()

		// First chunk triggers the lazy connect so a live conn exists.
		textCh <- TextChunk{Text: "hello", IsSentenceEnd: true}

		// Hammer Clear from several goroutines until the stream is fully
		// down, so one of them overlaps the sender's Close sentinel write.
		stop := make(chan struct{})
		var wg sync.WaitGroup
		for g := 0; g < 4; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					select {
					case <-stop:
						return
					default:
						_ = eng.Clear()
					}
				}
			}()
		}
		close(textCh)

		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("Stream did not return after teardown")
		}
		close(stop)
		wg.Wait()
		cleanup()
	}
}

// Clear with no live stream is a no-op, not an error or a panic — a caller
// racing a hangup against a barge-in must not be punished for it.
func TestDeepgramTTSWS_ClearWithoutLiveStream(t *testing.T) {
	eng := newTestEngine("ws://example.invalid")
	if err := eng.Clear(); err != nil {
		t.Fatalf("Clear with no stream should be a no-op, got %v", err)
	}
}
