package engines

import (
	"context"
	"encoding/json"
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

// Clear with no live stream is a no-op, not an error or a panic — a caller
// racing a hangup against a barge-in must not be punished for it.
func TestDeepgramTTSWS_ClearWithoutLiveStream(t *testing.T) {
	eng := newTestEngine("ws://example.invalid")
	if err := eng.Clear(); err != nil {
		t.Fatalf("Clear with no stream should be a no-op, got %v", err)
	}
}
