package engines

import (
	"context"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// Deepgram delivers the words in one segment and the end-of-utterance decision
// in a SEPARATE envelope whose transcript is empty. The engine must honour
// that signal: dropping it leaves a finished caller waiting out the whole
// flush debounce for nothing.
func TestDeepgramSTT_EmptyFinalCarriesSpeechFinal(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		// Words, but Deepgram has not decided the turn is over yet.
		words := `{"type":"Results","channel":{"alternatives":[{"transcript":"eight eight eight"}]},"is_final":true,"speech_final":false}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(words)); err != nil {
			return
		}
		// The endpointing decision, carrying no transcript of its own.
		endOfSpeech := `{"type":"Results","channel":{"alternatives":[{"transcript":""}]},"is_final":true,"speech_final":true}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(endOfSpeech)); err != nil {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	// A long debounce: if the empty final were ignored, this test would only
	// pass by waiting it out — so a prompt commit proves the signal was used.
	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 10000
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")

	start := time.Now()
	ev := recvSTTFinal(t, events)
	elapsed := time.Since(start)

	if ev.Text != "eight eight eight" {
		t.Fatalf("committed text = %q", ev.Text)
	}
	if ev.CommittedBy != CommitSpeechFinal {
		t.Errorf("CommittedBy = %q, want %q — the provider's own signal was ignored",
			ev.CommittedBy, CommitSpeechFinal)
	}
	if elapsed > 2*time.Second {
		t.Errorf("commit took %v: the utterance waited on the debounce instead of the signal", elapsed)
	}
}

// An empty INTERIM still carries nothing and must stay ignored — the fix must
// not turn every silent envelope into a turn boundary.
func TestDeepgramSTT_EmptyInterimStillIgnored(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		for _, m := range []string{
			`{"type":"Results","channel":{"alternatives":[{"transcript":""}]},"is_final":false,"speech_final":false}`,
			`{"type":"Results","channel":{"alternatives":[{"transcript":""}]},"is_final":true,"speech_final":false}`,
			`{"type":"Results","channel":{"alternatives":[{"transcript":"hello"}]},"is_final":true,"speech_final":true}`,
		} {
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

	// The first real event must be the speech-started for "hello" — the two
	// empty envelopes before it produce nothing.
	expectSTTEvent(t, events, STTSpeechStarted, "")
	ev := recvSTTFinal(t, events)
	if ev.Text != "hello" {
		t.Fatalf("committed %q, want the only real utterance", ev.Text)
	}
}

// A bare end-of-speech marker with nothing buffered must not emit an empty
// turn — the agent would answer silence.
func TestDeepgramSTT_EmptyFinalWithNothingBufferedEmitsNothing(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		bare := `{"type":"Results","channel":{"alternatives":[{"transcript":""}]},"is_final":true,"speech_final":true}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(bare)); err != nil {
			return
		}
		time.Sleep(150 * time.Millisecond)
		real := `{"type":"Results","channel":{"alternatives":[{"transcript":"yes"}]},"is_final":true,"speech_final":true}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(real)); err != nil {
			return
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
	if ev := recvSTTFinal(t, events); ev.Text != "yes" {
		t.Fatalf("first committed final = %q, want %q (an empty turn was emitted)", ev.Text, "yes")
	}
}
