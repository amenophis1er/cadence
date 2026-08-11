package engines

import (
	"context"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// A speaker mid-sentence keeps producing partials between is_final segments.
// The debounce must measure SILENCE, not time-since-the-last-is_final: if
// partials do not hold it off, a long utterance is committed mid-breath and
// the remainder arrives as a second turn — the caller gets cut in half.
func TestDeepgramSTT_PartialsHoldOffTheDebounce(t *testing.T) {
	const flushMs = 200

	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		send := func(s string) bool { return ws.WriteMessage(websocket.TextMessage, []byte(s)) == nil }
		// One is_final opens the utterance and arms the debounce.
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"my number is"}]},"is_final":true,"speech_final":false}`) {
			return
		}
		// The speaker keeps going — partials only, spanning well beyond the
		// flush window in total, but never idle for that long at once.
		for _, p := range []string{"my number is eight", "my number is eight eight", "my number is eight eight eight"} {
			time.Sleep(120 * time.Millisecond)
			if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"` + p + `"}]},"is_final":false}`) {
				return
			}
		}
		// Then they finish, and Deepgram says so.
		time.Sleep(120 * time.Millisecond)
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"eight eight eight"}]},"is_final":true,"speech_final":true}`) {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = flushMs
	})
	events := make(chan STTEvent, 64)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	ev := recvSTTFinal(t, events)

	// The whole breath must arrive as ONE turn, ended by the provider's own
	// signal — not chopped by the debounce while the speaker was mid-sentence.
	if ev.CommittedBy != CommitSpeechFinal {
		t.Errorf("CommittedBy = %q, want %q: the debounce fired while the speaker was still talking",
			ev.CommittedBy, CommitSpeechFinal)
	}
	if ev.Text != "my number is eight eight eight" {
		t.Errorf("committed %q — the utterance was split", ev.Text)
	}
}

// The debounce must still fire when the speaker genuinely stops: holding it
// off on partials must not disable the fallback entirely.
func TestDeepgramSTT_DebounceStillFiresOnRealSilence(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msg := `{"type":"Results","channel":{"alternatives":[{"transcript":"okay"}]},"is_final":true,"speech_final":false}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(msg)); err != nil {
			return
		}
		deepgramIgnoreAudio(ws) // nothing further: real silence
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 150
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	ev := recvSTTFinal(t, events)
	if ev.CommittedBy != CommitFlushTimeout {
		t.Errorf("CommittedBy = %q, want %q", ev.CommittedBy, CommitFlushTimeout)
	}
	if ev.Text != "okay" {
		t.Errorf("committed %q", ev.Text)
	}
}

// Partials arriving with NOTHING buffered must not arm a timer: there is no
// utterance to flush, and arming one would commit an empty turn later.
func TestDeepgramSTT_PartialsBeforeAnyFinalDoNotArm(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		send := func(s string) bool { return ws.WriteMessage(websocket.TextMessage, []byte(s)) == nil }
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"hel"}]},"is_final":false}`) {
			return
		}
		time.Sleep(250 * time.Millisecond) // longer than the flush window
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"hello"}]},"is_final":true,"speech_final":true}`) {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 150
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	ev := recvSTTFinal(t, events)
	if ev.Text != "hello" || ev.CommittedBy != CommitSpeechFinal {
		t.Errorf("got %q by %q; an empty turn was flushed from a bare partial", ev.Text, ev.CommittedBy)
	}
}
