package engines

import (
	"context"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// A partial stream that never stops must not hold a turn open forever: the
// caller is waiting for a reply. The ceiling commits the utterance and says
// so, so the wait is bounded even when the provider never signals the end.
func TestDeepgramSTT_MaxHoldBoundsAnEndlessPartialStream(t *testing.T) {
	const maxHoldMs = 400

	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		send := func(s string) bool { return ws.WriteMessage(websocket.TextMessage, []byte(s)) == nil }
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"still going"}]},"is_final":true,"speech_final":false}`) {
			return
		}
		// Partials WITH TEXT, arriving faster than the debounce, forever.
		// Without a ceiling these re-arm the timer indefinitely.
		for i := 0; i < 60; i++ {
			time.Sleep(60 * time.Millisecond)
			if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"still going on"}]},"is_final":false}`) {
				return
			}
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 1000 // longer than the ceiling, so the cap decides
		cfg.MaxHoldMs = maxHoldMs
	})
	events := make(chan STTEvent, 64)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	start := time.Now()
	ev := recvSTTFinal(t, events)
	elapsed := time.Since(start)

	if ev.CommittedBy != CommitMaxHold {
		t.Errorf("CommittedBy = %q, want %q — the turn was not bounded", ev.CommittedBy, CommitMaxHold)
	}
	// The partial stream runs ~3.6s; committing near the ceiling proves the
	// bound applied rather than the stream simply ending.
	if elapsed > 2*time.Second {
		t.Errorf("commit took %v: the ceiling did not bound the hold", elapsed)
	}
}

// The ceiling is measured from the utterance's START, not from the last
// re-arm — otherwise a steady partial stream keeps resetting the very bound
// meant to contain it, and the cap never fires.
func TestDeepgramSTT_MaxHoldMeasuredFromUtteranceStart(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		send := func(s string) bool { return ws.WriteMessage(websocket.TextMessage, []byte(s)) == nil }
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"one"}]},"is_final":true,"speech_final":false}`) {
			return
		}
		// Re-arm repeatedly at an interval well under the debounce.
		for i := 0; i < 40; i++ {
			time.Sleep(50 * time.Millisecond)
			if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"one two"}]},"is_final":false}`) {
				return
			}
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 800
		cfg.MaxHoldMs = 300
	})
	events := make(chan STTEvent, 64)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	start := time.Now()
	if ev := recvSTTFinal(t, events); ev.CommittedBy != CommitMaxHold {
		t.Fatalf("CommittedBy = %q, want %q", ev.CommittedBy, CommitMaxHold)
	}
	// ~300ms ceiling against 50ms re-arms: if the cap tracked the last re-arm
	// it would never fire and this would run the full 2s stream.
	if elapsed := time.Since(start); elapsed > 1500*time.Millisecond {
		t.Errorf("commit took %v: the ceiling is tracking re-arms, not the utterance", elapsed)
	}
}

// The ceiling must not pre-empt the provider: a speaker who finishes normally
// inside the bound still commits by speech_final.
func TestDeepgramSTT_MaxHoldDoesNotPreemptSpeechFinal(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		send := func(s string) bool { return ws.WriteMessage(websocket.TextMessage, []byte(s)) == nil }
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"all"}]},"is_final":true,"speech_final":false}`) {
			return
		}
		time.Sleep(80 * time.Millisecond)
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"all done"}]},"is_final":true,"speech_final":true}`) {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 1000
		cfg.MaxHoldMs = 5000
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	ev := recvSTTFinal(t, events)
	if ev.CommittedBy != CommitSpeechFinal {
		t.Errorf("CommittedBy = %q, want %q", ev.CommittedBy, CommitSpeechFinal)
	}
	if ev.Text != "all all done" {
		t.Errorf("committed %q", ev.Text)
	}
}

// A negative MaxHoldMs disables the ceiling, for a deployment that would
// rather never cut a speaker off than bound the wait.
func TestDeepgramSTT_MaxHoldDisabled(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		send := func(s string) bool { return ws.WriteMessage(websocket.TextMessage, []byte(s)) == nil }
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"hello"}]},"is_final":true,"speech_final":false}`) {
			return
		}
		deepgramIgnoreAudio(ws) // silence: only the debounce can commit this
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 150
		cfg.MaxHoldMs = -1
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	if ev := recvSTTFinal(t, events); ev.CommittedBy != CommitFlushTimeout {
		t.Errorf("CommittedBy = %q, want %q with the ceiling disabled", ev.CommittedBy, CommitFlushTimeout)
	}
}

// The ceiling must hold even when partials arrive FASTER than a scheduled
// timer callback can win flushMu: each re-arm invalidates the pending
// callback's generation, so a hot stream could starve a zero-delay timer
// forever. Once the ceiling has expired, the re-arm path commits
// synchronously instead — this pins that a hot stream cannot outrun it.
func TestDeepgramSTT_MaxHoldSurvivesAHotPartialStream(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		send := func(s string) bool { return ws.WriteMessage(websocket.TextMessage, []byte(s)) == nil }
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"hot"}]},"is_final":true,"speech_final":false}`) {
			return
		}
		// ~2ms cadence for ~2s: far faster than timer-callback scheduling,
		// far longer than the ceiling.
		for i := 0; i < 1000; i++ {
			time.Sleep(2 * time.Millisecond)
			if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"hot mic"}]},"is_final":false}`) {
				return
			}
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 1000
		cfg.MaxHoldMs = 150
	})
	events := make(chan STTEvent, 64)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	start := time.Now()
	ev := recvSTTFinal(t, events)
	if ev.CommittedBy != CommitMaxHold {
		t.Errorf("CommittedBy = %q, want %q", ev.CommittedBy, CommitMaxHold)
	}
	// The stream runs ~2s; a starved ceiling could only commit after it
	// ends. Landing well inside it proves the synchronous path bounded the
	// turn.
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("commit took %v: the hot stream starved the ceiling", elapsed)
	}
}
