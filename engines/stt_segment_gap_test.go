package engines

import (
	"context"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// A speaker who pauses mid-utterance and resumes produces two is_final
// segments the engine merges into one turn. The gap between them is reported,
// because a flush timeout below it would have split their sentence.
func TestDeepgramSTT_ReportsWithinUtteranceGap(t *testing.T) {
	const pause = 220 * time.Millisecond

	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		first := `{"type":"Results","channel":{"alternatives":[{"transcript":"my number is"}]},"is_final":true,"speech_final":false}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(first)); err != nil {
			return
		}
		time.Sleep(pause) // the speaker hesitates, then continues
		second := `{"type":"Results","channel":{"alternatives":[{"transcript":"eight eight eight"}]},"is_final":true,"speech_final":true}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(second)); err != nil {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	// Flush window comfortably longer than the pause, so the merge happens.
	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 2000
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	ev := recvSTTFinal(t, events)
	if ev.Text != "my number is eight eight eight" {
		t.Fatalf("segments should merge into one turn, got %q", ev.Text)
	}
	// The reported gap must reflect the real pause — this is the number that
	// decides the smallest safe flush timeout.
	if ev.MaxSegmentGapMs < pause.Milliseconds()-60 {
		t.Errorf("MaxSegmentGapMs = %d, want ≈%d", ev.MaxSegmentGapMs, pause.Milliseconds())
	}
	if ev.MaxSegmentGapMs > pause.Milliseconds()+400 {
		t.Errorf("MaxSegmentGapMs = %d, implausibly larger than the pause", ev.MaxSegmentGapMs)
	}
}

// An utterance delivered in one segment has no internal pause to report.
func TestDeepgramSTT_SingleSegmentHasNoGap(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msg := `{"type":"Results","channel":{"alternatives":[{"transcript":"yes"}]},"is_final":true,"speech_final":true}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(msg)); err != nil {
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
	if ev := recvSTTFinal(t, events); ev.MaxSegmentGapMs != 0 {
		t.Errorf("MaxSegmentGapMs = %d, want 0 for a single-segment utterance", ev.MaxSegmentGapMs)
	}
}

// The gap is per-utterance: a long pause in one turn must not be attributed to
// the next, or the measured distribution would drift upward over a call.
func TestDeepgramSTT_GapResetsBetweenUtterances(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		send := func(s string) bool {
			return ws.WriteMessage(websocket.TextMessage, []byte(s)) == nil
		}
		// Turn 1: two segments separated by a pause.
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"one"}]},"is_final":true,"speech_final":false}`) {
			return
		}
		time.Sleep(200 * time.Millisecond)
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"two"}]},"is_final":true,"speech_final":true}`) {
			return
		}
		// Turn 2: a single clean segment.
		time.Sleep(50 * time.Millisecond)
		if !send(`{"type":"Results","channel":{"alternatives":[{"transcript":"three"}]},"is_final":true,"speech_final":true}`) {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = 2000
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer eng.Stop()

	expectSTTEvent(t, events, STTSpeechStarted, "")
	if first := recvSTTFinal(t, events); first.MaxSegmentGapMs < 140 {
		t.Fatalf("first turn gap = %d, want ≈200", first.MaxSegmentGapMs)
	}
	second := recvSTTFinal(t, events)
	if second.MaxSegmentGapMs != 0 {
		t.Errorf("second turn inherited a gap of %d from the first", second.MaxSegmentGapMs)
	}
}
