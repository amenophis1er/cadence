package engines

import (
	"context"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// A commit driven by the provider's own end-of-utterance signal must be
// attributed to speech_final — the fast path a consumer should see most of
// the time.
func TestDeepgramSTT_CommitPolicySpeechFinal(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msg := `{"type":"Results","channel":{"alternatives":[{"transcript":"yes go ahead"}]},"is_final":true,"speech_final":true}`
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
	ev := recvSTTFinal(t, events)
	if ev.CommittedBy != CommitSpeechFinal {
		t.Errorf("CommittedBy = %q, want %q", ev.CommittedBy, CommitSpeechFinal)
	}
	// The provider had already decided; this engine should not have sat on it.
	if ev.HeldMs > 200 {
		t.Errorf("HeldMs = %d, want a near-immediate commit", ev.HeldMs)
	}
}

// When the provider never signals end-of-utterance, the commit comes from the
// fallback debounce — and must say so, with HeldMs reflecting the wait the
// caller actually experienced. Without this attribution the slow path is
// indistinguishable from the fast one.
func TestDeepgramSTT_CommitPolicyFlushTimeout(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msg := `{"type":"Results","channel":{"alternatives":[{"transcript":"okay"}]},"is_final":true,"speech_final":false}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(msg)); err != nil {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	const flushMs = 120
	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.FlushTimeoutMs = flushMs
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
	// The engine held this utterance for the debounce window; that latency is
	// this engine's, and the number must show it.
	if ev.HeldMs < flushMs-40 {
		t.Errorf("HeldMs = %d, want >= ~%d (the debounce it waited)", ev.HeldMs, flushMs)
	}
}

// The OnSTTCommit hook fires for each committed utterance, carrying the same
// attribution — so a consumer can graph endpointing without inspecting events.
func TestOnSTTCommitHookFires(t *testing.T) {
	type commit struct {
		engine string
		policy CommitPolicy
		heldMs int64
	}
	commits := make(chan commit, 8)

	prev := OnSTTCommit
	OnSTTCommit = func(engine string, policy CommitPolicy, heldMs int64) {
		commits <- commit{engine, policy, heldMs}
	}
	defer func() { OnSTTCommit = prev }()

	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msg := `{"type":"Results","channel":{"alternatives":[{"transcript":"hello there"}]},"is_final":true,"speech_final":true}`
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
	expectSTTEvent(t, events, STTTranscriptFinal, "hello there")

	// The event is emitted before the hook runs, so receiving it does not
	// guarantee the hook has fired yet — wait on the hook's own channel.
	var got commit
	select {
	case got = <-commits:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for OnSTTCommit")
	}
	if got.policy != CommitSpeechFinal {
		t.Errorf("hook policy = %q, want %q", got.policy, CommitSpeechFinal)
	}
	if got.engine != "deepgram-stt" {
		t.Errorf("hook engine = %q, want deepgram-stt", got.engine)
	}
	select {
	case extra := <-commits:
		t.Errorf("hook fired again unexpectedly: %+v", extra)
	default:
	}
}

// Non-final events carry no attribution: CommittedBy is meaningful only on a
// committed utterance.
func TestDeepgramSTT_DeltaHasNoCommitPolicy(t *testing.T) {
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msg := `{"type":"Results","channel":{"alternatives":[{"transcript":"partial"}]},"is_final":false}`
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
	select {
	case ev := <-events:
		if ev.Type != STTTranscriptDelta {
			t.Fatalf("got %v, want a delta", ev.Type)
		}
		if ev.CommittedBy != "" || ev.HeldMs != 0 {
			t.Errorf("delta carries commit attribution: %+v", ev)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no delta event")
	}
}

// recvSTTFinal waits for the next TranscriptFinal and returns it whole (the
// shared helper asserts only type+text).
func recvSTTFinal(t *testing.T, events <-chan STTEvent) STTEvent {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				t.Fatal("events channel closed before a final arrived")
			}
			if ev.Type == STTTranscriptFinal {
				return ev
			}
		case <-deadline:
			t.Fatal("timed out waiting for TranscriptFinal")
		}
	}
}
