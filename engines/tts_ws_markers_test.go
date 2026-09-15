package engines

// Marker-chunk tests: the vendor's own sentence and clear boundaries surface
// on audioCh as AudioChunks with Flushed/Cleared set and no Data, so a
// sequencing consumer (a telephony session pacing one sentence at a time)
// reads the boundary the vendor reported instead of inferring it from
// silence. See AudioChunk in types.go.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// The Deepgram engine forwards a server "Flushed" control as a marker chunk,
// after the audio that preceded it.
func TestDeepgramTTSWS_FlushedMarkerSurfaces(t *testing.T) {
	wsURL, cleanup := mockDeepgramTTSServer(t, "req-flush", func(t *testing.T, ws *websocket.Conn) {
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env map[string]any
			if json.Unmarshal(data, &env) != nil {
				continue
			}
			switch env["type"] {
			case "Speak":
				_ = ws.WriteMessage(websocket.BinaryMessage, []byte{1, 2, 3, 4})
			case "Flush":
				_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Flushed"}`))
			case "Close":
				_ = ws.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
				return
			}
		}
	})
	defer cleanup()

	eng := newTestEngine(wsURL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 8)
	textCh <- TextChunk{Text: "hello.", IsSentenceEnd: true}
	close(textCh)

	done := make(chan error, 1)
	go func() { done <- eng.Stream(ctx, textCh, audioCh) }()

	var got []AudioChunk
	for c := range audioCh {
		got = append(got, c)
	}
	if err := <-done; err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var sawAudioFirst, sawFlush bool
	for _, c := range got {
		if c.Flushed {
			if len(c.Data) != 0 || c.Cleared {
				t.Fatalf("Flushed marker must be bare, got %+v", c)
			}
			sawFlush = true
		} else if len(c.Data) > 0 && !sawFlush {
			sawAudioFirst = true
		}
	}
	if !sawAudioFirst || !sawFlush {
		t.Fatalf("want audio then a Flushed marker, got %+v", got)
	}
}

// A Flushed that acknowledges an ABANDONED turn's flush (a Clear is pending)
// is suppressed — forwarding it would tell the consumer a live sentence
// completed — while the Cleared ack itself surfaces as a marker, ungated,
// because it is exactly what a draining consumer waits on.
func TestDeepgramTTSWS_ClearSuppressesStaleFlushedAndSurfacesClearedMarker(t *testing.T) {
	sentToWire := make(chan struct{})
	wsURL, cleanup := mockDeepgramTTSServer(t, "req-clear", func(t *testing.T, ws *websocket.Conn) {
		flushSeen := false
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env map[string]any
			if json.Unmarshal(data, &env) != nil {
				continue
			}
			switch env["type"] {
			case "Flush":
				if !flushSeen {
					flushSeen = true
					close(sentToWire)
				}
			case "Clear":
				// Deepgram acks a Clear by flushing what was in
				// flight for the abandoned turn, then Cleared.
				_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Flushed"}`))
				_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"type":"Cleared"}`))
			case "Close":
				_ = ws.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
				return
			}
		}
	})
	defer cleanup()

	eng := newTestEngine(wsURL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 8)
	textCh <- TextChunk{Text: "to be abandoned.", IsSentenceEnd: true}

	done := make(chan error, 1)
	go func() { done <- eng.Stream(ctx, textCh, audioCh) }()

	<-sentToWire // the sentence reached the vendor; now barge in
	if err := eng.Clear(); err != nil {
		t.Fatalf("Clear: %v", err)
	}

	deadline := time.After(3 * time.Second)
wait:
	for {
		select {
		case c := <-audioCh:
			if c.Flushed {
				t.Fatalf("stale Flushed marker escaped the clear gate: %+v", c)
			}
			if c.Cleared {
				if len(c.Data) != 0 {
					t.Fatalf("Cleared marker must carry no Data, got %+v", c)
				}
				break wait
			}
		case <-deadline:
			t.Fatal("no Cleared marker arrived")
		}
	}
	close(textCh)
	for range audioCh {
	} // drain to Stream's close
	if err := <-done; err != nil {
		t.Fatalf("Stream: %v", err)
	}
}

// The ElevenLabs engine surfaces a pre-EOS isFinal — its per-flush boundary —
// as a Flushed marker, after that segment's audio.
func TestElevenTTSWS_PerSegmentIsFinalSurfacesFlushedMarker(t *testing.T) {
	audioB64 := base64.StdEncoding.EncodeToString([]byte{9, 9, 9, 9})
	wsURL, cleanup := mockElevenTTSServer(t, func(t *testing.T, ws *websocket.Conn) {
		for {
			_, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var env map[string]any
			if json.Unmarshal(data, &env) != nil {
				continue
			}
			text, _ := env["text"].(string)
			switch {
			case text == " ": // BOS
			case text == "": // EOS — answer with session-final isFinal
				_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"isFinal":true}`))
				_ = ws.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"))
				return
			default: // the sentence: audio, then the per-segment isFinal
				_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"audio":"`+audioB64+`"}`))
				_ = ws.WriteMessage(websocket.TextMessage, []byte(`{"isFinal":true}`))
			}
		}
	})
	defer cleanup()

	eng := newTestElevenWSEngine(wsURL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// textCh stays OPEN while the segment plays — as in a real call —
	// so the per-segment isFinal cannot race the client's own EOS and be
	// misread as session-end.
	textCh := make(chan TextChunk, 1)
	audioCh := make(chan AudioChunk, 8)
	textCh <- TextChunk{Text: "hello.", IsSentenceEnd: true}

	done := make(chan error, 1)
	go func() { done <- eng.Stream(ctx, textCh, audioCh) }()

	var sawAudio, sawFlush bool
	deadline := time.After(3 * time.Second)
wait:
	for {
		select {
		case c := <-audioCh:
			if len(c.Data) > 0 {
				sawAudio = true
			}
			if c.Flushed {
				if len(c.Data) != 0 {
					t.Fatalf("Flushed marker must carry no Data, got %+v", c)
				}
				if !sawAudio {
					t.Fatal("Flushed marker arrived before the segment's audio")
				}
				sawFlush = true
				break wait
			}
		case <-deadline:
			t.Fatalf("no Flushed marker (sawAudio=%v)", sawAudio)
		}
	}
	close(textCh)
	for range audioCh {
	} // drain to Stream's close
	if err := <-done; err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if !sawFlush {
		t.Fatal("unreachable")
	}
}
