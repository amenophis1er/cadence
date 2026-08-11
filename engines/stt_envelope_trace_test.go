package engines

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
)

// The trace must describe the SHAPE of each envelope and never its content:
// transcripts are caller speech, and a diagnostic is not a licence to log it.
func TestDeepgramSTT_EnvelopeTraceLogsShapeNotContent(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})))
	defer slog.SetDefault(prev)

	const secret = "my social is four two four two"
	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msgs := []string{
			`{"type":"Results","channel":{"alternatives":[{"transcript":"` + secret + `"}]},"is_final":false}`,
			`{"type":"Results","channel":{"alternatives":[{"transcript":"` + secret + `"}]},"is_final":true,"speech_final":true}`,
		}
		for _, m := range msgs {
			if err := ws.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				return
			}
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, func(cfg *DeepgramSTTConfig) {
		cfg.TraceEnvelopes = true
	})
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	expectSTTEvent(t, events, STTSpeechStarted, "")
	if ev := recvSTTFinal(t, events); ev.Text != secret {
		t.Fatalf("committed %q", ev.Text)
	}
	eng.Stop()

	out := buf.String()
	if strings.Contains(out, secret) || strings.Contains(out, "four two") {
		t.Fatalf("the trace leaked transcript text:\n%s", out)
	}
	for _, want := range []string{"deepgram-stt: envelope", "is_final=", "speech_final=", "text_len=", "since_prev_ms="} {
		if !strings.Contains(out, want) {
			t.Errorf("trace missing %q:\n%s", want, out)
		}
	}
}

// Off by default: this is per-envelope volume, not steady-state logging.
func TestDeepgramSTT_EnvelopeTraceOffByDefault(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(prev)

	wsURL, _, cleanup := mockSTTWSServer(t, func(ws *websocket.Conn) {
		msg := `{"type":"Results","channel":{"alternatives":[{"transcript":"hello"}]},"is_final":true,"speech_final":true}`
		if err := ws.WriteMessage(websocket.TextMessage, []byte(msg)); err != nil {
			return
		}
		deepgramIgnoreAudio(ws)
	})
	defer cleanup()

	eng := newTestDeepgramEngine(wsURL, nil) // no TraceEnvelopes
	events := make(chan STTEvent, 32)
	if err := eng.Start(context.Background(), events); err != nil {
		t.Fatalf("Start: %v", err)
	}
	expectSTTEvent(t, events, STTSpeechStarted, "")
	expectSTTEvent(t, events, STTTranscriptFinal, "hello")
	eng.Stop()

	if strings.Contains(buf.String(), "deepgram-stt: envelope") {
		t.Fatal("envelope trace emitted without being enabled")
	}
}
