// e2ebench — live pipeline latency decomposition for cadence.
//
// Runs the real chain: LLM stream → RunSentenceBuffer → ElevenLabs WS TTS,
// timestamping each stage boundary:
//
//	t0                 request dispatched to LLM
//	t_first_delta      first LLM text delta arrives
//	t_first_flush      sentence buffer emits first chunk (TTS may start)
//	t_first_audio      first synthesized audio chunk arrives
//
// Time-to-first-audio = network+model TTFT + buffer holding time + TTS TTFB.
// N repetitions per prompt style; JSON out with per-stage medians.
//
// Env:
//	LLM_PROVIDER   openai | anthropic          (default openai)
//	OPENAI_URL     default https://api.openai.com/v1/chat/completions
//	OPENAI_API_KEY / OPENAI_MODEL (default gpt-4o-mini)
//	ANTHROPIC_API_KEY / ANTHROPIC_MODEL (default claude-haiku-4-5-20251001)
//	ELEVENLABS_API_KEY / ELEVEN_VOICE (default JBFqnCBsd6RMkjVDRZzb)
//	REPS           default 5
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"time"

	cadence "github.com/amenophis1er/cadence"
	"github.com/amenophis1er/cadence/engines"
)

var prompts = map[string]string{
	"short":     "Answer in one short sentence: what timezone is Chicago in?",
	"paragraph": "In about 80 words, explain the difference between per-minute and concurrent-channel pricing for business phone service.",
	"listy":     "Give three comma-separated troubleshooting steps for a SIP registration failure, as a single flowing sentence with clauses, no numbered list formatting.",
}

type stage struct {
	Style        string  `json:"style"`
	Rep          int     `json:"rep"`
	TTFDeltaMs   float64 `json:"llm_first_delta_ms"`
	TTFFlushMs   float64 `json:"first_flush_ms"`
	TTFAudioMs   float64 `json:"first_audio_ms"`
	LLMTotalMs   float64 `json:"llm_total_ms"`
	DeltaCount   int     `json:"delta_count"`
	FlushCount   int     `json:"flush_count"`
	Err          string  `json:"err,omitempty"`
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func newLLM() (engines.LLMEngine, string) {
	switch env("LLM_PROVIDER", "openai") {
	case "anthropic":
		m := env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
		return engines.NewAnthropic(engines.AnthropicConfig{
			URL: "https://api.anthropic.com", APIKey: os.Getenv("ANTHROPIC_API_KEY"), Model: m,
		}), "anthropic/" + m
	default:
		m := env("OPENAI_MODEL", "gpt-4o-mini")
		return engines.NewOpenAICompat(engines.ChatConfig{
			URL:    env("OPENAI_URL", "https://api.openai.com/v1/chat/completions"),
			APIKey: os.Getenv("OPENAI_API_KEY"), Model: m,
		}), "openai/" + m
	}
}

func runOnce(style, prompt string, rep int) stage {
	s := stage{Style: style, Rep: rep}
	llm, _ := newLLM()

	tts := engines.NewElevenTTSWS(engines.ElevenTTSConfig{
		APIKey: os.Getenv("ELEVENLABS_API_KEY"),
		Voice:  env("ELEVEN_VOICE", "JBFqnCBsd6RMkjVDRZzb"),
	})

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	llmEvents := make(chan engines.LLMEvent, 64)
	sbOut := make(chan string, 32)
	textCh := make(chan engines.TextChunk, 32)
	audioCh := make(chan engines.AudioChunk, 256)

	t0 := time.Now()
	ms := func(t time.Time) float64 { return float64(t.Sub(t0).Microseconds()) / 1000 }

	// LLM producer
	llmErr := make(chan error, 1)
	go func() {
		llmErr <- llm.Stream(ctx, []engines.LLMMessage{
			{Role: "system", Content: "You are a concise phone-support agent."},
			{Role: "user", Content: prompt},
		}, nil, llmEvents)
	}()

	// tap deltas for timing, forward to sentence buffer
	tapped := make(chan engines.LLMEvent, 64)
	go func() {
		defer close(tapped)
		for ev := range llmEvents {
			if ev.Type == "text" {
				if s.DeltaCount == 0 {
					s.TTFDeltaMs = ms(time.Now())
				}
				s.DeltaCount++
			}
			if ev.Type == "done" {
				s.LLMTotalMs = ms(time.Now())
			}
			tapped <- ev
		}
	}()

	// TTS consumer
	ttsErr := make(chan error, 1)
	go func() { ttsErr <- tts.Stream(ctx, textCh, audioCh) }()
	audioDone := make(chan struct{})
	go func() {
		defer close(audioDone)
		first := true
		for range audioCh {
			if first {
				s.TTFAudioMs = ms(time.Now())
				first = false
			}
		}
	}()

	// bridge sentence buffer → TTS TextChunks
	bridgeDone := make(chan struct{})
	go func() {
		defer close(bridgeDone)
		defer close(textCh)
		for chunk := range sbOut {
			if s.FlushCount == 0 {
				s.TTFFlushMs = ms(time.Now())
			}
			s.FlushCount++
			textCh <- engines.TextChunk{Text: chunk, IsSentenceEnd: true}
		}
	}()

	cadence.RunSentenceBuffer(ctx, cadence.DefaultSentenceBufferConfig(), tapped, sbOut)
	<-bridgeDone
	if err := <-llmErr; err != nil {
		s.Err = "llm: " + err.Error()
	}
	if err := <-ttsErr; err != nil && s.Err == "" {
		s.Err = "tts: " + err.Error()
	}
	<-audioDone
	return s
}

func med(xs []float64) float64 {
	if len(xs) == 0 {
		return -1
	}
	sort.Float64s(xs)
	return xs[len(xs)/2]
}

func main() {
	reps, _ := strconv.Atoi(env("REPS", "5"))
	if os.Getenv("ELEVENLABS_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "need ELEVENLABS_API_KEY (and LLM provider key)")
		os.Exit(1)
	}
	_, llmName := newLLM()

	var all []stage
	for style, prompt := range prompts {
		for r := 0; r < reps; r++ {
			st := runOnce(style, prompt, r)
			all = append(all, st)
			fmt.Fprintf(os.Stderr, "%s#%d: delta=%.0fms flush=%.0fms audio=%.0fms err=%s\n",
				style, r, st.TTFDeltaMs, st.TTFFlushMs, st.TTFAudioMs, st.Err)
			time.Sleep(500 * time.Millisecond)
		}
	}

	summary := map[string]any{"llm": llmName, "runs": all}
	for style := range prompts {
		var d, f, a []float64
		for _, st := range all {
			if st.Style != style || st.Err != "" {
				continue
			}
			d = append(d, st.TTFDeltaMs)
			f = append(f, st.TTFFlushMs)
			a = append(a, st.TTFAudioMs)
		}
		summary["med_"+style] = map[string]float64{
			"llm_first_delta_ms": med(d),
			"first_flush_ms":     med(f),
			"first_audio_ms":     med(a),
		}
	}
	out, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile("e2e_results.json", out, 0o644)
	fmt.Println(string(out))
}
