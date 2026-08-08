// sbbench — sentence-buffer latency characterization for cadence.
//
// Simulates an LLM emitting word-level text deltas at a controlled rate into
// cadence's RunSentenceBuffer and measures, per (responseStyle, tokenRate,
// config): time-to-first-chunk (what gates TTS start), chunk count, and chunk
// word-size distribution (prosody proxy). N repetitions; JSON out.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	cadence "github.com/amenophis1er/cadence"
	"github.com/amenophis1er/cadence/engines"
)

// Three response styles a voice agent actually produces.
var responses = map[string]string{
	"short": "Sure, I can help with that. Your appointment is confirmed for Tuesday at three thirty.",
	"paragraph": "The main difference between the two plans comes down to how usage is billed. " +
		"On the starter plan, every call is metered per minute, which works well when volume is low or unpredictable. " +
		"The growth plan switches to concurrent-channel pricing, so heavy steady traffic costs less overall, " +
		"but you pay for capacity even during quiet hours. Most teams switch once they pass about two thousand minutes a month.",
	"listy": "There are three things to check: first, the trunk credentials, which expire every ninety days; " +
		"second, the outbound caller ID, which must match a verified number; " +
		"and third, the codec list, which should put G711 before anything exotic.",
}

type cfgVariant struct {
	Name string
	Cfg  cadence.SentenceBufferConfig
}

type runResult struct {
	Style     string  `json:"style"`
	TokRate   int     `json:"tok_per_s"`
	Config    string  `json:"config"`
	TTFCms    float64 `json:"ttfc_ms"` // time to first chunk
	Chunks    int     `json:"chunks"`
	MeanWords float64 `json:"mean_words"`
	MinWords  int     `json:"min_words"`
	TotalMs   float64 `json:"total_ms"`
}

func words(s string) int { return len(strings.Fields(s)) }

func runOnce(style, text string, tokRate int, v cfgVariant, rng *rand.Rand) runResult {
	in := make(chan engines.LLMEvent, 8)
	out := make(chan string, 32)

	interval := time.Second / time.Duration(tokRate)
	start := time.Now()

	go func() {
		defer close(in)
		for _, w := range strings.Fields(text) {
			// emit word + trailing space as one delta, with ±30% jitter
			j := 1 + (rng.Float64()-0.5)*0.6
			time.Sleep(time.Duration(float64(interval) * j))
			in <- engines.LLMEvent{Type: "text", TextDelta: w + " "}
		}
		in <- engines.LLMEvent{Type: "done"}
	}()

	var ttfc time.Duration
	var chunkWords []int
	done := make(chan struct{})
	go func() {
		defer close(done)
		first := true
		for chunk := range out {
			if first {
				ttfc = time.Since(start)
				first = false
			}
			chunkWords = append(chunkWords, words(chunk))
		}
	}()

	cadence.RunSentenceBuffer(context.Background(), v.Cfg, in, out)
	<-done
	total := time.Since(start)

	sum, min := 0, 1<<30
	for _, w := range chunkWords {
		sum += w
		if w < min {
			min = w
		}
	}
	mean := 0.0
	if len(chunkWords) > 0 {
		mean = float64(sum) / float64(len(chunkWords))
	} else {
		min = 0
	}
	return runResult{
		Style: style, TokRate: tokRate, Config: v.Name,
		TTFCms: float64(ttfc.Microseconds()) / 1000,
		Chunks: len(chunkWords), MeanWords: mean, MinWords: min,
		TotalMs: float64(total.Microseconds()) / 1000,
	}
}

func main() {
	reps := 10
	rates := []int{20, 40, 80} // words/s ~ realistic LLM streaming spread
	variants := []cfgVariant{
		{"default_15w_300ms", cadence.DefaultSentenceBufferConfig()},
		{"eager_5w_100ms", cadence.SentenceBufferConfig{ClauseMinWords: 5, MaxWords: 40, FlushTimeout: 100 * time.Millisecond}},
		{"clause_10w_300ms", cadence.SentenceBufferConfig{ClauseMinWords: 10, MaxWords: 40, FlushTimeout: 300 * time.Millisecond}},
		{"patient_25w_500ms", cadence.SentenceBufferConfig{ClauseMinWords: 25, MaxWords: 60, FlushTimeout: 500 * time.Millisecond}},
	}

	rng := rand.New(rand.NewSource(42))
	var all []runResult
	for style, text := range responses {
		for _, rate := range rates {
			for _, v := range variants {
				for r := 0; r < reps; r++ {
					all = append(all, runOnce(style, text, rate, v, rng))
				}
			}
		}
	}

	// aggregate: median TTFC per (style, rate, config)
	type key struct {
		style, cfg string
		rate       int
	}
	groups := map[key][]runResult{}
	for _, r := range all {
		k := key{r.Style, r.Config, r.TokRate}
		groups[k] = append(groups[k], r)
	}
	type agg struct {
		Style     string  `json:"style"`
		TokRate   int     `json:"tok_per_s"`
		Config    string  `json:"config"`
		MedTTFCms float64 `json:"med_ttfc_ms"`
		MinTTFCms float64 `json:"min_ttfc_ms"`
		MaxTTFCms float64 `json:"max_ttfc_ms"`
		MedChunks float64 `json:"med_chunks"`
		MeanWords float64 `json:"mean_chunk_words"`
	}
	var aggs []agg
	for k, rs := range groups {
		sort.Slice(rs, func(i, j int) bool { return rs[i].TTFCms < rs[j].TTFCms })
		mid := rs[len(rs)/2]
		mw := 0.0
		for _, r := range rs {
			mw += r.MeanWords
		}
		aggs = append(aggs, agg{
			Style: k.style, TokRate: k.rate, Config: k.cfg,
			MedTTFCms: mid.TTFCms, MinTTFCms: rs[0].TTFCms, MaxTTFCms: rs[len(rs)-1].TTFCms,
			MedChunks: float64(mid.Chunks), MeanWords: mw / float64(len(rs)),
		})
	}
	sort.Slice(aggs, func(i, j int) bool {
		if aggs[i].Style != aggs[j].Style {
			return aggs[i].Style < aggs[j].Style
		}
		if aggs[i].TokRate != aggs[j].TokRate {
			return aggs[i].TokRate < aggs[j].TokRate
		}
		return aggs[i].Config < aggs[j].Config
	})

	out, _ := json.MarshalIndent(aggs, "", "  ")
	os.WriteFile("sb_results.json", out, 0o644)
	fmt.Println(string(out))
	raw, _ := json.Marshal(all)
	os.WriteFile("sb_raw.json", raw, 0o644)
	fmt.Fprintf(os.Stderr, "%d runs total\n", len(all))
}
