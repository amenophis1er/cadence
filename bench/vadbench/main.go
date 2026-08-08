// vadbench — endpointing characterization for cadence's EnergyVAD.
//
// Synthesizes mu-law 8 kHz "speech" as syllabic energy bursts (60–180 ms
// voiced, RMS 1500–5000) separated by articulation micro-gaps (20–80 ms,
// RMS 100–400), matching the RMS bands documented in vad.go. Each trial:
//   utterance A (1.5–3 s) — hesitation pause P — utterance B (1–2 s) — end.
// Sweeps the offset (silence-to-commit) threshold and measures:
//   cut rate: how often the VAD commits during the hesitation (false turn end)
//   commit latency: time from true speech end to justEnded (turn-commit delay)
// The sweep uses a faithful parameterized copy of the EnergyVAD state machine
// (fields are private upstream); the shipped NewEnergyVAD() is run on the
// same trials as a cross-check at its 300 ms default.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"

	cadence "github.com/amenophis1er/cadence"
	"github.com/amenophis1er/cadence/audio"
)

const frameMs = 20
const frameSamples = 160 // 8 kHz * 20 ms

// ---- signal synthesis ----------------------------------------------------

// frameWithRMS produces one mu-law frame whose PCM RMS ≈ target.
func frameWithRMS(rng *rand.Rand, target float64) []byte {
	mu := make([]byte, frameSamples)
	for i := 0; i < frameSamples; i++ {
		// voiced-ish: 200 Hz fundamental + noise
		t := float64(i) / 8000
		s := 0.8*math.Sin(2*math.Pi*200*t+rng.Float64()) + 0.6*(rng.Float64()*2-1)
		mu[i] = audio.MuLawEncode(int16(s * target * 1.2)) // 1.2 ≈ RMS→peak fudge
	}
	return mu
}

// utterance emits syllabic speech totalling ~durMs.
func utterance(rng *rand.Rand, durMs int) [][]byte {
	var frames [][]byte
	elapsed := 0
	for elapsed < durMs {
		syl := 60 + rng.Intn(120) // 60–180 ms voiced
		for j := 0; j < syl/frameMs; j++ {
			frames = append(frames, frameWithRMS(rng, 1500+rng.Float64()*3500))
		}
		elapsed += syl
		gap := 20 + rng.Intn(60) // 20–80 ms articulation gap
		for j := 0; j < gap/frameMs; j++ {
			frames = append(frames, frameWithRMS(rng, 100+rng.Float64()*300))
		}
		elapsed += gap
	}
	return frames
}

func silence(rng *rand.Rand, durMs int) [][]byte {
	var frames [][]byte
	for j := 0; j < durMs/frameMs; j++ {
		frames = append(frames, frameWithRMS(rng, 100+rng.Float64()*200))
	}
	return frames
}

// ---- parameterized copy of EnergyVAD's state machine ---------------------

type vad struct {
	speechRMS                 float64
	onsetFrames, offsetFrames int
	speakingF, silenceF       int
	speaking                  bool
}

func (v *vad) observe(mu []byte) (bool, bool) {
	rms := cadence.MuLawRMS(mu)
	if rms >= v.speechRMS {
		v.silenceF = 0
		v.speakingF++
		if !v.speaking && v.speakingF >= v.onsetFrames {
			v.speaking = true
		}
	} else {
		v.speakingF = 0
		if v.speaking {
			v.silenceF++
			if v.silenceF >= v.offsetFrames {
				v.speaking = false
				v.silenceF = 0
				return false, true
			}
		}
	}
	return v.speaking, false
}

// ---- trial ---------------------------------------------------------------

type trial struct {
	frames        [][]byte
	hesitStart    int // frame index where hesitation begins
	hesitEnd      int
	trueEndFrame  int // last speech frame of utterance B
}

func makeTrial(rng *rand.Rand, pauseMs int) trial {
	a := utterance(rng, 1500+rng.Intn(1500))
	p := silence(rng, pauseMs)
	b := utterance(rng, 1000+rng.Intn(1000))
	tail := silence(rng, 1500)

	var t trial
	t.frames = append(t.frames, a...)
	t.hesitStart = len(t.frames)
	t.frames = append(t.frames, p...)
	t.hesitEnd = len(t.frames)
	t.frames = append(t.frames, b...)
	t.trueEndFrame = len(t.frames)
	t.frames = append(t.frames, tail...)
	return t
}

type observer interface{ observe([]byte) (bool, bool) }

type shippedVAD struct{ v *cadence.EnergyVAD }

func (s shippedVAD) observe(mu []byte) (bool, bool) { return s.v.Observe(mu) }

// runTrial returns (cutDuringHesitation, commitLatencyMs [-1 if never]).
func runTrial(t trial, o observer) (bool, float64) {
	cut := false
	commit := -1.0
	for i, f := range t.frames {
		_, ended := o.observe(f)
		if !ended {
			continue
		}
		if i >= t.hesitStart && i < t.hesitEnd {
			cut = true
		}
		if i >= t.trueEndFrame && commit < 0 {
			commit = float64(i-t.trueEndFrame) * frameMs
		}
	}
	return cut, commit
}

func main() {
	reps := 50
	pauses := []int{200, 300, 400, 500, 600, 800}
	offsets := []int{200, 300, 400, 500, 600} // ms

	type row struct {
		OffsetMs   int     `json:"offset_ms"`
		PauseMs    int     `json:"pause_ms"`
		CutRate    float64 `json:"cut_rate"`
		MedCommit  float64 `json:"med_commit_ms"`
		P95Commit  float64 `json:"p95_commit_ms"`
		Shipped    bool    `json:"shipped_impl"`
	}
	var rows []row

	run := func(offsetMs, pauseMs int, shipped bool) row {
		rng := rand.New(rand.NewSource(int64(offsetMs*10000 + pauseMs))) // same trials per offset? no: same per (offset,pause)
		cuts := 0
		var commits []float64
		for r := 0; r < reps; r++ {
			t := makeTrial(rng, pauseMs)
			var o observer
			if shipped {
				o = shippedVAD{cadence.NewEnergyVAD()}
			} else {
				o = &vad{speechRMS: 800, onsetFrames: 3, offsetFrames: offsetMs / frameMs}
			}
			cut, commit := runTrial(t, o)
			if cut {
				cuts++
			}
			if commit >= 0 {
				commits = append(commits, commit)
			}
		}
		sort.Float64s(commits)
		med, p95 := -1.0, -1.0
		if len(commits) > 0 {
			med = commits[len(commits)/2]
			p95 = commits[int(0.95*float64(len(commits)-1))]
		}
		return row{OffsetMs: offsetMs, PauseMs: pauseMs,
			CutRate: float64(cuts) / float64(reps), MedCommit: med, P95Commit: p95, Shipped: shipped}
	}

	for _, off := range offsets {
		for _, p := range pauses {
			rows = append(rows, run(off, p, false))
		}
	}
	// cross-check: shipped implementation at its fixed 300 ms
	for _, p := range pauses {
		rows = append(rows, run(300, p, true))
	}

	out, _ := json.MarshalIndent(rows, "", "  ")
	os.WriteFile("vad_results.json", out, 0o644)
	fmt.Println(string(out))
}
