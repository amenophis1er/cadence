package cadence

import (
	"log/slog"
	"math"

	"github.com/amenophis1er/cadence/audio"
)

// energyVAD is a stateful, RMS-based voice activity detector. It consumes
// 20 ms frames of mu-law 8 kHz audio (160 bytes each) and reports whether
// the user has just stopped speaking.
//
// The detector is intentionally simple — speech-vs-silence by short-time RMS
// against a fixed threshold, with frame-count debouncing for both onset and
// offset. Replace with Silero (CGo onnxruntime) when noise robustness matters.
type energyVAD struct {
	speechRMS    float64 // RMS above which a frame is "speech"
	onsetFrames  int     // consecutive speech frames to declare "speaking"
	offsetFrames int     // consecutive silence frames to declare "stopped"

	speakingFrames int  // consecutive speech frames seen while idle
	silenceFrames  int  // consecutive silence frames seen while speaking
	speaking       bool // current state

	// Aggregate stats for diagnostic logging — without these we have no
	// visibility into actual RMS levels during a call and can't tune the
	// threshold sensibly.
	statFrames int
	statMinRMS float64
	statMaxRMS float64
	statSumRMS float64
}

func newEnergyVAD() *energyVAD {
	return &energyVAD{
		// Tuned for telephony mu-law: typical room background ~100-300 RMS,
		// telephone speech 1500-5000 RMS. 800 leaves comfortable margin.
		speechRMS: 800,
		// 60 ms onset (3 × 20 ms) to ignore key clicks and short bursts.
		onsetFrames: 3,
		// 300 ms silence (15 × 20 ms) to commit a turn. Tighter than the
		// classic 600 ms — trades a small risk of mid-sentence cuts for
		// noticeably snappier perceived response (this is the cheap half
		// of the latency story; the proper fix is server-side VAD via
		// /v1/realtime).
		offsetFrames: 15,
		statMinRMS:   math.MaxFloat64,
	}
}

// observe ingests one mu-law frame and returns (speakingNow, justEnded).
// "justEnded" fires exactly once per utterance, on the frame that crosses
// the silence threshold after speech.
func (v *energyVAD) observe(mu []byte) (speakingNow bool, justEnded bool) {
	rms := mulawRMS(mu)

	// Aggregate RMS for diagnostic logging once per second (50 × 20 ms).
	v.statFrames++
	v.statSumRMS += rms
	if rms < v.statMinRMS {
		v.statMinRMS = rms
	}
	if rms > v.statMaxRMS {
		v.statMaxRMS = rms
	}
	if v.statFrames >= 50 {
		slog.Debug("cadence: VAD stats (1s window)",
			"min_rms", int(v.statMinRMS),
			"max_rms", int(v.statMaxRMS),
			"avg_rms", int(v.statSumRMS/float64(v.statFrames)),
			"threshold", int(v.speechRMS),
			"speaking", v.speaking,
		)
		v.statFrames = 0
		v.statMinRMS = math.MaxFloat64
		v.statMaxRMS = 0
		v.statSumRMS = 0
	}

	wasSpeaking := v.speaking
	if rms >= v.speechRMS {
		v.silenceFrames = 0
		v.speakingFrames++
		if !v.speaking && v.speakingFrames >= v.onsetFrames {
			v.speaking = true
		}
	} else {
		v.speakingFrames = 0
		if v.speaking {
			v.silenceFrames++
			if v.silenceFrames >= v.offsetFrames {
				v.speaking = false
				v.silenceFrames = 0
				slog.Debug("cadence: VAD speech ended", "rms", int(rms))
				return false, true
			}
		}
	}
	if v.speaking && !wasSpeaking {
		slog.Debug("cadence: VAD speech started", "rms", int(rms))
	}
	return v.speaking, false
}

// reset clears all state, e.g. after committing a turn.
func (v *energyVAD) reset() {
	v.speakingFrames = 0
	v.silenceFrames = 0
	v.speaking = false
}

func mulawRMS(mu []byte) float64 {
	if len(mu) == 0 {
		return 0
	}
	var sumSq float64
	for _, b := range mu {
		s := float64(audio.MuLawDecode(b))
		sumSq += s * s
	}
	return math.Sqrt(sumSq / float64(len(mu)))
}
