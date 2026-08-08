package cadence

import (
	"testing"

	"github.com/amenophis1er/cadence/audio"
)

// loud returns a 20 ms mu-law frame well above the speech threshold; quiet
// returns near-silence.
func loud() []byte {
	pcm := make([]int16, 160)
	for i := range pcm {
		if i%2 == 0 {
			pcm[i] = 6000
		} else {
			pcm[i] = -6000
		}
	}
	return audio.PCM16ToMuLaw(pcm)
}

func quiet() []byte {
	return audio.PCM16ToMuLaw(make([]int16, 160))
}

func TestVADOnsetNeedsConsecutiveSpeechFrames(t *testing.T) {
	v := NewEnergyVAD()
	if sp, _ := v.Observe(loud()); sp {
		t.Fatal("one loud frame must not trigger onset (key-click guard)")
	}
	v.Observe(loud())
	sp, _ := v.Observe(loud()) // third consecutive frame = 60ms onset
	if !sp {
		t.Fatal("three consecutive loud frames should declare speaking")
	}
}

func TestVADEndsExactlyOnceAfterSilence(t *testing.T) {
	v := NewEnergyVAD()
	for i := 0; i < 5; i++ {
		v.Observe(loud())
	}
	ended := 0
	for i := 0; i < 40; i++ {
		if _, just := v.Observe(quiet()); just {
			ended++
		}
	}
	if ended != 1 {
		t.Fatalf("justEnded must fire exactly once per utterance, fired %d times", ended)
	}
}

func TestVADResetClearsState(t *testing.T) {
	v := NewEnergyVAD()
	for i := 0; i < 5; i++ {
		v.Observe(loud())
	}
	v.Reset()
	// After reset the detector is idle: silence must never emit justEnded…
	for i := 0; i < 40; i++ {
		if _, just := v.Observe(quiet()); just {
			t.Fatal("justEnded after Reset without new speech")
		}
	}
	// …and onset debouncing starts over from scratch.
	if sp, _ := v.Observe(loud()); sp {
		t.Fatal("one loud frame after Reset must not re-declare speaking")
	}
}

func TestVADStatsWindowRolls(t *testing.T) {
	// 60 frames crosses the 50-frame diagnostic window; must not disturb
	// detection state.
	v := NewEnergyVAD()
	for i := 0; i < 60; i++ {
		if sp, just := v.Observe(quiet()); sp || just {
			t.Fatal("silence must stay silent across the stats window roll")
		}
	}
}

func TestMuLawRMSEmptyFrame(t *testing.T) {
	if rms := MuLawRMS(nil); rms != 0 {
		t.Fatalf("empty frame RMS = %v, want 0", rms)
	}
}

func TestVADIgnoresShortBurst(t *testing.T) {
	v := NewEnergyVAD()
	v.Observe(loud()) // single burst
	for i := 0; i < 40; i++ {
		if _, just := v.Observe(quiet()); just {
			t.Fatal("a burst below onset must never produce justEnded")
		}
	}
}
