package audio

import "testing"

// mu-law encode/decode must round-trip within codec tolerance: G.711 is 8-bit
// log-companded, so exact equality is impossible, but error must stay small
// relative to amplitude.
func TestMuLawRoundTrip(t *testing.T) {
	for _, pcm := range []int16{0, 1, -1, 100, -100, 1000, -1000, 8000, -8000, 30000, -30000} {
		got := MuLawDecode(MuLawEncode(pcm))
		diff := int32(got) - int32(pcm)
		if diff < 0 {
			diff = -diff
		}
		// Companding error grows with amplitude; 3% + 8 LSB covers G.711.
		limit := int32(pcm)/33 + 8
		if limit < 0 {
			limit = -limit + 8
		}
		if diff > limit {
			t.Fatalf("round trip %d → %d (err %d > %d)", pcm, got, diff, limit)
		}
	}
}

func TestSliceConversionsPreserveLength(t *testing.T) {
	mu := []byte{0x00, 0x7f, 0x80, 0xff, 0x55, 0xd5}
	pcm := MuLawToPCM16(mu)
	if len(pcm) != len(mu) {
		t.Fatalf("MuLawToPCM16 length %d != %d", len(pcm), len(mu))
	}
	back := PCM16ToMuLaw(pcm)
	if len(back) != len(mu) {
		t.Fatalf("PCM16ToMuLaw length mismatch")
	}
	// G.711 has two encodings of zero (+0/-0), so bytes need not round-trip
	// exactly — the decoded VALUES must.
	for i := range mu {
		if MuLawDecode(back[i]) != MuLawDecode(mu[i]) {
			t.Fatalf("byte %d: decode(%02x)=%d != decode(%02x)=%d", i, back[i], MuLawDecode(back[i]), mu[i], MuLawDecode(mu[i]))
		}
	}
}

func TestResampleLengths(t *testing.T) {
	in := make([]int16, 160) // 20ms at 8k
	up := Upsample8kTo16k(in)
	if len(up) != 320 {
		t.Fatalf("upsample: %d", len(up))
	}
	down := Downsample16kTo8k(up)
	if len(down) != 160 {
		t.Fatalf("downsample: %d", len(down))
	}
}

func TestBytesRoundTrip(t *testing.T) {
	pcm := []int16{0, 1, -1, 32767, -32768, 12345}
	b := PCM16ToBytes(pcm)
	if len(b) != len(pcm)*2 {
		t.Fatalf("bytes length %d", len(b))
	}
	back := BytesToPCM16(b)
	for i := range pcm {
		if back[i] != pcm[i] {
			t.Fatalf("sample %d: %d != %d", i, back[i], pcm[i])
		}
	}
}
