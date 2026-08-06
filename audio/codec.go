package audio

// Basic G.711 mu-law codec and simple 8k<->16k resampling helpers.

// MuLawDecode decodes a single 8-bit mu-law byte into 16-bit PCM sample.
func MuLawDecode(mu byte) int16 {
	mu = ^mu
	t := ((int(mu) & 0x0F) << 3) + 0x84
	t <<= uint((uint(mu) & 0x70) >> 4)
	if (mu & 0x80) != 0 {
		return int16(0x84 - t)
	}
	return int16(t - 0x84)
}

// MuLawEncode encodes a 16-bit PCM sample into 8-bit mu-law.
func MuLawEncode(pcm int16) byte {
	const (
		mu     = 255
		maxVal = 32635
	)
	sign := byte(0)
	x := int(pcm)
	if x < 0 {
		x = -x
		sign = 0x80
	}
	if x > maxVal {
		x = maxVal
	}
	x += 0x84 // bias
	var exponent int
	var mask int = 0x4000
	for exponent = 7; exponent > 0; exponent-- {
		if x&mask != 0 {
			break
		}
		mask >>= 1
	}
	mantissa := (x >> (exponent + 3)) & 0x0F
	muLaw := ^(sign | byte(exponent<<4) | byte(mantissa))
	return muLaw
}

// MuLawToPCM16 converts a mu-law byte slice to PCM16 little-endian samples.
func MuLawToPCM16(mu []byte) []int16 {
	out := make([]int16, len(mu))
	for i, b := range mu {
		out[i] = MuLawDecode(b)
	}
	return out
}

// PCM16ToMuLaw converts PCM16 samples to mu-law bytes.
func PCM16ToMuLaw(samples []int16) []byte {
	out := make([]byte, len(samples))
	for i, s := range samples {
		out[i] = MuLawEncode(s)
	}
	return out
}

// Upsample8kTo16k duplicates samples to convert 8k to 16k.
func Upsample8kTo16k(in []int16) []int16 {
	out := make([]int16, len(in)*2)
	j := 0
	for _, s := range in {
		out[j] = s
		out[j+1] = s
		j += 2
	}
	return out
}

// Downsample16kTo8k drops every other sample to convert 16k to 8k.
func Downsample16kTo8k(in []int16) []int16 {
	out := make([]int16, (len(in)+1)/2)
	j := 0
	for i := 0; i < len(in); i += 2 {
		out[j] = in[i]
		j++
	}
	return out
}

// PCM16ToBytes packs int16 samples into little-endian byte slice.
func PCM16ToBytes(samples []int16) []byte {
	out := make([]byte, len(samples)*2)
	j := 0
	for _, s := range samples {
		out[j] = byte(s)
		out[j+1] = byte(s >> 8)
		j += 2
	}
	return out
}

// BytesToPCM16 unpacks little-endian bytes to int16 samples.
func BytesToPCM16(b []byte) []int16 {
	n := len(b) / 2
	out := make([]int16, n)
	for i := 0; i < n; i++ {
		out[i] = int16(b[2*i]) | int16(b[2*i+1])<<8
	}
	return out
}
