package engines

// Observability hooks. Cadence records nothing itself — the consuming
// application installs these to feed its own metrics system (voiceapp wires
// them to Prometheus; others may log or drop them). Defaults are no-ops so
// the engines never nil-check.
var (
	// OnTTSStreamEnd fires when a streaming TTS session finishes.
	// status is "ok" or "error"; seconds is the stream's wall duration.
	OnTTSStreamEnd = func(engine, status string, seconds float64) {}
)
