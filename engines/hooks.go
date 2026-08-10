package engines

// Observability hooks. Cadence records nothing itself — the consuming
// application installs these to feed its own metrics system (Prometheus,
// OpenTelemetry, plain logs — cadence does not care). Defaults are no-ops so
// the engines never nil-check.
var (
	// OnTTSStreamEnd fires when a streaming TTS session finishes.
	// status is "ok" or "error"; seconds is the stream's wall duration.
	OnTTSStreamEnd = func(engine, status string, seconds float64) {}

	// OnSTTCommit fires each time a streaming STT engine commits an
	// utterance, with the rule that committed it and how long the engine
	// held it locally (see STTEvent.CommittedBy / HeldMs for the precise
	// meaning, including what HeldMs excludes).
	//
	// Endpointing is the one pipeline stage an orchestrator can actually
	// tune, so it is the one worth graphing: a rising share of
	// CommitFlushTimeout means the provider has stopped signalling
	// end-of-utterance and every affected turn is paying the debounce.
	//
	// Implementations MUST NOT block: unlike OnTTSStreamEnd, this hook is
	// invoked on the engine's hot path while its internal commit lock is
	// held, so a slow hook stalls transcript delivery and shutdown. Record
	// the observation (counter increment, channel send to a buffered
	// collector) and return.
	OnSTTCommit = func(engine string, policy CommitPolicy, heldMs int64) {}
)
