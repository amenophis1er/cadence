# cadence

Provider-agnostic voice AI engines for Go: streaming STT, TTS and LLM behind
uniform interfaces, so an application never knows which vendor is talking.

Extracted from voiceapp's `internal/providers/cadence`, where it orchestrates
production call traffic. This module is the **engines layer** — the provider
matrix and its contracts. Conversation orchestration (mediators, turn-taking)
stays in the consuming application.

## Engines

| Role | Engines |
|---|---|
| STT | Deepgram (streaming, endpointing + trailing-silence fallback), Cartesia, Whisper/WhisperLive |
| TTS | ElevenLabs, Deepgram, Cartesia, Inworld, Kokoro — HTTP and warm-WebSocket streaming |
| LLM | OpenAI-compatible (chat completions), Anthropic, Gemini |

All interfaces are in `engines/types.go`:

- `StreamingSTTEngine` — push μ-law 8 kHz frames, receive partials, finals and
  server-VAD speech start/stop events.
- `StreamingTTSEngine` — one warm connection per call; text chunks in, audio
  chunks out; context cancellation is barge-in.
- `LLMEngine` — event-streamed chat with tool-call support.
- `UsageReporter` — per-session audio seconds, tokens in/out and cached-token
  counts for cost attribution.

`audio/` carries the μ-law/PCM16 codec and resampling helpers. The root
package has the sentence buffer (LLM stream → TTS-sized sentences) and a
local energy VAD for engines without server-side endpointing.

## Observability

Cadence records nothing itself. Install hooks from your own metrics system:

```go
engines.OnTTSStreamEnd = func(engine, status string, seconds float64) {
    myMetrics.TTSStreams.WithLabelValues(engine, status).Inc()
}
```

## Provenance

History begins at the extraction (2026-08-06); prior evolution lives in the
voiceapp repository.
