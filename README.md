# cadence

[![CI](https://github.com/amenophis1er/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/amenophis1er/cadence/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/amenophis1er/cadence.svg)](https://pkg.go.dev/github.com/amenophis1er/cadence)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A provider-agnostic Go engine for low-latency STT→LLM→TTS voice
pipelines.** Streaming engines behind small uniform interfaces, so your
voice application never knows — or cares — which vendor is talking. Swap
providers with a config change, run several side by side, or plug in your
own in-house engine.

cadence owns the **hot path** — the latency-critical loop between a
committed user turn and the first synthesized audio — and deliberately
nothing else:

```
             your voice application
        tools · prompts · business logic
                     │
                     ▼
         ╔═══════════════════════╗
         ║        cadence        ║
 audio → ║  endpoint → STT → LLM ║
         ║               ↓       ║
 audio ← ║      TTS ← buffer     ║
         ╚═══════════════════════╝
                     │
               instrumentation
```

**What cadence is not:** a transport (bring your own telephony or WebRTC —
[xphone](https://github.com/x-phone/xphone-go) if you want a phone line in
Go), an agent framework (turn policy, tools and state stay in your
application), or a deployment platform. Small, embeddable, instrumented —
the engine underneath a voice agent, not the agent.

Battle-tested in a production telephony platform handling real call traffic.

```
go get github.com/amenophis1er/cadence
```

## Engines

| Role | Engines | Streaming |
|---|---|---|
| **STT** | Deepgram (Nova), Cartesia, Whisper / WhisperLive | partials, finals, server-VAD speech events |
| **TTS** | ElevenLabs, Deepgram, Cartesia, Inworld, Kokoro | warm WebSocket per call, sentence-level flush, HTTP fallback |
| **LLM** | OpenAI-compatible (chat completions), Anthropic, Gemini | event stream with tool-call support |

Every role is an interface (`engines/types.go`); anything satisfying it is a
first-class engine — including local/self-hosted models (Ollama and other
OpenAI-compatible servers work out of the box via `NewOpenAICompat`).

## Quick start

### Streaming speech-to-text

```go
stt := engines.NewDeepgramSTT(engines.DeepgramSTTConfig{
    APIKey: os.Getenv("DEEPGRAM_API_KEY"),
    Model:  "nova-3",
})

events := make(chan engines.STTEvent, 64)
if err := stt.Start(ctx, events); err != nil {
    log.Fatal(err)
}
defer stt.Stop()

go func() {
    for frame := range telephonyAudio { // mu-law 8 kHz, 160-byte / 20 ms frames
        stt.PushAudio(frame)
    }
}()

for ev := range events {
    switch ev.Type {
    case engines.STTSpeechStarted:   // caller began talking — barge-in signal
    case engines.STTTranscriptDelta: // live partial: ev.Text
    case engines.STTTranscriptFinal: // committed utterance: ev.Text
    }
}
```

### Streaming text-to-speech

```go
tts := engines.NewElevenTTSWS(engines.ElevenTTSConfig{
    APIKey: os.Getenv("ELEVENLABS_API_KEY"),
    Model:  "eleven_flash_v2_5",
    Voice:  "JBFqnCBsd6RMkjVDRZzb",
})

textCh := make(chan engines.TextChunk)
audioCh := make(chan engines.AudioChunk, 64)

go tts.Stream(ctx, textCh, audioCh) // one warm connection for the whole call

textCh <- engines.TextChunk{Text: "Hello! How can I help?", IsSentenceEnd: true}
// … keep sending sentences as the LLM produces them; close(textCh) when done.
// Cancel ctx to barge-in: the engine drains and exits.

for chunk := range audioCh {
    playToCaller(chunk.Data)
}
```

### Streaming LLM, sentence-aligned for TTS

`RunSentenceBuffer` sits between the LLM's token stream and TTS, flushing on
sentence terminators, clause boundaries, idle timeouts or a hard word cap —
so synthesis starts before generation finishes:

```go
llm := engines.NewOpenAICompat(engines.ChatConfig{
    URL:    "https://api.openai.com/v1/chat/completions",
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4.1",
})

llmEvents := make(chan engines.LLMEvent, 64)
sentences := make(chan string, 8)

go llm.Stream(ctx, messages, nil, llmEvents)
go func() {
    for s := range sentences {
        textCh <- engines.TextChunk{Text: s, IsSentenceEnd: true} // → TTS above
    }
}()

result := cadence.RunSentenceBuffer(ctx, cadence.DefaultSentenceBufferConfig(), llmEvents, sentences)
fmt.Println("assistant said:", result.FullText)
```

### Cost attribution

Engines that implement `UsageReporter` expose per-session consumption —
audio seconds, tokens in/out, and **cached-token counts** (the effective
prompt-cache hit ratio):

```go
if ur, ok := llm.(engines.UsageReporter); ok {
    u := ur.Usage()
    log.Printf("tokens in=%d (cached=%d) out=%d", u.TokensIn, u.CachedTokens, u.TokensOut)
}
```

## Extras

- **`audio/`** — dependency-free G.711 μ-law ↔ PCM16 codec and 8k/16k
  resampling helpers.
- **`EnergyVAD`** — local energy-based voice activity detection for STT
  engines without server-side endpointing (onset debounce, exactly-once
  utterance-end signaling).
- **Observability hooks** — cadence records nothing itself; install
  `engines.OnTTSStreamEnd` (and friends as they grow) to feed Prometheus,
  OpenTelemetry, or plain logs.

## Measured, not claimed

The hot path is characterized, not guessed at:
["The anatomy of time-to-first-audio in a voice agent"](https://amenophis.dev/writing/anatomy-of-first-audio/)
decomposes a live LLM→buffer→TTS chain (median post-commit first audio
1.06 s: ~59 % LLM, ~28 % TTS, ~14 % cadence's buffering) and sweeps the
sentence-buffer and VAD trade-offs. The harnesses live in
[`bench/`](bench/) — reproduce the numbers before trusting them:

| Harness | Measures | Needs |
|---|---|---|
| [`bench/sbbench`](bench/sbbench/) | sentence-buffer time-to-first-chunk vs word rate × config | nothing |
| [`bench/vadbench`](bench/vadbench/) | VAD hesitation-cut rate + commit latency vs offset | nothing |
| [`bench/e2ebench`](bench/e2ebench/) | live LLM → buffer → TTS stage decomposition | provider keys |

## Design notes

- **Interfaces over adapters.** The application depends on `STTEngine` /
  `TTSEngine` / `LLMEngine`; vendors are implementation details. The
  streaming variants (`StreamingSTTEngine`, `StreamingTTSEngine`) are
  optional richer contracts — consumers prefer them when available.
- **Telephony-native.** Audio contracts are μ-law 8 kHz, 20 ms frames —
  what real phone networks speak. `PushAudio` is non-blocking and drops on
  overrun rather than stalling the audio loop.
- **Barge-in is context cancellation.** No custom stop protocol: cancel the
  context, the engine sends the vendor's EOS, drains, and returns.
- **Orchestration stays yours.** Cadence deliberately ships engines, not a
  conversation loop — turn-taking, interruption policy and state belong to
  the application.

## License

MIT — see [LICENSE](LICENSE).
