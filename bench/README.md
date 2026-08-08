# bench — latency characterization harnesses

The measurement harnesses behind
["The anatomy of time-to-first-audio in a voice agent"](https://amenophis.dev/writing/anatomy-of-first-audio/).
Each is a standalone `package main` inside the cadence module.

| Harness | Measures | Needs |
|---|---|---|
| [`sbbench`](sbbench/) | `RunSentenceBuffer` time-to-first-chunk vs token rate × config (360 runs) | nothing — `go run ./bench/sbbench` |
| [`vadbench`](vadbench/) | `EnergyVAD` hesitation-cut rate + commit latency vs offset (1,800 trials) | nothing — `go run ./bench/vadbench` |
| [`e2ebench`](e2ebench/) | Live LLM → sentence buffer → TTS stage decomposition | `OPENAI_API_KEY` (or `LLM_PROVIDER=anthropic` + key) and `ELEVENLABS_API_KEY` |

Results are written as JSON next to the binary. Methodology, limitations,
and the numbers we measured are in the article.
