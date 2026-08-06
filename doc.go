// Package cadence provides provider-agnostic voice AI engines: streaming
// speech-to-text, text-to-speech and LLM chat behind small uniform
// interfaces, plus the glue a real-time voice pipeline needs between them
// (sentence-aligned buffering for TTS, local energy VAD).
//
// The engine implementations and their contracts live in the engines
// subpackage; G.711/PCM audio helpers live in audio. This root package
// carries the pieces that sit between engines: RunSentenceBuffer turns an
// LLM token stream into TTS-sized sentences, and EnergyVAD detects speech
// onset/offset for STT engines without server-side endpointing.
//
// Cadence deliberately does not include a conversation loop — turn-taking,
// barge-in policy and dialogue state belong to the application.
package cadence
