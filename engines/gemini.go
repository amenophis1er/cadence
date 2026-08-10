package engines

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// GeminiConfig configures the native Google Gemini streaming engine.
// Different request/response shape from OpenAI-compatible — see the
// translation rules in buildGeminiRequest below. Auth is via
// x-goog-api-key header.
type GeminiConfig struct {
	URL    string // base — e.g. https://generativelanguage.googleapis.com
	APIKey string // x-goog-api-key header value
	Model  string // e.g. gemini-2.5-flash, gemini-2.5-pro
}

type geminiEngine struct {
	cfg    GeminiConfig
	client *http.Client

	usageMu             sync.Mutex
	tokensIn, tokensOut int
}

// NewGemini builds a native Gemini streamGenerateContent engine.
func NewGemini(cfg GeminiConfig) LLMEngine {
	if cfg.URL == "" {
		cfg.URL = "https://generativelanguage.googleapis.com"
	}
	return &geminiEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(0),
	}
}

func (g *geminiEngine) Name() string { return "gemini" }

func (g *geminiEngine) Usage() Usage {
	g.usageMu.Lock()
	defer g.usageMu.Unlock()
	return Usage{TokensIn: g.tokensIn, TokensOut: g.tokensOut}
}

// Stream sends one streamGenerateContent request and emits LLMEvents
// as the response streams in. Closes `out` on completion; a terminal
// "done" event is always emitted first (see emitTerminalDone for the
// failure-path semantics).
func (g *geminiEngine) Stream(ctx context.Context, messages []LLMMessage, toolDefs []ToolDef, out chan<- LLMEvent) error {
	streamOK := false
	defer func() {
		if !streamOK {
			emitTerminalDone(ctx, out)
		}
		close(out)
	}()

	if g.cfg.Model == "" {
		return fmt.Errorf("gemini: model not configured")
	}

	body, err := json.Marshal(buildGeminiRequest(g.cfg, messages, toolDefs))
	if err != nil {
		return fmt.Errorf("marshal gemini request: %w", err)
	}

	// streamGenerateContent + alt=sse selects Server-Sent Events
	// framing; without alt=sse the response is a JSON array stream
	// (each chunk is a JSON object, but they're not framed by SSE
	// `data:` prefixes — easier to parse with SSE on).
	endpoint := strings.TrimRight(g.cfg.URL, "/") +
		"/v1beta/models/" + url.PathEscape(g.cfg.Model) + ":streamGenerateContent?alt=sse"

	// Debug log so we can see exactly which model + history shape
	// hit Gemini when a "Multiturn chat is not enabled" 400 fires —
	// the error message itself doesn't echo the model id, and
	// settings-cache staleness can ship the wrong model invisibly.
	slog.Debug("gemini: request",
		"model", g.cfg.Model,
		"messages", len(messages),
		"tools", len(toolDefs),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("x-goog-api-key", g.cfg.APIKey)

	resp, err := g.client.Do(req)
	if err != nil {
		return fmt.Errorf("gemini request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("gemini status %d: %s", resp.StatusCode, string(buf))
	}

	usage, err := parseGeminiSSE(ctx, resp.Body, out)
	streamOK = err == nil // parser emitted its own "done"
	if usage.TokensIn != 0 || usage.TokensOut != 0 {
		g.usageMu.Lock()
		g.tokensIn += usage.TokensIn
		g.tokensOut += usage.TokensOut
		g.usageMu.Unlock()
	}
	return err
}

// buildGeminiRequest translates cadence's normalised LLMMessage list
// into Gemini's request body.
//
// Translations that aren't 1-to-1 with OpenAI:
//
//   - Role "system" → top-level `systemInstruction` field; multiple
//     system messages are concatenated.
//   - Role "assistant" → role "model" (Gemini's term for the same
//     thing); with ToolCalls, content becomes `parts` containing
//     `functionCall` entries instead of a parallel tool_calls field.
//   - Role "tool" → role "function" with a `functionResponse` part
//     referencing the tool name. Gemini doesn't carry a tool_use_id
//     — function calls and responses match by name + ordering.
//   - Tool definitions go inside a single tools[0].functionDeclarations
//     array (Gemini wraps them in a tools envelope).
func buildGeminiRequest(cfg GeminiConfig, messages []LLMMessage, toolDefs []ToolDef) map[string]interface{} {
	var systemParts []string
	contents := make([]map[string]interface{}, 0, len(messages))

	for _, m := range messages {
		switch m.Role {
		case "system":
			if m.Content != "" {
				systemParts = append(systemParts, m.Content)
			}

		case "user":
			contents = append(contents, map[string]interface{}{
				"role":  "user",
				"parts": []map[string]interface{}{{"text": m.Content}},
			})

		case "assistant":
			parts := make([]map[string]interface{}, 0, 1+len(m.ToolCalls))
			if m.Content != "" {
				parts = append(parts, map[string]interface{}{"text": m.Content})
			}
			for _, tc := range m.ToolCalls {
				var args interface{}
				if tc.Arguments != "" {
					if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
						args = map[string]interface{}{}
					}
				} else {
					args = map[string]interface{}{}
				}
				parts = append(parts, map[string]interface{}{
					"functionCall": map[string]interface{}{
						"name": tc.Name,
						"args": args,
					},
				})
			}
			if len(parts) == 0 {
				continue
			}
			contents = append(contents, map[string]interface{}{
				"role":  "model",
				"parts": parts,
			})

		case "tool":
			// Gemini's functionResponse expects the tool's output
			// as a JSON object. Cadence stores it as a JSON string;
			// parse if possible, else wrap as a {result: "..."}
			// envelope so Gemini gets a valid object either way.
			var responseObj interface{}
			if m.Content != "" {
				if err := json.Unmarshal([]byte(m.Content), &responseObj); err != nil {
					responseObj = map[string]interface{}{"result": m.Content}
				}
			} else {
				responseObj = map[string]interface{}{}
			}
			contents = append(contents, map[string]interface{}{
				"role": "function",
				"parts": []map[string]interface{}{
					{
						"functionResponse": map[string]interface{}{
							"name":     m.Name,
							"response": responseObj,
						},
					},
				},
			})
		}
	}

	// Gemini's chat endpoint expects strict user↔model alternation in
	// contents[]. Cadence's history can violate this when barge-in
	// cancels a turn before the LLM emits an assistant message: we
	// then accumulate consecutive `user` messages with no `model` in
	// between (e.g. user says A, barges in mid-thinking, says B).
	// OpenAI/Anthropic handle this fine but Gemini hits a slow path
	// — observed 30+ second responses for runs with three
	// consecutive user messages. Merge same-role neighbours into one
	// part-list so the alternation invariant holds.
	contents = mergeAdjacentGeminiContents(contents)

	req := map[string]interface{}{
		"contents": contents,
	}
	if len(systemParts) > 0 {
		req["systemInstruction"] = map[string]interface{}{
			"parts": []map[string]interface{}{
				{"text": strings.Join(systemParts, "\n\n")},
			},
		}
	}
	if len(toolDefs) > 0 {
		decls := make([]map[string]interface{}, len(toolDefs))
		for i, t := range toolDefs {
			decls[i] = map[string]interface{}{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			}
		}
		req["tools"] = []map[string]interface{}{
			{"functionDeclarations": decls},
		}
	}
	return req
}

// mergeAdjacentGeminiContents folds runs of same-role neighbours in
// the contents[] slice into a single entry, concatenating their
// parts[]. Required for Gemini's strict alternation rule (see the
// comment at the call site). The function role is treated as
// alternating with model since Gemini doesn't strictly require
// alternation between function and model — only between user/model
// for chat — but consecutive `user`s and consecutive `model`s are
// the practical risk.
func mergeAdjacentGeminiContents(contents []map[string]interface{}) []map[string]interface{} {
	if len(contents) <= 1 {
		return contents
	}
	merged := make([]map[string]interface{}, 0, len(contents))
	for _, c := range contents {
		role, _ := c["role"].(string)
		if len(merged) > 0 {
			prev := merged[len(merged)-1]
			prevRole, _ := prev["role"].(string)
			if prevRole == role && (role == "user" || role == "model") {
				// Append parts[] from c onto prev. Both should be
				// []map[string]interface{}; any other shape we leave
				// alone (defensive — the builder always uses the
				// concrete slice type).
				prevParts, ok1 := prev["parts"].([]map[string]interface{})
				curParts, ok2 := c["parts"].([]map[string]interface{})
				if ok1 && ok2 {
					prev["parts"] = append(prevParts, curParts...)
					continue
				}
			}
		}
		merged = append(merged, c)
	}
	return merged
}

// parseGeminiSSE consumes Gemini's SSE stream of GenerateContentResponse
// chunks and emits cadence LLMEvents. Each `data: {...}` line is one
// candidate response with parts[]:
//
//	{"candidates":[{"content":{"role":"model","parts":[{"text":"..."}, {"functionCall":{name, args}}]},
//	                "finishReason":"..."}]}
//
// Text parts are emitted as text deltas immediately. Function calls
// are buffered and emitted as a single batched tool_calls LLMEvent at
// stream end (matching the OpenAI-compat / Anthropic contract).
func parseGeminiSSE(ctx context.Context, body io.Reader, out chan<- LLMEvent) (Usage, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	var calls []ToolCall
	finishReason := ""
	var usage Usage

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return usage, ctx.Err()
		default:
		}

		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}

		var chunk struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						Text         string `json:"text"`
						FunctionCall *struct {
							Name string                 `json:"name"`
							Args map[string]interface{} `json:"args"`
						} `json:"functionCall"`
					} `json:"parts"`
				} `json:"content"`
				FinishReason string `json:"finishReason"`
			} `json:"candidates"`
			UsageMetadata struct {
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
			} `json:"usageMetadata"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue // tolerant
		}
		// usageMetadata is cumulative on each chunk Gemini emits; the
		// final chunk carries the authoritative totals. Overwriting (vs
		// adding) per chunk keeps us aligned with the cumulative model.
		if chunk.UsageMetadata.PromptTokenCount > 0 {
			usage.TokensIn = chunk.UsageMetadata.PromptTokenCount
		}
		if chunk.UsageMetadata.CandidatesTokenCount > 0 {
			usage.TokensOut = chunk.UsageMetadata.CandidatesTokenCount
		}
		if len(chunk.Candidates) == 0 {
			continue
		}
		cand := chunk.Candidates[0]

		for _, part := range cand.Content.Parts {
			if part.Text != "" {
				out <- LLMEvent{Type: "text", TextDelta: part.Text}
			}
			if part.FunctionCall != nil {
				args, err := json.Marshal(part.FunctionCall.Args)
				if err != nil {
					args = []byte("{}")
				}
				calls = append(calls, ToolCall{
					// Gemini doesn't supply an id; synthesise one
					// from the name plus the call's index in this
					// turn so the supervisor can correlate the
					// eventual tool_result back to this call even
					// when the model calls the same function twice.
					ID:        fmt.Sprintf("gemini_%s_%d", part.FunctionCall.Name, len(calls)),
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				})
			}
		}

		if cand.FinishReason != "" {
			finishReason = cand.FinishReason
		}
	}
	if err := scanner.Err(); err != nil {
		return usage, fmt.Errorf("gemini sse scan: %w", err)
	}

	if len(calls) > 0 {
		out <- LLMEvent{Type: "tool_calls", ToolCalls: calls}
	}
	out <- LLMEvent{Type: "done", FinishReason: finishReason}
	return usage, nil
}
