package engines

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ChatConfig configures an OpenAI-compatible chat completions engine
// (Ollama, vLLM, OpenAI itself, anything that speaks /v1/chat/completions).
type ChatConfig struct {
	URL         string  // e.g. http://localhost:11434/v1/chat/completions
	APIKey      string  // optional bearer
	Model       string  // e.g. gpt-4o-mini, llama3.1:8b
	Temperature float64 // optional, 0 = use server default
	Timeout     time.Duration
}

type chatEngine struct {
	cfg    ChatConfig
	client *http.Client

	usageMu                       sync.Mutex
	tokensIn, tokensOut, cachedIn int
}

// NewOpenAICompat builds an SSE-streaming chat-completions engine.
//
// The HTTP client has no global timeout because SSE responses can run for
// minutes; cancellation is driven by the caller's context. cfg.Timeout, when
// non-zero, is used by callers as a per-request deadline they layer on top.
func NewOpenAICompat(cfg ChatConfig) LLMEngine {
	// timeout=0 → no client-level deadline (SSE may run for minutes).
	// Cancellation is driven by the caller's context.
	return &chatEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(0),
	}
}

func (c *chatEngine) Name() string { return "openai-compat" }

func (c *chatEngine) Usage() Usage {
	c.usageMu.Lock()
	defer c.usageMu.Unlock()
	return Usage{TokensIn: c.tokensIn, TokensOut: c.tokensOut, CachedTokens: c.cachedIn}
}

// Stream sends one chat completion request and emits LLMEvents as the response
// streams in. Closes `out` on completion (success or error).
func (c *chatEngine) Stream(ctx context.Context, messages []LLMMessage, toolDefs []ToolDef, out chan<- LLMEvent) error {
	defer close(out)

	body := buildChatRequest(c.cfg, messages, toolDefs)
	raw, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.URL, bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("chat request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chat status %d: %s", resp.StatusCode, string(buf))
	}

	usage, err := parseSSE(ctx, resp.Body, out)
	if usage.TokensIn != 0 || usage.TokensOut != 0 {
		c.usageMu.Lock()
		c.tokensIn += usage.TokensIn
		c.tokensOut += usage.TokensOut
		c.cachedIn += usage.CachedTokens
		c.usageMu.Unlock()
	}
	return err
}

func buildChatRequest(cfg ChatConfig, messages []LLMMessage, toolDefs []ToolDef) map[string]interface{} {
	msgs := make([]map[string]interface{}, 0, len(messages))
	for _, m := range messages {
		entry := map[string]interface{}{"role": m.Role}
		if m.Content != "" {
			entry["content"] = m.Content
		}
		if m.Name != "" {
			entry["name"] = m.Name
		}
		if m.ToolCallID != "" {
			entry["tool_call_id"] = m.ToolCallID
		}
		if len(m.ToolCalls) > 0 {
			tcs := make([]map[string]interface{}, len(m.ToolCalls))
			for i, tc := range m.ToolCalls {
				tcs[i] = map[string]interface{}{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]interface{}{
						"name":      tc.Name,
						"arguments": tc.Arguments,
					},
				}
			}
			entry["tool_calls"] = tcs
		}
		msgs = append(msgs, entry)
	}

	req := map[string]interface{}{
		"model":    cfg.Model,
		"messages": msgs,
		"stream":   true,
		// include_usage asks the server to emit a final SSE chunk with
		// usage.{prompt,completion}_tokens. OpenAI honours this; many
		// OpenAI-compat servers (vLLM/Ollama) ignore it silently — the
		// parser tolerates that case (zero tokens recorded, cost
		// calculator falls back to cost_source=computed with zero cents).
		"stream_options": map[string]interface{}{"include_usage": true},
	}
	if cfg.Temperature > 0 {
		req["temperature"] = cfg.Temperature
	}
	if len(toolDefs) > 0 {
		tools := make([]map[string]interface{}, len(toolDefs))
		for i, t := range toolDefs {
			tools[i] = map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.Parameters,
				},
			}
		}
		req["tools"] = tools
	}
	return req
}

// parseSSE consumes an OpenAI-compatible SSE stream of chat.completion.chunk
// events, accumulating tool-call argument fragments by index, and emits one
// LLMEvent per text delta, plus a final tool_calls / done event.
func parseSSE(ctx context.Context, body io.Reader, out chan<- LLMEvent) (Usage, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	// Tool calls arrive over multiple chunks, each tagged with an Index that
	// identifies which call in the LLM's parallel output the fragment belongs
	// to. We accumulate by index and emit in index order so the supervisor
	// sees calls in the same order the model produced them — map iteration
	// would scramble this.
	type pendingToolCall struct {
		ID   string
		Name string
		Args strings.Builder
	}
	var pending []*pendingToolCall
	finishReason := ""
	var usage Usage

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return usage, ctx.Err()
		default:
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage struct {
				PromptTokens        int `json:"prompt_tokens"`
				CompletionTokens    int `json:"completion_tokens"`
				PromptTokensDetails struct {
					// CachedTokens is the OpenAI implicit-cache hit count
					// (≥1024-token prefix matches against the org-scoped cache,
					// reused for ~5-10 min). Mirrored by most OpenAI-compatible
					// proxies (Groq, OpenRouter, LiteLLM) when their upstream
					// surfaces it; absent on vendors without prefix caching.
					CachedTokens int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue // tolerant: skip malformed lines
		}
		// usage chunk arrives last (when stream_options.include_usage:true
		// is honoured), with empty choices[]. Capture before the
		// len(Choices)==0 short-circuit below would skip it.
		if chunk.Usage.PromptTokens > 0 {
			usage.TokensIn = chunk.Usage.PromptTokens
		}
		if chunk.Usage.CompletionTokens > 0 {
			usage.TokensOut = chunk.Usage.CompletionTokens
		}
		if chunk.Usage.PromptTokensDetails.CachedTokens > 0 {
			usage.CachedTokens = chunk.Usage.PromptTokensDetails.CachedTokens
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		ch := chunk.Choices[0]

		if ch.Delta.Content != "" {
			out <- LLMEvent{Type: "text", TextDelta: ch.Delta.Content}
		}

		for _, tc := range ch.Delta.ToolCalls {
			for tc.Index >= len(pending) {
				pending = append(pending, nil)
			}
			p := pending[tc.Index]
			if p == nil {
				p = &pendingToolCall{}
				pending[tc.Index] = p
			}
			if tc.ID != "" {
				p.ID = tc.ID
			}
			if tc.Function.Name != "" {
				p.Name = tc.Function.Name
			}
			if tc.Function.Arguments != "" {
				p.Args.WriteString(tc.Function.Arguments)
			}
		}

		if ch.FinishReason != "" {
			finishReason = ch.FinishReason
		}
	}
	if err := scanner.Err(); err != nil {
		return usage, fmt.Errorf("sse scan: %w", err)
	}

	if len(pending) > 0 {
		calls := make([]ToolCall, 0, len(pending))
		for _, p := range pending {
			if p == nil {
				continue
			}
			calls = append(calls, ToolCall{
				ID:        p.ID,
				Name:      p.Name,
				Arguments: p.Args.String(),
			})
		}
		out <- LLMEvent{Type: "tool_calls", ToolCalls: calls}
	}

	out <- LLMEvent{Type: "done", FinishReason: finishReason}
	return usage, nil
}
