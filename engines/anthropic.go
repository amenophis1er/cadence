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
)

// anthropicAPIVersion pins the Anthropic-Version header. Anthropic
// versions everything via this header (their stated stability
// contract) — without it you get whatever the account-default is at
// request time, which can shift under us.
const anthropicAPIVersion = "2023-06-01"

// AnthropicConfig configures the native Anthropic /v1/messages
// streaming engine. Different request/response shape from the
// OpenAI-compatible chat.go engine — see the package docs in the
// engine source for the translation rules.
type AnthropicConfig struct {
	URL       string // e.g. https://api.anthropic.com (we append /v1/messages)
	APIKey    string // x-api-key header value
	Model     string // e.g. claude-sonnet-4-5, claude-haiku-4-5
	MaxTokens int    // required by Anthropic; defaults to 4096
}

type anthropicEngine struct {
	cfg    AnthropicConfig
	client *http.Client

	usageMu                                        sync.Mutex
	tokensIn, tokensOut, cachedIn, cacheCreationIn int
}

// NewAnthropic builds a native Anthropic Messages streaming engine.
// Auth via x-api-key header (NOT Bearer); request body uses a
// top-level `system` field separate from `messages`; tool use is
// expressed as content blocks rather than a parallel tool_calls
// array. Streaming response is SSE with named event types
// (message_start / content_block_* / message_delta / message_stop).
func NewAnthropic(cfg AnthropicConfig) LLMEngine {
	if cfg.URL == "" {
		cfg.URL = "https://api.anthropic.com"
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 4096
	}
	// timeout=0 — SSE streams may run for minutes; cancellation comes
	// from the caller's context.
	return &anthropicEngine{
		cfg:    cfg,
		client: newEngineHTTPClient(0),
	}
}

func (a *anthropicEngine) Name() string { return "anthropic" }

func (a *anthropicEngine) Usage() Usage {
	a.usageMu.Lock()
	defer a.usageMu.Unlock()
	return Usage{
		TokensIn:            a.tokensIn,
		TokensOut:           a.tokensOut,
		CachedTokens:        a.cachedIn,
		CacheCreationTokens: a.cacheCreationIn,
	}
}

// Stream sends one Messages request and emits LLMEvents as the
// response streams in. Closes `out` on completion (success or error);
// a terminal "done" event is always emitted first (see emitTerminalDone
// for the failure-path semantics).
func (a *anthropicEngine) Stream(ctx context.Context, messages []LLMMessage, toolDefs []ToolDef, out chan<- LLMEvent) error {
	streamOK := false
	defer func() {
		if !streamOK {
			emitTerminalDone(ctx, out)
		}
		close(out)
	}()

	body, err := json.Marshal(buildAnthropicRequest(a.cfg, messages, toolDefs))
	if err != nil {
		return fmt.Errorf("marshal anthropic request: %w", err)
	}

	endpoint := strings.TrimRight(a.cfg.URL, "/") + "/v1/messages"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("x-api-key", a.cfg.APIKey)
	req.Header.Set("anthropic-version", anthropicAPIVersion)

	resp, err := a.client.Do(req)
	if err != nil {
		return fmt.Errorf("anthropic request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		buf, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("anthropic status %d: %s", resp.StatusCode, string(buf))
	}

	usage, err := parseAnthropicSSE(ctx, resp.Body, out)
	streamOK = err == nil // parser emitted its own "done"
	if usage.TokensIn != 0 || usage.TokensOut != 0 {
		a.usageMu.Lock()
		a.tokensIn += usage.TokensIn
		a.tokensOut += usage.TokensOut
		a.cachedIn += usage.CachedTokens
		a.cacheCreationIn += usage.CacheCreationTokens
		a.usageMu.Unlock()
	}
	return err
}

// buildAnthropicRequest translates cadence's normalised LLMMessage
// list into Anthropic's request body. The translations that aren't
// 1-to-1 with OpenAI:
//
//   - Role "system" → top-level `system` field (NOT a message). We
//     concatenate multiple system messages with newlines if the agent
//     config has more than one.
//   - Role "assistant" with ToolCalls → message content is an array
//     of `tool_use` blocks (one per ToolCall) instead of a parallel
//     tool_calls field. The text content (if any) is prepended as a
//     `text` block before the tool_use blocks.
//   - Role "tool" → translated to a `user` role message with a
//     `tool_result` content block; Anthropic doesn't have a "tool"
//     role at all.
func buildAnthropicRequest(cfg AnthropicConfig, messages []LLMMessage, toolDefs []ToolDef) map[string]interface{} {
	var systemParts []string
	msgs := make([]map[string]interface{}, 0, len(messages))

	for _, m := range messages {
		switch m.Role {
		case "system":
			if m.Content != "" {
				systemParts = append(systemParts, m.Content)
			}

		case "user":
			msgs = append(msgs, map[string]interface{}{
				"role":    "user",
				"content": m.Content,
			})

		case "assistant":
			// Assistant-with-tools is the only multi-block path.
			// Plain assistant text uses a string content field.
			if len(m.ToolCalls) == 0 {
				msgs = append(msgs, map[string]interface{}{
					"role":    "assistant",
					"content": m.Content,
				})
				continue
			}
			blocks := make([]map[string]interface{}, 0, 1+len(m.ToolCalls))
			if m.Content != "" {
				blocks = append(blocks, map[string]interface{}{
					"type": "text",
					"text": m.Content,
				})
			}
			for _, tc := range m.ToolCalls {
				// Anthropic wants `input` as a JSON object; cadence
				// stores it as a JSON string. Decode if possible;
				// fall back to {} on malformed JSON rather than
				// failing the whole turn.
				var input interface{}
				if tc.Arguments != "" {
					if err := json.Unmarshal([]byte(tc.Arguments), &input); err != nil {
						input = map[string]interface{}{}
					}
				} else {
					input = map[string]interface{}{}
				}
				blocks = append(blocks, map[string]interface{}{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Name,
					"input": input,
				})
			}
			msgs = append(msgs, map[string]interface{}{
				"role":    "assistant",
				"content": blocks,
			})

		case "tool":
			msgs = append(msgs, map[string]interface{}{
				"role": "user",
				"content": []map[string]interface{}{
					{
						"type":        "tool_result",
						"tool_use_id": m.ToolCallID,
						"content":     m.Content,
					},
				},
			})
		}
	}

	req := map[string]interface{}{
		"model":      cfg.Model,
		"max_tokens": cfg.MaxTokens,
		"messages":   msgs,
		"stream":     true,
	}
	if len(systemParts) > 0 {
		req["system"] = strings.Join(systemParts, "\n\n")
	}
	if len(toolDefs) > 0 {
		tools := make([]map[string]interface{}, len(toolDefs))
		for i, t := range toolDefs {
			tools[i] = map[string]interface{}{
				"name":         t.Name,
				"description":  t.Description,
				"input_schema": t.Parameters,
			}
		}
		req["tools"] = tools
	}
	return req
}

// parseAnthropicSSE consumes Anthropic's named-event SSE stream and
// emits cadence LLMEvents. Event flow per turn:
//
//	message_start
//	content_block_start { content_block: {type: "text" | "tool_use", id, name, ...} }
//	content_block_delta { delta: {type: "text_delta", text} }                   (text)
//	content_block_delta { delta: {type: "input_json_delta", partial_json} }     (tool args)
//	content_block_stop
//	... repeat for additional content blocks ...
//	message_delta { delta: {stop_reason}, usage }
//	message_stop
//
// We accumulate tool-call args by content-block index, then emit one
// `tool_calls` LLMEvent at message_stop time (matching the
// OpenAI-compat path's contract — supervisor expects a single batched
// tool_calls event, not one-per-tool).
func parseAnthropicSSE(ctx context.Context, body io.Reader, out chan<- LLMEvent) (Usage, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

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
		if !strings.HasPrefix(line, "data:") {
			// `event:` lines, blank separators, comments — all
			// ignored. The data line carries the JSON payload with
			// its own `type` field, which is what we dispatch on.
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}

		// Generic envelope — every Anthropic SSE event has a `type`.
		// We only fully decode the events we care about; the rest
		// (ping, etc.) are no-ops. message_start carries usage.input_tokens
		// (set once); message_delta carries usage.output_tokens (cumulative).
		var env struct {
			Type         string `json:"type"`
			Index        int    `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
				StopReason  string `json:"stop_reason"`
			} `json:"delta"`
			Message struct {
				Usage struct {
					InputTokens              int `json:"input_tokens"`
					OutputTokens             int `json:"output_tokens"`
					CacheReadInputTokens     int `json:"cache_read_input_tokens"`
					CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				} `json:"usage"`
			} `json:"message"`
			Usage struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &env); err != nil {
			continue // tolerant: skip malformed lines
		}

		switch env.Type {
		case "message_start":
			if env.Message.Usage.InputTokens > 0 {
				usage.TokensIn = env.Message.Usage.InputTokens
			}
			if env.Message.Usage.OutputTokens > 0 {
				usage.TokensOut = env.Message.Usage.OutputTokens
			}
			// Anthropic emits cache hit/write counts on message_start. The
			// total input_tokens above EXCLUDES cache_read_input_tokens, so
			// we add the cached read count into the surfaced TokensIn to
			// match the OpenAI shape where prompt_tokens is the total
			// (including cached). See Anthropic docs "Prompt caching".
			if env.Message.Usage.CacheReadInputTokens > 0 {
				usage.CachedTokens = env.Message.Usage.CacheReadInputTokens
				usage.TokensIn += env.Message.Usage.CacheReadInputTokens
			}
			if env.Message.Usage.CacheCreationInputTokens > 0 {
				usage.CacheCreationTokens = env.Message.Usage.CacheCreationInputTokens
			}

		case "content_block_start":
			if env.ContentBlock.Type == "tool_use" {
				for env.Index >= len(pending) {
					pending = append(pending, nil)
				}
				pending[env.Index] = &pendingToolCall{
					ID:   env.ContentBlock.ID,
					Name: env.ContentBlock.Name,
				}
			}

		case "content_block_delta":
			switch env.Delta.Type {
			case "text_delta":
				if env.Delta.Text != "" {
					out <- LLMEvent{Type: "text", TextDelta: env.Delta.Text}
				}
			case "input_json_delta":
				if env.Index < len(pending) && pending[env.Index] != nil {
					pending[env.Index].Args.WriteString(env.Delta.PartialJSON)
				}
			}

		case "message_delta":
			if env.Delta.StopReason != "" {
				finishReason = env.Delta.StopReason
			}
			if env.Usage.OutputTokens > 0 {
				usage.TokensOut = env.Usage.OutputTokens
			}

		case "message_stop":
			// Terminal — nothing more after this.
		}
	}
	if err := scanner.Err(); err != nil {
		return usage, fmt.Errorf("anthropic sse scan: %w", err)
	}

	if len(pending) > 0 {
		calls := make([]ToolCall, 0, len(pending))
		for _, p := range pending {
			if p == nil {
				continue
			}
			args := p.Args.String()
			if args == "" {
				args = "{}"
			}
			calls = append(calls, ToolCall{
				ID:        p.ID,
				Name:      p.Name,
				Arguments: args,
			})
		}
		if len(calls) > 0 {
			out <- LLMEvent{Type: "tool_calls", ToolCalls: calls}
		}
	}

	out <- LLMEvent{Type: "done", FinishReason: finishReason}
	return usage, nil
}
