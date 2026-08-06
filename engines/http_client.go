package engines

import (
	"net/http"
	"time"
)

// newEngineHTTPClient builds an http.Client tuned for cadence engines:
// HTTP/2-preferred, generous keep-alive pool sized for many concurrent
// calls. The request-level timeout covers connect + first byte; the
// streaming body itself is bounded by the caller's context, not the
// client timeout. Pass timeout=0 for streaming engines that mustn't
// time out the body (e.g. the LLM chat completions stream).
//
// ForceAttemptHTTP2 is harmless on plain-HTTP localhost connections
// (HTTP/2 needs TLS or h2c, neither present in self-hosted speaches),
// so we set it unconditionally rather than parameterising it.
func newEngineHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 64,
			IdleConnTimeout:     90 * time.Second,
			ForceAttemptHTTP2:   true,
		},
	}
}
