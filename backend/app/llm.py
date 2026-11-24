import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm.caching.caching import Cache

logger = logging.getLogger(__name__)

# Initialize LiteLLM local cache
try:
    litellm.cache = Cache(type="local")
except Exception as e:
    logger.warning("Could not initialize litellm.cache: %s", e)

# Memory store backing identical (model, messages) lookups
_LOCAL_COMPLETION_CACHE: Dict[str, Dict[str, Any]] = {}


class ProviderUnavailableException(Exception):
    """Raised when all provider keys have been exhausted or failed."""
    pass


def parse_simulate_failure(header_val: Optional[str]) -> Dict[str, int]:
    """
    Parses comma-separated <key_id>=<status> pairs:
    e.g. "key_1=429, key_2=500, *=429"
    Allowed statuses: 429, 500, 502, 503
    """
    if not header_val:
        return {}
    results = {}
    valid_statuses = {429, 500, 502, 503}
    pairs = [p.strip() for p in header_val.split(",") if p.strip()]
    for pair in pairs:
        if "=" in pair:
            key_part, status_part = pair.split("=", 1)
            key_id = key_part.strip()
            status_str = status_part.strip()
            if status_str.isdigit():
                status_int = int(status_str)
                if status_int in valid_statuses:
                    results[key_id] = status_int
    return results


def make_cache_key(model: str, messages: List[Dict[str, str]]) -> str:
    """Cache key strictly depends on (model, full_messages_list)."""
    return json.dumps({"model": model, "messages": messages}, sort_keys=True)


def execute_chat(
    model: str,
    messages: List[Dict[str, str]],
    api_keys: List[str],
    simulate_failure_header: Optional[str] = None,
    enable_failure_simulation: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Executes chat completion with:
    - LiteLLM local cache (identical model + full messages list)
    - Sequential key failover (key_1, key_2, ...)
    - X-Simulate-Failure header evaluation
    - Per-request token accounting
    """
    start_time = time.perf_counter()
    cache_key = make_cache_key(model, messages)

    # Parse simulation header if enabled
    sim_failures: Dict[str, int] = {}
    if enable_failure_simulation and simulate_failure_header:
        sim_failures = parse_simulate_failure(simulate_failure_header)

    # LiteLLM Cache check (only when not actively simulating failure on this single request)
    if not sim_failures and cache_key in _LOCAL_COMPLETION_CACHE:
        cached = _LOCAL_COMPLETION_CACHE[cache_key]
        latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))
        metadata = {
            "resolution_source": "LLM_CACHE",
            "model": cached["model"],
            "latency_ms": latency_ms,
            "input_tokens": cached["input_tokens"],
            "output_tokens": cached["output_tokens"],
            "key_id": None,
            "fallback_attempts": 0,
        }
        return cached["reply"], metadata

    fallback_attempts = 0
    num_keys = len(api_keys)

    for idx, key in enumerate(api_keys):
        key_id = f"key_{idx + 1}"

        # 1. Check if failure simulation applies to this key or wildcard '*'
        if enable_failure_simulation and sim_failures:
            simulated_status = None
            if key_id in sim_failures:
                simulated_status = sim_failures[key_id]
            elif "*" in sim_failures:
                simulated_status = sim_failures["*"]

            if simulated_status is not None:
                logger.info(
                    "Simulating failure %d for %s (attempt %d)",
                    simulated_status,
                    key_id,
                    fallback_attempts,
                )
                fallback_attempts += 1
                continue

        # 2. Attempt real LiteLLM call
        try:
            call_start = time.perf_counter()
            response = litellm.completion(
                model=model,
                messages=messages,
                api_key=key,
                timeout=15,
                caching=True,
            )

            # Check if LiteLLM internal cache hit
            is_litellm_cache = False
            hidden_params = getattr(response, "_hidden_params", {}) or {}
            if hidden_params.get("cache_hit") is True:
                is_litellm_cache = True
            elif getattr(response, "cache_hit", False) is True:
                is_litellm_cache = True

            # Extract reply text
            reply = ""
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message"):
                    reply = getattr(first_choice.message, "content", "") or ""
                elif isinstance(first_choice, dict):
                    reply = first_choice.get("message", {}).get("content", "")
            elif isinstance(response, dict) and "choices" in response:
                reply = response["choices"][0].get("message", {}).get("content", "")

            # Extract per-request token usage
            input_tokens = 0
            output_tokens = 0
            usage = getattr(response, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
            elif isinstance(response, dict) and "usage" in response:
                u = response["usage"]
                input_tokens = u.get("prompt_tokens", 0) or 0
                output_tokens = u.get("completion_tokens", 0) or 0

            # Fallback estimation if provider returns 0 tokens
            if input_tokens <= 0:
                input_tokens = max(1, sum(len(m.get("content", "").split()) for m in messages))
            if output_tokens <= 0:
                output_tokens = max(1, len(reply.split()))

            latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))

            if is_litellm_cache:
                metadata = {
                    "resolution_source": "LLM_CACHE",
                    "model": getattr(response, "model", model) or model,
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "key_id": None,
                    "fallback_attempts": 0,
                }
            else:
                metadata = {
                    "resolution_source": "LIVE_INFERENCE",
                    "model": getattr(response, "model", model) or model,
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "key_id": key_id,
                    "fallback_attempts": fallback_attempts,
                }

            # Store in local cache for deterministic instant hits
            _LOCAL_COMPLETION_CACHE[cache_key] = {
                "reply": reply,
                "model": metadata["model"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

            return reply, metadata

        except Exception as exc:
            # Check for 429, timeout, 5xx, or quota errors
            exc_str = str(exc).lower()
            status_code = getattr(exc, "status_code", None)
            is_failover_worthy = False

            if status_code in (429, 500, 502, 503, 504):
                is_failover_worthy = True
            elif any(s in exc_str for s in ("429", "rate limit", "quota", "timeout", "500", "502", "503", "unavailable", "server error")):
                is_failover_worthy = True
            else:
                # Any provider communication failure triggers failover to next key
                is_failover_worthy = True

            logger.warning("Error with key %s (%s): %s", key_id, type(exc).__name__, exc)
            fallback_attempts += 1
            continue

    # If all keys failed or were simulated as failed
    raise ProviderUnavailableException("all providers unavailable")
