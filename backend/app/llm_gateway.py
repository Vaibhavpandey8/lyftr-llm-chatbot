from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import litellm
from litellm.caching.caching import Cache

from app.config import KeyEntry, Settings
from app.failure_sim import build_simulated_error, get_simulated_status

logger = logging.getLogger(__name__)

# Silence noisy LiteLLM debug banners and telemetry in production
litellm.telemetry = False
litellm.suppress_debug_info = True


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails to produce a valid completion."""


@dataclass(frozen=True)
class GatewayResult:
    reply: str
    resolution_source: str  # "LIVE_INFERENCE" | "LLM_CACHE"
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


def _is_failover_error(exc: Exception) -> bool:
    """Return True if this provider exception should trigger automatic key failover."""
    failover_types = (
        litellm.RateLimitError,
        litellm.ServiceUnavailableError,
        litellm.InternalServerError,
        litellm.BadGatewayError,
        litellm.APIConnectionError,
        litellm.Timeout,
        litellm.AuthenticationError,
        litellm.PermissionDeniedError,
        litellm.BudgetExceededError,
    )
    if isinstance(exc, failover_types):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code in {401, 403, 408, 429} or status_code >= 500):
        return True

    lowered = str(exc).lower()
    quota_or_transient_markers = (
        "429",
        "rate limit",
        "rate_limit",
        "quota",
        "resource_exhausted",
        "resource exhausted",
        "exceeded your current quota",
        "insufficient_quota",
        "overloaded",
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "bad gateway",
        "internal server error",
        "timeout",
        "timed out",
    )
    return any(marker in lowered for marker in quota_or_transient_markers)


def _extract_reply_text(response: Any) -> str:
    """Extract non-empty assistant text from a LiteLLM ModelResponse or dict."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)

    return str(content).strip() if content is not None else ""


def _extract_tokens(
    response: Any,
    model: str,
    messages: list[dict[str, str]],
    reply: str,
) -> tuple[int, int]:
    """Extract per-request (input_tokens, output_tokens) with safe LiteLLM fallback."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)

    if prompt_tokens is None or int(prompt_tokens) < 0:
        try:
            prompt_tokens = int(litellm.token_counter(model=model, messages=messages))
        except Exception:
            prompt_tokens = max(1, sum(len(m.get("content", "").split()) for m in messages))

    if completion_tokens is None or int(completion_tokens) < 0:
        try:
            completion_tokens = int(litellm.token_counter(model=model, text=reply))
        except Exception:
            completion_tokens = max(1, len(reply.split()))

    return int(prompt_tokens), int(completion_tokens)


class LLMGateway:
    """LiteLLM-backed gateway providing ordered key failover and built-in response caching."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key_pool: list[KeyEntry] = settings.key_pool
        self.cache_enabled: bool = settings.cache_enabled
        self.cache: Optional[Cache] = None

        if self.cache_enabled:
            if settings.cache_type == "disk":
                self.cache = Cache(type="disk", disk_cache_dir=settings.disk_cache_dir)
            else:
                self.cache = Cache(type="local")
            litellm.cache = self.cache
        else:
            litellm.cache = None

    def clear_cache(self) -> None:
        """Clear LiteLLM cache (primarily useful for test isolation)."""
        if self.cache is None:
            return
        if hasattr(self.cache, "cache") and hasattr(self.cache.cache, "flush_cache"):
            self.cache.cache.flush_cache()
        elif hasattr(self.cache, "flush_cache"):
            self.cache.flush_cache()

    def _ordered_models(self) -> list[str]:
        seen: list[str] = []
        for entry in self.key_pool:
            if entry.model not in seen:
                seen.append(entry.model)
        if not seen:
            seen.append(self.settings.llm_model)
        return seen

    async def _get_from_litellm_cache(
        self,
        messages: list[dict[str, str]],
    ) -> Optional[tuple[str, Any]]:
        """Look up (model, messages) in LiteLLM's built-in cache."""
        if not self.cache_enabled or self.cache is None:
            return None

        for model in self._ordered_models():
            try:
                cached = await self.cache.async_get_cache(
                    model=model,
                    messages=messages,
                )
                if cached is not None:
                    return model, cached
            except Exception as exc:
                logger.warning("LiteLLM cache lookup failed (continuing to live inference): %s", exc)
        return None

    async def _store_in_litellm_cache(
        self,
        model: str,
        messages: list[dict[str, str]],
        response: Any,
    ) -> None:
        """Store a successful live completion in LiteLLM's built-in cache."""
        if not self.cache_enabled or self.cache is None:
            return
        try:
            await self.cache.async_add_cache(
                result=response,
                model=model,
                messages=messages,
            )
        except Exception as exc:
            logger.warning("LiteLLM cache write failed: %s", exc)

    def _simulation_affects_pool(self, simulation_rules: dict[str, int]) -> bool:
        if not simulation_rules or not self.settings.enable_failure_simulation:
            return False
        if "*" in simulation_rules:
            return True
        pool_ids = {entry.key_id for entry in self.key_pool}
        return any(key_id in pool_ids for key_id in simulation_rules)

    async def generate(
        self,
        messages: list[dict[str, str]],
        simulation_rules: Optional[dict[str, int]] = None,
    ) -> GatewayResult:
        """Execute a chat completion with LiteLLM caching and deterministic key failover."""
        start_time = time.perf_counter()
        active_sim_rules = (
            simulation_rules
            if (simulation_rules and self.settings.enable_failure_simulation)
            else {}
        )
        bypass_cache_read = self._simulation_affects_pool(active_sim_rules)

        # 1. Check LiteLLM's built-in cache first (unless failure simulation is actively targeting the pool)
        if not bypass_cache_read:
            cached_hit = await self._get_from_litellm_cache(messages)
            if cached_hit is not None:
                cached_model, cached_response = cached_hit
                reply = _extract_reply_text(cached_response)
                if reply:
                    input_tokens, output_tokens = _extract_tokens(
                        cached_response, cached_model, messages, reply
                    )
                    latency_ms = max(0, int(round((time.perf_counter() - start_time) * 1000)))
                    return GatewayResult(
                        reply=reply,
                        resolution_source="LLM_CACHE",
                        model=cached_model,
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        key_id=None,
                        fallback_attempts=0,
                    )

        # 2. Live inference with ordered key pool failover (key_1 -> key_2 -> ...)
        if not self.key_pool:
            logger.error("No API keys configured in LLM_API_KEYS")
            raise AllProvidersUnavailableError("all providers unavailable")

        failed_attempts = 0
        for entry in self.key_pool:
            try:
                sim_status = get_simulated_status(active_sim_rules, entry.key_id)
                if sim_status is not None:
                    raise build_simulated_error(sim_status, entry.model, entry.key_id)

                response = await litellm.acompletion(
                    model=entry.model,
                    messages=messages,
                    api_key=entry.api_key,
                    num_retries=0,
                    timeout=self.settings.llm_timeout_seconds,
                    caching=False,  # Cache lookup/store is managed explicitly via LiteLLM Cache above/below
                )

                reply = _extract_reply_text(response)
                if not reply:
                    raise litellm.InternalServerError(
                        message=f"Empty completion content returned by {entry.key_id}",
                        llm_provider=entry.model.split("/", 1)[0] if "/" in entry.model else "llm",
                        model=entry.model,
                    )

                input_tokens, output_tokens = _extract_tokens(response, entry.model, messages, reply)

                # Store real completion in LiteLLM's built-in cache
                await self._store_in_litellm_cache(entry.model, messages, response)

                latency_ms = max(1, int(round((time.perf_counter() - start_time) * 1000)))
                return GatewayResult(
                    reply=reply,
                    resolution_source="LIVE_INFERENCE",
                    model=entry.model,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=entry.key_id,
                    fallback_attempts=failed_attempts,
                )

            except Exception as exc:
                failed_attempts += 1
                if _is_failover_error(exc):
                    logger.warning(
                        "Key %s (model=%s) failed with failover-eligible error (%s); trying next key.",
                        entry.key_id,
                        entry.model,
                        type(exc).__name__,
                    )
                    continue
                logger.warning(
                    "Key %s (model=%s) failed unexpectedly (%s); trying next key.",
                    entry.key_id,
                    entry.model,
                    type(exc).__name__,
                )
                continue

        raise AllProvidersUnavailableError("all providers unavailable")
