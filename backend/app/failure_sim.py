from __future__ import annotations

from typing import Optional

import litellm

SUPPORTED_FAILURE_STATUSES = {429, 500, 502, 503}


def parse_simulate_failure_header(header_value: Optional[str]) -> dict[str, int]:
    """Parse X-Simulate-Failure header into {key_id_or_star: status_code}.

    Examples:
        "key_1=429" -> {"key_1": 429}
        "key_1=429,key_2=500" -> {"key_1": 429, "key_2": 500}
        "*=429" -> {"*": 429}

    Invalid entries or unsupported status codes are safely ignored.
    """
    if not header_value:
        return {}

    rules: dict[str, int] = {}
    for raw_pair in header_value.split(","):
        pair = raw_pair.strip()
        if not pair or "=" not in pair:
            continue
        key_part, status_part = pair.split("=", 1)
        key_id = key_part.strip()
        if not key_id:
            continue
        try:
            status_code = int(status_part.strip())
        except ValueError:
            continue
        if status_code in SUPPORTED_FAILURE_STATUSES:
            rules[key_id] = status_code
    return rules


def get_simulated_status(rules: dict[str, int], key_id: str) -> Optional[int]:
    """Return simulated failure HTTP status for key_id, or None if not simulated."""
    if not rules:
        return None
    if key_id in rules:
        return rules[key_id]
    if "*" in rules:
        return rules["*"]
    return None


def _provider_from_model(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def build_simulated_error(status_code: int, model: str, key_id: str) -> Exception:
    """Build the matching LiteLLM exception for a simulated provider failure."""
    provider = _provider_from_model(model)
    msg = f"Simulated HTTP {status_code} failure for {key_id}"

    if status_code == 429:
        return litellm.RateLimitError(message=msg, llm_provider=provider, model=model)
    if status_code == 500:
        return litellm.InternalServerError(message=msg, llm_provider=provider, model=model)
    if status_code == 502:
        return litellm.BadGatewayError(message=msg, llm_provider=provider, model=model)
    if status_code == 503:
        return litellm.ServiceUnavailableError(message=msg, llm_provider=provider, model=model)
    return litellm.APIError(
        status_code=status_code,
        message=msg,
        llm_provider=provider,
        model=model,
    )
