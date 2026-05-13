"""
LLM model pricing — canonical USD-per-million-tokens table.

Sources:
  - Anthropic: https://platform.claude.com/docs/en/docs/about-claude/models/overview (2026-05-07)
  - OpenAI: https://openai.com/api/pricing/ (2026-05-13)

Pricing lives in code (not in the database) so changes are auditable via PR.
Bump this table whenever a provider publishes new pricing and reference the
source URL in the commit message.
"""

from decimal import Decimal
from typing import Dict


# Per-million-tokens (MTok) pricing in USD.
# Both pinned model IDs and aliases map to the same row so the lookup is
# tolerant of whichever form DSPy/LiteLLM reports back.
PRICING_TABLE: Dict[str, Dict[str, Decimal]] = {
    # Claude Haiku 4.5 — $1 / input MTok, $5 / output MTok
    "claude-haiku-4-5": {
        "input_per_mtok": Decimal("1.00"),
        "output_per_mtok": Decimal("5.00"),
    },
    "claude-haiku-4-5-20251001": {
        "input_per_mtok": Decimal("1.00"),
        "output_per_mtok": Decimal("5.00"),
    },
    # Claude Sonnet 4.6 — $3 / input MTok, $15 / output MTok
    "claude-sonnet-4-6": {
        "input_per_mtok": Decimal("3.00"),
        "output_per_mtok": Decimal("15.00"),
    },
    # Claude Opus 4.7 — $5 / input MTok, $25 / output MTok
    "claude-opus-4-7": {
        "input_per_mtok": Decimal("5.00"),
        "output_per_mtok": Decimal("25.00"),
    },
    # GPT-4o mini — $0.15 / input MTok, $0.60 / output MTok
    "gpt-4o-mini": {
        "input_per_mtok": Decimal("0.15"),
        "output_per_mtok": Decimal("0.60"),
    },
    # GPT-4.1 mini — $0.40 / input MTok, $1.60 / output MTok (OpenAI 2025)
    "gpt-4.1-mini": {
        "input_per_mtok": Decimal("0.40"),
        "output_per_mtok": Decimal("1.60"),
    },
    # GPT-5 Nano — $0.05 / input MTok, $0.40 / output MTok (OpenAI 2026).
    # Used by GreetingAgent v24+.
    "gpt-5-nano": {
        "input_per_mtok": Decimal("0.05"),
        "output_per_mtok": Decimal("0.40"),
    },
    # DeepSeek V4 Flash — $0.14 / input MTok (cache miss), $0.28 / output MTok
    # Used by GreetingAgent v17-v23 (non-thinking mode). Source: api-docs.deepseek.com 2026.
    "deepseek-v4-flash": {
        "input_per_mtok": Decimal("0.14"),
        "output_per_mtok": Decimal("0.28"),
    },
}


_MTOK = Decimal("1000000")


def _normalize_model(model: str) -> str:
    """Strip LiteLLM provider prefixes (e.g. ``anthropic/claude-opus-4-7``)."""
    if not model:
        return ""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    """
    Return USD cost for a single LM call, quantized to 6 decimal places.

    Unknown models or zero-token deterministic agents both return ``Decimal("0")``.
    The caller is responsible for storing the value (NUMERIC(10, 6) on
    ``sf_agent_activations.cost_usd``).
    """
    if not prompt_tokens and not completion_tokens:
        return Decimal("0")

    rates = PRICING_TABLE.get(_normalize_model(model))
    if not rates:
        return Decimal("0")

    input_cost = (Decimal(prompt_tokens) * rates["input_per_mtok"]) / _MTOK
    output_cost = (Decimal(completion_tokens) * rates["output_per_mtok"]) / _MTOK
    return (input_cost + output_cost).quantize(Decimal("0.000001"))
