"""
Unit tests for app.core.pricing.compute_cost.

Pricing source: https://platform.claude.com/docs/en/docs/about-claude/models/overview
(captured 2026-05-07).
"""

from decimal import Decimal

import pytest

from app.core.pricing import PRICING_TABLE, compute_cost


def test_haiku_45_input_only():
    # 1,000,000 input tokens at $1/MTok = $1.000000
    assert compute_cost("claude-haiku-4-5", 1_000_000, 0) == Decimal("1.000000")


def test_haiku_45_id_alias_pinned():
    # Pinned snapshot ID resolves to the same row as the alias.
    assert compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0) == Decimal("1.000000")


def test_sonnet_46_mixed():
    # 500k input @ $3/MTok = $1.50, 100k output @ $15/MTok = $1.50, total $3.00
    assert compute_cost("claude-sonnet-4-6", 500_000, 100_000) == Decimal("3.000000")


def test_opus_47_mixed():
    # 200k input @ $5/MTok = $1.00, 50k output @ $25/MTok = $1.25, total $2.25
    assert compute_cost("claude-opus-4-7", 200_000, 50_000) == Decimal("2.250000")


def test_provider_prefix_is_stripped():
    # LiteLLM reports models as "anthropic/claude-...". Prefix must not break lookup.
    assert compute_cost("anthropic/claude-opus-4-7", 1000, 1000) == compute_cost(
        "claude-opus-4-7", 1000, 1000
    )


def test_unknown_model_returns_zero():
    assert compute_cost("gpt-4o-mini", 10_000, 5_000) == Decimal("0")


def test_deterministic_agent_returns_zero():
    # Greeting/Closure agents make no LLM call → 0 tokens → $0 regardless of model.
    assert compute_cost("claude-opus-4-7", 0, 0) == Decimal("0")


def test_empty_model_returns_zero():
    assert compute_cost("", 1000, 1000) == Decimal("0")


def test_quantized_to_six_decimals():
    # Small token counts should round to the NUMERIC(10, 6) scale.
    cost = compute_cost("claude-opus-4-7", 7, 3)
    # 7 * 5 / 1e6 + 3 * 25 / 1e6 = 0.000035 + 0.000075 = 0.000110
    assert cost == Decimal("0.000110")
    assert cost.as_tuple().exponent == -6


@pytest.mark.parametrize("model", list(PRICING_TABLE.keys()))
def test_pricing_table_rates_are_decimal(model: str):
    rates = PRICING_TABLE[model]
    assert isinstance(rates["input_per_mtok"], Decimal)
    assert isinstance(rates["output_per_mtok"], Decimal)
    assert rates["input_per_mtok"] >= 0
    assert rates["output_per_mtok"] >= 0
