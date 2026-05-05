"""Shared validators.

Centralizes the few literal sets and regex patterns that appear in both
Pydantic schemas and runtime checks. Keep this thin — Pydantic does the
heavy lifting at the API boundary.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Final

SKU_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{2}-\d{3}$")

VALID_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"Milk", "Yogurt", "ReadyMeal", "Juice", "SnackBar"}
)
VALID_CHANNELS: Final[frozenset[str]] = frozenset(
    {"Retail", "Discount", "E-commerce"}
)
VALID_REGIONS: Final[frozenset[str]] = frozenset(
    {"PL-Central", "PL-North", "PL-South"}
)
VALID_PACK_TYPES: Final[frozenset[str]] = frozenset(
    {"Multipack", "Single", "Carton"}
)
VALID_LIFECYCLE_STAGES: Final[frozenset[str]] = frozenset(
    {"Growth", "Mature", "Decline"}
)


def is_valid_sku(sku: str) -> bool:
    """Return True if `sku` matches the canonical `XX-NNN` form."""
    return bool(SKU_PATTERN.fullmatch(sku))


def validate_sku(sku: str) -> str:
    """Validate and normalize a SKU.

    Args:
        sku: Candidate SKU string.

    Returns:
        The SKU unchanged if valid.

    Raises:
        ValueError: If `sku` does not match `^[A-Z]{2}-\\d{3}$`.
    """
    if not is_valid_sku(sku):
        raise ValueError(f"Invalid SKU format: {sku!r}; expected pattern XX-NNN.")
    return sku


def validate_category(category: str) -> str:
    """Validate that `category` is one of the five known categories."""
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category {category!r}; expected one of {sorted(VALID_CATEGORIES)}."
        )
    return category


def validate_channel(channel: str) -> str:
    """Validate that `channel` is one of the three known channels."""
    if channel not in VALID_CHANNELS:
        raise ValueError(
            f"Invalid channel {channel!r}; expected one of {sorted(VALID_CHANNELS)}."
        )
    return channel


def validate_region(region: str) -> str:
    """Validate that `region` is one of the three known regions."""
    if region not in VALID_REGIONS:
        raise ValueError(
            f"Invalid region {region!r}; expected one of {sorted(VALID_REGIONS)}."
        )
    return region


def validate_date_range(start: date, end: date) -> tuple[date, date]:
    """Validate `start <= end`.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        The pair `(start, end)` unchanged if valid.

    Raises:
        ValueError: If `start > end`.
    """
    if start > end:
        raise ValueError(f"start date {start} is after end date {end}.")
    return start, end
