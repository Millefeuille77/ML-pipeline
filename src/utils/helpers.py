"""Small shared helpers used by ETL, models, and routes."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel


def iso_week_start(reference: date) -> date:
    """Return the Monday (ISO week start) of the week containing `reference`.

    Args:
        reference: Any date.

    Returns:
        The Monday of that ISO week.
    """
    return reference - timedelta(days=reference.weekday())


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide `numerator / denominator`, returning `default` if denominator ~= 0.

    Args:
        numerator: Top of the fraction.
        denominator: Bottom of the fraction.
        default: Value returned when `abs(denominator) < 1e-12`.

    Returns:
        The quotient, or `default` for protected division.
    """
    if denominator is None or abs(float(denominator)) < 1e-12:
        return default
    return float(numerator) / float(denominator)


def df_to_pydantic_list(
    frame: pd.DataFrame, model: type[BaseModel]
) -> list[BaseModel]:
    """Convert a DataFrame to a list of validated Pydantic models.

    Args:
        frame: DataFrame whose columns superset the model's required fields.
        model: Pydantic model class.

    Returns:
        List of model instances; empty when the frame is empty.
    """
    if frame.empty:
        return []
    return [model.model_validate(record) for record in frame.to_dict(orient="records")]


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield successive `size`-sized chunks from `items`.

    Args:
        items: Source iterable.
        size: Chunk size; must be positive.

    Yields:
        Lists of up to `size` consecutive items.

    Raises:
        ValueError: If `size <= 0`.
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}.")
    buffer: list[Any] = []
    for item in items:
        buffer.append(item)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def coerce_week_start(value: Any) -> date:
    """Coerce a date-like value to the Monday of its ISO week.

    Args:
        value: A `date`, `datetime`, pandas Timestamp, or ISO string.

    Returns:
        Monday (ISO weekday 1) of the week containing `value`.

    Raises:
        ValueError: If `value` cannot be parsed.
    """
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse date value: {value!r}.")
    as_date = timestamp.date()
    return iso_week_start(as_date)
