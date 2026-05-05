"""Logger accessor.

Use this everywhere instead of `logging.getLogger(...)` directly so any future
formatting/transport changes happen in one place.
"""
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Conventionally `__name__` from the calling module.

    Returns:
        A configured logger inheriting handlers from the root logger.
    """
    return logging.getLogger(name)
