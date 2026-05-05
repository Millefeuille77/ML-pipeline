"""SQLAlchemy engine, session factory, and health probe."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_POOL_SIZE = 5
_DEFAULT_MAX_OVERFLOW = 10
_DEFAULT_POOL_RECYCLE_SECONDS = 1800

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_engine() -> Engine:
    """Create the engine with conservative pool defaults."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_size=_DEFAULT_POOL_SIZE,
        max_overflow=_DEFAULT_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=_DEFAULT_POOL_RECYCLE_SECONDS,
        future=True,
    )


def get_engine() -> Engine:
    """Return the lazily-initialized SQLAlchemy engine.

    Returns:
        Process-wide Engine instance.
    """
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _build_engine()
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
        logger.info("database_engine_initialized", extra={"db": get_settings().db_name})
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the configured session factory."""
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    Yields:
        A SQLAlchemy Session. Always closed in `finally`.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager: open a session, commit on success, rollback on error."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("database_transaction_rolled_back")
        raise
    finally:
        session.close()


def check_health() -> bool:
    """Run `SELECT 1` to verify database connectivity.

    Returns:
        True if the query returns 1; False otherwise.
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar()
            return result == 1
    except SQLAlchemyError:
        logger.exception("database_health_check_failed")
        return False


def shutdown_engine() -> None:
    """Dispose of the engine on app shutdown."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        logger.info("database_engine_disposed")
    _engine = None
    _SessionLocal = None
