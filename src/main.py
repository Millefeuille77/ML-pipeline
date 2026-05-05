"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from config.logging_config import configure_logging
from config.settings import get_settings
from src.api.middleware.auth import ApiKeyAuthMiddleware
from src.api.middleware.error_handler import ErrorHandlerMiddleware
from src.api.middleware.rate_limiter import RateLimitMiddleware
from src.api.routes import analytics, forecast, health
from src.database.connection import get_engine, shutdown_engine
from src.utils.logger import get_logger

API_PREFIX = "/api/v1"

_settings = get_settings()
configure_logging(_settings.app_log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: warm up DB, yield, then dispose."""
    try:
        get_engine()
        logger.info("app_startup", extra={"version": _settings.app_version})
    except Exception:  # noqa: BLE001 — startup must not crash containers
        logger.exception("app_startup_db_warmup_failed")
    try:
        yield
    finally:
        shutdown_engine()
        logger.info("app_shutdown")


def create_app() -> FastAPI:
    """Assemble the FastAPI app with middleware and routes mounted."""
    application = FastAPI(
        title="FMCG Demand Forecasting & Product Intelligence Platform",
        description=(
            "REST API for weekly SKU demand forecasting, product clustering, "
            "anomaly detection, and analytics over a real FMCG distribution dataset."
        ),
        version=_settings.app_version,
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    # Middleware order: outermost first. Starlette executes them in the
    # reverse of registration, so register error_handler last so it wraps.
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(ApiKeyAuthMiddleware)
    application.add_middleware(ErrorHandlerMiddleware)

    application.include_router(health.router, prefix=API_PREFIX)
    application.include_router(forecast.router, prefix=API_PREFIX)
    application.include_router(analytics.router, prefix=API_PREFIX)
    return application


app = create_app()
