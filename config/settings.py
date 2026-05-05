"""Typed application settings loaded from `.env` via pydantic-settings.

Single source of truth for runtime configuration. Every other module reads
config from `get_settings()` rather than `os.environ` directly.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings.

    Attributes:
        db_host: PostgreSQL host.
        db_port: PostgreSQL port.
        db_name: PostgreSQL database name.
        db_user: PostgreSQL user.
        db_password: PostgreSQL password (SecretStr — never serialized in logs).
        api_host: Bind host for the HTTP server.
        api_port: Bind port for the HTTP server.
        api_key: Shared API key required in `X-API-Key` header.
        api_rate_limit_per_min: Max requests per minute per API key.
        model_forecast_horizon_weeks: Default horizon for forecast endpoints.
        model_n_estimators: Trees in the GradientBoostingRegressor.
        model_max_depth: Max tree depth for GBR.
        model_learning_rate: Learning rate for GBR.
        model_ridge_alpha: L2 regularization strength for the Ridge baseline.
        raw_data_dir: Directory holding raw CSV/parquet inputs.
        processed_data_dir: Directory for ETL-processed artifacts.
        model_dir: Directory for serialized model artifacts.
        app_version: Semver string surfaced via /health.
        app_log_level: Root logger level.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT", ge=1, le=65535)
    db_name: str = Field(default="fmcg_intelligence", alias="DB_NAME")
    db_user: str = Field(default="fmcg_user", alias="DB_USER")
    db_password: SecretStr = Field(default=SecretStr("change_me"), alias="DB_PASSWORD")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT", ge=1, le=65535)
    api_key: SecretStr = Field(default=SecretStr("dev_only_key"), alias="API_KEY")
    api_rate_limit_per_min: int = Field(default=100, alias="API_RATE_LIMIT_PER_MIN", ge=1)

    model_forecast_horizon_weeks: int = Field(default=4, alias="MODEL_FORECAST_HORIZON_WEEKS", ge=1, le=12)
    model_n_estimators: int = Field(default=200, alias="MODEL_N_ESTIMATORS", ge=10)
    model_max_depth: int = Field(default=5, alias="MODEL_MAX_DEPTH", ge=1)
    model_learning_rate: float = Field(default=0.05, alias="MODEL_LEARNING_RATE", gt=0.0, le=1.0)
    model_ridge_alpha: float = Field(default=1.0, alias="MODEL_RIDGE_ALPHA", gt=0.0)
    model_min_samples_leaf: int = Field(default=20, alias="MODEL_MIN_SAMPLES_LEAF", ge=1)

    raw_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw", alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed", alias="PROCESSED_DATA_DIR")
    model_dir: Path = Field(default=PROJECT_ROOT / "data" / "models", alias="MODEL_DIR")

    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")

    @property
    def database_url(self) -> str:
        """Return SQLAlchemy-compatible psycopg2 URL.

        Returns:
            URL of the form `postgresql+psycopg2://user:pass@host:port/db`.
        """
        password = self.db_password.get_secret_value()
        return (
            f"postgresql+psycopg2://{self.db_user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def resolved_raw_dir(self) -> Path:
        """Return raw data directory resolved against project root."""
        return self._resolve_path(self.raw_data_dir)

    def resolved_processed_dir(self) -> Path:
        """Return processed data directory resolved against project root."""
        return self._resolve_path(self.processed_data_dir)

    def resolved_model_dir(self) -> Path:
        """Return model directory resolved against project root."""
        return self._resolve_path(self.model_dir)

    @staticmethod
    def _resolve_path(candidate: Path) -> Path:
        """Resolve a path relative to PROJECT_ROOT when not absolute."""
        path = Path(candidate)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor for the singleton Settings instance.

    Returns:
        Process-wide Settings instance.
    """
    return Settings()
