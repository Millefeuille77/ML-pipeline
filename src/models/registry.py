"""File-based model registry under `config.settings.MODEL_DIR`.

Versioned filenames: `{name}_v{N}_{YYYYMMDD-HHMMSS}.joblib` with a
sidecar JSON file capturing metadata.

Phase D extensions: compare_to_incumbent, record_lineage.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import get_settings
from src.database.connection import session_scope
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<name>[A-Za-z0-9_\-]+)_v(?P<version>\d+)_(?P<timestamp>\d{8}-\d{6})\.joblib$"
)
_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass(frozen=True)
class ModelInfo:
    """Metadata describing a registered model artifact."""

    name: str
    version: int
    path: Path
    metadata_path: Path
    metrics: dict[str, float]
    features: list[str]
    training_rows: int
    category: str | None
    created_at: str


def _models_dir() -> Path:
    """Return the resolved model directory, creating it if missing."""
    directory = get_settings().resolved_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_name(name: str) -> str:
    """Reject names containing path separators or unsafe characters."""
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid model name: {name!r}")
    return name


def _next_version(name: str) -> int:
    """Return the next integer version for a model with this name."""
    existing = list_models(name)
    if not existing:
        return 1
    return max(info.version for info in existing) + 1


def _compute_feature_baselines(frame: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """Return per-feature {mean, std} dict for drift detection.

    WHY: snapshot at training time avoids reloading full CSV during live drift checks.
    """
    if frame is None or frame.empty:
        return {}
    cols = frame.select_dtypes(include=[np.number]).columns
    return {c: {"mean": float(frame[c].mean()), "std": float(frame[c].std() or 0.0)} for c in cols}


def save_model(
    estimator: Any,
    name: str,
    metrics: dict[str, float],
    features: list[str],
    *,
    training_rows: int = 0,
    category: str | None = None,
    feature_frame: pd.DataFrame | None = None,
) -> ModelInfo:
    """Persist `estimator` under a versioned filename inside MODEL_DIR.

    Args:
        estimator: Any joblib-serializable object (estimator, dict, etc.).
        name: Logical model name (alnum/underscore/dash only).
        metrics: Evaluation metrics dict.
        features: Ordered feature names used at training time.
        training_rows: Number of training rows.
        category: Optional category this model is specialized for.
        feature_frame: Training feature matrix used to compute drift baselines.

    Returns:
        ModelInfo describing the saved artifact.
    """
    _validate_name(name)
    version = _next_version(name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    artifact_path = _models_dir() / f"{name}_v{version}_{timestamp}.joblib"
    metadata_path = artifact_path.with_suffix(".json")
    joblib.dump(estimator, artifact_path)
    feature_baselines = _compute_feature_baselines(feature_frame)  # WHY: persisted for Phase D drift checks
    metadata = {
        "name": name,
        "version": version,
        "metrics": metrics,
        "features": features,
        "training_rows": training_rows,
        "category": category,
        "created_at": timestamp,
        "feature_baselines": feature_baselines,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # WHY: "name" is a reserved LogRecord field; use model_name to avoid KeyError
    logger.info(
        "model_saved",
        extra={"model_name": name, "version": version, "path": str(artifact_path)},
    )
    return ModelInfo(
        name=name,
        version=version,
        path=artifact_path,
        metadata_path=metadata_path,
        metrics=metrics,
        features=features,
        training_rows=training_rows,
        category=category,
        created_at=timestamp,
    )


def list_models(name: str | None = None) -> list[ModelInfo]:
    """Enumerate registered models, optionally filtered by `name`.

    Args:
        name: If provided, only return models with this logical name.

    Returns:
        List of ModelInfo sorted by version ascending.
    """
    directory = _models_dir()
    results: list[ModelInfo] = []
    for path in sorted(directory.glob("*.joblib")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue
        if name is not None and match.group("name") != name:
            continue
        metadata_path = path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("model_metadata_corrupt", extra={"path": str(metadata_path)})
        results.append(
            ModelInfo(
                name=match.group("name"),
                version=int(match.group("version")),
                path=path,
                metadata_path=metadata_path,
                metrics=metadata.get("metrics", {}),
                features=metadata.get("features", []),
                training_rows=int(metadata.get("training_rows", 0)),
                category=metadata.get("category"),
                created_at=metadata.get("created_at", match.group("timestamp")),
            )
        )
    results.sort(key=lambda info: info.version)
    return results


def get_model(name: str, version: str | int = "latest") -> tuple[Any, ModelInfo]:
    """Load a registered model and its metadata.

    Args:
        name: Logical model name.
        version: Integer version, the string of an integer, or "latest".

    Returns:
        Tuple of (deserialized estimator, ModelInfo).

    Raises:
        FileNotFoundError: If no matching artifact exists.
    """
    _validate_name(name)
    candidates = list_models(name)
    if not candidates:
        raise FileNotFoundError(f"no model registered under name {name!r}")
    if version == "latest":
        info = candidates[-1]
    else:
        target = int(version)
        matches = [item for item in candidates if item.version == target]
        if not matches:
            raise FileNotFoundError(f"model {name!r} version {target} not found")
        info = matches[0]
    estimator = joblib.load(info.path)
    return estimator, info


# ── Phase D: MLOps extensions ──────────────────────────────────────────────

_INCUMBENT_TOLERANCE: Final[float] = 1.05  # challenger MAPE must be ≤ incumbent * 1.05


def compare_to_incumbent(name: str, candidate_metrics: dict[str, float]) -> str:
    """Return 'promoted', 'rejected', or 'no_incumbent' for a challenger model.

    Args:
        name: Logical model name.
        candidate_metrics: Must contain "mape".

    Returns:
        Decision string; rejected when candidate MAPE > incumbent * 1.05.
    """
    try:
        _estimator, incumbent_info = get_model(name, "latest")
    except FileNotFoundError:
        return "no_incumbent"
    incumbent_mape = incumbent_info.metrics.get("mape", float("inf"))
    candidate_mape = candidate_metrics.get("mape", float("inf"))
    # WHY: 5% slack lets minor noise not trigger spurious promotions
    if candidate_mape > incumbent_mape * _INCUMBENT_TOLERANCE:
        logger.info("model_rejected", extra={"model_name": name, "candidate_mape": candidate_mape})
        return "rejected"
    return "promoted"


def record_lineage(
    run_id: str,
    category: str,
    model_version: str,
    data_hash: str,
    training_rows: int,
    train_metrics: dict[str, float],
    incumbent_version: str | None,
    incumbent_mape: float | None,
    promotion_decision: str,
    drift_warnings: list[dict],
    notes: str = "",
) -> None:
    """Insert one audit row into model_lineage regardless of promotion decision."""
    sql = text(
        "INSERT INTO model_lineage "
        "(run_id, started_at, completed_at, category, model_version, data_hash, "
        "training_rows, train_mape, train_rmse, train_mae, train_r2, "
        "incumbent_version, incumbent_mape, promotion_decision, drift_warnings, notes) "
        "VALUES (:run_id, NOW(), NOW(), :category, :model_version, :data_hash, "
        ":training_rows, :train_mape, :train_rmse, :train_mae, :train_r2, "
        ":incumbent_version, :incumbent_mape, :promotion_decision, "
        "CAST(:drift_warnings AS JSONB), :notes)"
    )
    # WHY: NUMERIC(10,6) caps at 9999.999999; runaway MAPE from near-zero targets would overflow
    def _safe_metric(val: float | None) -> float | None:
        if val is None:
            return None
        v = float(val)
        return round(min(abs(v), 9999.0) * (1.0 if v >= 0 else -1.0), 6)

    params = {
        "run_id": run_id,
        "category": category,
        "model_version": model_version,
        "data_hash": data_hash,
        "training_rows": training_rows,
        "train_mape": _safe_metric(train_metrics.get("mape")),
        "train_rmse": _safe_metric(train_metrics.get("rmse")),
        "train_mae": _safe_metric(train_metrics.get("mae")),
        "train_r2": _safe_metric(train_metrics.get("r2")),
        "incumbent_version": incumbent_version,
        "incumbent_mape": _safe_metric(incumbent_mape),
        "promotion_decision": promotion_decision,
        "drift_warnings": json.dumps(drift_warnings),
        "notes": notes,
    }
    with session_scope() as session:
        session.execute(sql, params)
    logger.info(
        "lineage_recorded",
        extra={"run_id": run_id, "category": category, "decision": promotion_decision},
    )
