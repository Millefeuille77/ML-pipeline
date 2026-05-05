"""File-based model registry under `config.settings.MODEL_DIR`.

Versioned filenames: `{name}_v{N}_{YYYYMMDD-HHMMSS}.joblib` with a
sidecar JSON file capturing metadata.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import joblib

from config.settings import get_settings
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


def save_model(
    estimator: Any,
    name: str,
    metrics: dict[str, float],
    features: list[str],
    *,
    training_rows: int = 0,
    category: str | None = None,
) -> ModelInfo:
    """Persist `estimator` under a versioned filename inside MODEL_DIR.

    Args:
        estimator: Any joblib-serializable object (estimator, dict, etc.).
        name: Logical model name (alnum/underscore/dash only).
        metrics: Evaluation metrics dict.
        features: Ordered feature names used at training time.
        training_rows: Number of training rows.
        category: Optional category this model is specialized for.

    Returns:
        ModelInfo describing the saved artifact.
    """
    _validate_name(name)
    version = _next_version(name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    artifact_path = _models_dir() / f"{name}_v{version}_{timestamp}.joblib"
    metadata_path = artifact_path.with_suffix(".json")
    joblib.dump(estimator, artifact_path)
    metadata = {
        "name": name,
        "version": version,
        "metrics": metrics,
        "features": features,
        "training_rows": training_rows,
        "category": category,
        "created_at": timestamp,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "model_saved",
        extra={"name": name, "version": version, "path": str(artifact_path)},
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
