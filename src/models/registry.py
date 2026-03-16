"""
Model registry: save and load versioned model artifacts with joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.utils.logger import get_logger
from src.utils.paths import ARTIFACTS_MODELS, ensure_dirs

logger = get_logger(__name__)


def save_artifact(obj: Any, name: str, subdir: Path | None = None) -> Path:
    """
    Persist an object (model, scaler, feature list, etc.) to artifacts/models/.

    Args:
        obj: Any picklable Python object.
        name: Artifact name (without extension).
        subdir: Optional subdirectory override.

    Returns:
        Path where the artifact was saved.
    """
    ensure_dirs()
    directory = subdir or ARTIFACTS_MODELS
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    joblib.dump(obj, path)
    logger.info("Artifact saved -> %s", path)
    return path


def load_artifact(name: str, subdir: Path | None = None) -> Any:
    """
    Load a persisted artifact by name.

    Args:
        name: Artifact name (without extension).
        subdir: Optional subdirectory override.

    Returns:
        The deserialized object.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    directory = subdir or ARTIFACTS_MODELS
    path = directory / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{name}' not found at {path}. " "Have you run the training pipeline?"
        )
    obj = joblib.load(path)
    logger.info("Artifact loaded <- %s", path)
    return obj


def list_artifacts(subdir: Path | None = None) -> list[str]:
    """Return names of all saved artifacts."""
    directory = subdir or ARTIFACTS_MODELS
    return [p.stem for p in sorted(directory.glob("*.joblib"))]
