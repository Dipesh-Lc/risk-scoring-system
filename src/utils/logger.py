"""Logging setup for the project."""

import logging
import logging.config
import logging.handlers
from pathlib import Path

import yaml

from src.utils.paths import ARTIFACTS_LOGS, CONFIGS

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    global _configured
    if not _configured:
        _setup_logging()
        _configured = True
    return logging.getLogger(name)


def _setup_logging() -> None:
    """Load logging config from YAML, falling back to basic config."""
    ARTIFACTS_LOGS.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS / "logging.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Ensure log file directory exists
        for handler in config.get("handlers", {}).values():
            if "filename" in handler:
                Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
