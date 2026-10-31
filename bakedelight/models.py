"""The de-lighter models that ship with the app: models/<name>.onnx, chosen by name.

DELIGHTER_MODEL (a file path) overrides the choice, for testing a freshly trained model.
"""

from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT = "overcast"


def names() -> list[str]:
    """The bundled models, the default first."""
    found = sorted(path.stem for path in MODELS_DIR.glob("*.onnx"))
    return sorted(found, key=lambda name: name != DEFAULT) or [DEFAULT]


def resolve(name: str = "") -> Path:
    """The model file for a name; an existing file path (older projects stored one) is used as it is."""
    override = os.environ.get("DELIGHTER_MODEL", "")
    for candidate in (override, name):
        if candidate and Path(candidate).suffix.lower() == ".onnx" and Path(candidate).is_file():
            return Path(candidate)
    for candidate in (name, DEFAULT, *names()):
        path = MODELS_DIR / f"{candidate}.onnx"
        if candidate and path.is_file():
            return path
    raise FileNotFoundError(f"no de-lighter model in {MODELS_DIR}")
