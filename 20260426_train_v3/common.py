"""Re-export v9 common utilities for Scheme B checkpoints."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_V9_COMMON = Path(__file__).resolve().parent.parent / "20260425_train_v9" / "common.py"
_spec = importlib.util.spec_from_file_location("_v9_common", _V9_COMMON)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

globals().update({name: value for name, value in vars(_mod).items() if not name.startswith("__")})
