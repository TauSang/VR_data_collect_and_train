"""Re-export v9 ACT-Chunk implementation for 20260426 VR4 filtering checkpoints."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_V9_DIR = Path(__file__).resolve().parent.parent / "20260425_train_v9"
if str(_V9_DIR) not in sys.path:
    sys.path.insert(0, str(_V9_DIR))

_V9_TAC = _V9_DIR / "train_act_chunk.py"
_spec = importlib.util.spec_from_file_location("_v9_train_act_chunk", _V9_TAC)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

globals().update({name: value for name, value in vars(_mod).items() if not name.startswith("__")})

if __name__ == "__main__":
    _mod.main()
