from pathlib import Path
from typing import Union
import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: Union[str, Path], raw_path: Union[str, Path]) -> Path:
    cp = Path(config_path).resolve()
    rp = Path(raw_path)
    if rp.is_absolute():
        return rp
    return (cp.parent / rp).resolve()
