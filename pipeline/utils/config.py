from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def timestamp_tag(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def copy_yaml(src: str | Path, dst_dir: str | Path, dst_name: str = "config_input.yaml") -> Path:
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / dst_name
    shutil.copy2(src, dst)
    return dst


def copy_tree_file(src: str | Path, dst: str | Path) -> Path:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def merge_dict(base: dict, updates: dict) -> dict:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out

