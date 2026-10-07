from __future__ import annotations

from pathlib import Path


def get_planet_name(config: dict) -> str:
    return str(config.get("planet_name") or config.get("target_name") or config["target"])


def build_step1_root(output_root: str | Path, planet_name: str, eclipse_number: int | str) -> Path:
    return Path(output_root) / planet_name / "step1" / f"eclipse{eclipse_number}"


def build_step2_run_dir(output_root: str | Path, run_label: str, planet_name: str, eclipse_number: int | str, timestamp: str) -> Path:
    return Path(output_root) / f"individual_fit_{run_label}_{timestamp}"


def build_step3_run_dir(output_root: str | Path, planet_name: str, timestamp: str) -> Path:
    return Path(output_root) / f"joint_fit_{timestamp}"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

