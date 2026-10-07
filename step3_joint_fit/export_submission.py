from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.utils.config import load_yaml
from step3_joint_fit.run_step3 import resolve_eclipse_specs
from step3_joint_fit.submission import export_joint_submission_components


def _load_saved_fit(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        config_path = run_dir / "config_input.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No config_resolved.yaml or config_input.yaml in {run_dir}"
        )
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best_params.npy"
    if not summary_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(
            f"The joint run must contain summary.json and best_params.npy: {run_dir}"
        )
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    posterior_path = run_dir / "posterior_samples.npy"
    if posterior_path.is_file():
        samples = np.asarray(np.load(posterior_path), dtype=float)
    elif str(summary.get("sampler", "")).lower() == "emcee":
        chain_path = run_dir / "chain.npy"
        if not chain_path.is_file():
            raise FileNotFoundError(f"Missing chain.npy in {run_dir}")
        chain = np.asarray(np.load(chain_path), dtype=float)
        samples = chain.reshape(-1, chain.shape[-1])
    else:
        raise FileNotFoundError(
            "This older dynesty run has no posterior_samples.npy. Its saved "
            "chain contains unequal-weight nested samples, so a valid equal-weight "
            "posterior cannot be reconstructed from the saved files."
        )
    labels = [str(label) for label in summary["labels"]]
    if samples.ndim != 2 or samples.shape[1] != len(labels):
        raise ValueError(
            f"Saved posterior shape {samples.shape} does not match {len(labels)} labels"
        )
    fit_result = {
        "samples": samples,
        "labels": labels,
        "best": np.asarray(np.load(best_path), dtype=float),
        "sampler_name": str(summary.get("sampler", "unknown")),
    }
    return run_dir, load_yaml(config_path), fit_result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create one target's Data Challenge submission components from an "
            "existing joint-fit run without rerunning the sampler."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Existing joint-fit run directory")
    parser.add_argument("--form", required=True, help="Completed challenge form JSON")
    parser.add_argument(
        "--target", choices=("LHS1140b", "GJ3929b"),
        help="Challenge target; defaults to the target saved in the run config",
    )
    args = parser.parse_args()

    run_dir, config, fit_result = _load_saved_fit(args.run_dir)
    submission = dict(config.get("submission", {}))
    submission["enabled"] = True
    submission["form_path"] = str(Path(args.form).expanduser().resolve())
    if args.target is not None:
        submission["target"] = args.target
    config["submission"] = submission

    eclipse_specs = resolve_eclipse_specs(config)
    datasets = [spec["data"] for spec in eclipse_specs]
    export_joint_submission_components(
        datasets, eclipse_specs, config, fit_result, run_dir
    )


if __name__ == "__main__":
    main()
