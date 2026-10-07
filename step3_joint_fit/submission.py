from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import h5py
import numpy as np

from pipeline.utils.models import signal, transit_model

try:
    from step3_joint_fit.joint_mcmc import (
        joint_split_theta,
        mjd_to_observation_hours,
        reconstruct_configured_joint,
        transit_config_for_eclipse,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from joint_mcmc import (
        joint_split_theta,
        mjd_to_observation_hours,
        reconstruct_configured_joint,
        transit_config_for_eclipse,
    )


TARGETS = {
    "lhs1140b": {
        "stem": "LHS1140b",
        "display_name": "LHS 1140 b",
    },
    "gj3929b": {
        "stem": "GJ3929b",
        "display_name": "GJ 3929 b",
    },
}

STANDARD_PARAMETER_NAMES = {
    "fp": "depth_ecl",
    "t0_mjd": "t_tra",
    "period_days": "per",
    "rp": "rprs",
    "w": "omega",
}


def _canonical_target(value: str) -> dict:
    key = "".join(character for character in str(value).lower() if character.isalnum())
    if key not in TARGETS:
        raise ValueError(
            "submission.target must identify either LHS1140b or GJ3929b; "
            f"got {value!r}"
        )
    return TARGETS[key]


def _posterior_outputs(
    fit_result, fp_best, minimum_samples=10_000, maximum_samples=None,
    random_seed=42,
):
    samples = np.asarray(fit_result["samples"], dtype=float)
    labels = [str(label) for label in fit_result["labels"]]
    if samples.ndim != 2 or samples.shape[1] != len(labels):
        raise ValueError(
            "Joint posterior samples must have shape (n_samples, n_parameters); "
            f"got {samples.shape} for {len(labels)} labels"
        )
    if maximum_samples is not None:
        maximum_samples = int(maximum_samples)
        if maximum_samples < 1:
            raise ValueError("submission.maximum_posterior_samples must be positive")
        if samples.shape[0] > maximum_samples:
            indices = np.linspace(0, samples.shape[0] - 1, maximum_samples, dtype=int)
            samples = samples[indices]
    minimum_samples = int(minimum_samples)
    if minimum_samples < 1_000:
        raise ValueError("submission.minimum_posterior_samples must be at least 1,000")
    resampled = samples.shape[0] < minimum_samples
    source_sample_count = int(samples.shape[0])
    if resampled:
        warnings.warn(
            f"The fit produced {source_sample_count} equal-weight posterior draws. "
            f"Sampling those draws with replacement to create {minimum_samples} "
            "challenge rows; this does not increase the effective sample size.",
            UserWarning,
        )
        rng = np.random.default_rng(None if random_seed is None else int(random_seed))
        samples = samples[rng.choice(samples.shape[0], size=minimum_samples, replace=True)]

    output_labels = []
    columns = []
    for index, label in enumerate(labels):
        output_label = STANDARD_PARAMETER_NAMES.get(label, label)
        values = samples[:, index].copy()
        if label == "fp":
            values *= 1e6
        if output_label in output_labels:
            raise ValueError(f"Duplicate submission posterior key: {output_label}")
        output_labels.append(output_label)
        columns.append(values)

    if "depth_ecl" not in output_labels:
        output_labels.insert(0, "depth_ecl")
        columns.insert(0, np.full(samples.shape[0], float(fp_best) * 1e6))
    posterior = np.vstack(columns)
    if not np.all(np.isfinite(posterior)):
        raise ValueError("Submission posterior contains non-finite samples")
    return posterior, output_labels, {
        "source_equal_weight_samples": source_sample_count,
        "exported_samples": int(posterior.shape[1]),
        "resampled_with_replacement": resampled,
    }


def _best_eclipse_depth(config, fit_result):
    if "fp" in fit_result["labels"]:
        return float(fit_result["best"][fit_result["labels"].index("fp")])
    specification = config.get("parameters", {}).get("eclipse", {}).get("fp", {})
    for key in ("value", "initial", "mean"):
        if key in specification:
            return float(specification[key])
    return float(config.get("fit", {}).get("fp_initial", 0.0))


def _as_challenge_bjd(time):
    time = np.asarray(time, dtype=float)
    # The pipeline accepts either full BJD or BJD-2,400,000.5/BMJD arrays.
    return time + 2_400_000.5 if np.nanmedian(time) < 1_000_000 else time.copy()


def _photometry_outputs(datasets, eclipse_specs, config, fit_result):
    best = np.asarray(fit_result["best"], dtype=float)
    if config.get("parameters") is not None:
        fp, parts, _shared, _system = reconstruct_configured_joint(
            best, config, eclipse_specs, datasets
        )
    else:
        fp, parts = joint_split_theta(best, eclipse_specs)

    products = {
        key: [] for key in (
            "time", "raw_flux", "raw_flux_err", "astro_model", "noise_model",
            "full_model", "centroid_x", "centroid_y", "eclipse_number",
            "fitted_jitter",
        )
    }
    for spec, dataset in zip(eclipse_specs, datasets):
        part = parts[spec["eclipse_number"]]
        t_secondary_hours = mjd_to_observation_hours(
            part["t0_mjd"], dataset["time_mjd"][0]
        )
        theta = np.concatenate(
            ([t_secondary_hours, fp], part["detec"], [part["sigF"]])
        )
        transit_config = part.get("transit_config")
        if transit_config is None:
            transit_config = transit_config_for_eclipse(config, part["ecc"])
        transit_config["time_ref_mjd"] = float(dataset["time_mjd"][0])
        astro = transit_model(
            dataset["time_hours"], t_secondary_hours, fp, transit_config
        )
        full = signal(
            dataset["time_hours"], dataset["centroid_x"], dataset["centroid_y"],
            theta, spec["detrending"]["model_type"], transit_config,
            spec["detrending"],
        )
        noise = np.divide(
            full, astro, out=np.full_like(full, np.nan), where=astro != 0
        )
        n_points = len(dataset["time_mjd"])
        products["time"].append(_as_challenge_bjd(dataset["time_mjd"]))
        products["raw_flux"].append(np.asarray(dataset["flux"], dtype=float))
        products["raw_flux_err"].append(np.asarray(dataset["flux_err"], dtype=float))
        products["astro_model"].append(astro)
        products["noise_model"].append(noise)
        products["full_model"].append(full)
        products["centroid_x"].append(np.asarray(dataset["centroid_x"], dtype=float))
        products["centroid_y"].append(np.asarray(dataset["centroid_y"], dtype=float))
        products["eclipse_number"].append(
            np.full(n_points, int(spec["eclipse_number"]), dtype=int)
        )
        products["fitted_jitter"].append(
            np.full(n_points, float(part["sigF"]), dtype=float)
        )

    products = {key: np.concatenate(value) for key, value in products.items()}
    order = np.argsort(products["time"])
    products = {key: value[order] for key, value in products.items()}
    shapes = {value.shape for value in products.values()}
    if len(shapes) != 1 or not all(np.all(np.isfinite(value)) for value in products.values()):
        raise ValueError("Submission photometry arrays must be finite and have matching shapes")
    return products


def _standardized_form(form_path):
    path = Path(form_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Could not find completed challenge form: {path}")
    with path.open("r", encoding="utf-8") as stream:
        form = json.load(stream)
    if not isinstance(form, dict) or not form:
        raise ValueError("The challenge form must be a non-empty JSON object")
    standardized = {}
    for key in sorted(form, key=lambda item: int(item)):
        fields = dict(form[key])
        fields.setdefault("response", "")
        standardized[f"{int(key):02d}"] = fields
    return standardized


def _write_posterior(path, posterior, labels):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(labels)
        writer.writerows(posterior.T)


def _read_posterior(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        labels = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    posterior = np.asarray(rows, dtype=float).T
    if posterior.ndim != 2 or posterior.shape[0] != len(labels):
        raise ValueError(f"Malformed challenge posterior table: {path}")
    return posterior, labels


def _write_photometry(path, posterior, labels, photometry):
    with h5py.File(path, "w") as h5_file:
        sample_dataset = h5_file.create_dataset("posterior_samples", data=posterior)
        sample_dataset.attrs["parameter_keys"] = labels
        for key, values in photometry.items():
            h5_file.create_dataset(key, data=values)


def _validate_with_official_package(posterior, labels, photometry, form):
    try:
        import rocky_worlds_data_challenge as rw
    except ImportError:
        return {
            "official_package_available": False,
            "official_validation": "not run",
            "message": (
                "Install rocky-worlds-data-challenge in this environment to run "
                "the official component validators."
            ),
        }
    rw.Posterior(samples=posterior, parameter_keys=labels).validate()
    rw.Photometry(**photometry).validate()
    rw.Form(dictionary=form).validate()
    return {
        "official_package_available": True,
        "official_validation": "passed",
    }


def prepare_joint_posterior(config, fit_result, run_dir):
    """Always save equal-weight samples and the target's challenge TXT table."""
    submission = config.get("submission", {})
    target = _canonical_target(
        submission.get("target", config.get("target_name", config.get("planet_name")))
    )
    output_dir = Path(run_dir) / "submission_components"
    output_dir.mkdir(parents=True, exist_ok=True)
    posterior, labels, sampling = _posterior_outputs(
        fit_result,
        fp_best=_best_eclipse_depth(config, fit_result),
        minimum_samples=submission.get("minimum_posterior_samples", 10_000),
        maximum_samples=submission.get("maximum_posterior_samples"),
        random_seed=submission.get(
            "random_seed", config.get("fit", {}).get("random_seed", 42)
        ),
    )
    posterior_path = output_dir / f"posterior_{target['stem']}.txt"
    _write_posterior(posterior_path, posterior, labels)
    report = {
        "target": target["display_name"],
        "posterior_file": posterior_path.name,
        "posterior_parameters": labels,
        **sampling,
    }
    report_path = Path(run_dir) / "posterior_export.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Challenge posterior written automatically: {posterior_path}")
    return posterior_path, report


def export_joint_submission_components(
    datasets, eclipse_specs, config, fit_result, run_dir,
):
    """Export one target's three challenge components without altering its fit."""
    submission = config.get("submission", {})
    target = _canonical_target(
        submission.get("target", config.get("target_name", config.get("planet_name")))
    )
    form_path = submission.get("form_path")
    if not form_path:
        raise ValueError(
            "submission.form_path must point to a completed challenge JSON form"
        )
    output_dir = Path(run_dir) / "submission_components"
    output_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = output_dir / f"posterior_{target['stem']}.txt"
    if not posterior_path.is_file():
        posterior_path, _report = prepare_joint_posterior(
            config, fit_result, run_dir
        )
    posterior, labels = _read_posterior(posterior_path)
    photometry = _photometry_outputs(
        datasets, eclipse_specs, config, fit_result
    )
    form = _standardized_form(form_path)

    photometry_path = output_dir / f"lc_{target['stem']}.h5"
    form_output_path = output_dir / f"form_{target['stem']}.json"
    _write_photometry(photometry_path, posterior, labels, photometry)
    with form_output_path.open("w", encoding="utf-8") as stream:
        json.dump(form, stream, sort_keys=True, indent=2)
        stream.write("\n")

    validation = _validate_with_official_package(
        posterior, labels, photometry, form
    )
    report = {
        "target": target["display_name"],
        "posterior_samples": int(posterior.shape[1]),
        "posterior_parameters": labels,
        "photometry_points": int(len(photometry["time"])),
        "files": [
            posterior_path.name, photometry_path.name, form_output_path.name,
        ],
        **validation,
    }
    report_path = Path(run_dir) / "submission_export.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Challenge submission components written to: {output_dir}")
    print(f"Submission validation status: {validation['official_validation']}")
    return report
