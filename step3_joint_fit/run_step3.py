from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from pipeline.utils.config import copy_yaml, dump_yaml, load_yaml, timestamp_tag
from pipeline.utils.io import load_photometry_dataset, preprocess_dataset, save_results_bundle
from pipeline.utils.paths import build_step3_run_dir, ensure_dir, get_planet_name
from pipeline.utils.mcmc import _configured_parameter
from pipeline.utils.models import (
    default_initial_guess,
    detec_parameter_count,
    detrending_parameter_names,
    transit_model,
)
from pipeline.utils.plotting import plot_binned_fit, plot_corner, plot_corrected_flux, plot_rednoise, plot_walkers
from pipeline.utils.evidence import save_evidence
try:
    from step3_joint_fit.joint_mcmc import joint_split_theta, mjd_to_observation_hours, reconstruct_configured_joint, run_joint_mcmc, transit_config_for_eclipse
    from step3_joint_fit.joint_plotting import plot_combined_joint_fit, plot_stacked_joint_fit
    from step3_joint_fit.submission import export_joint_submission_components, prepare_joint_posterior
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from joint_mcmc import joint_split_theta, mjd_to_observation_hours, reconstruct_configured_joint, run_joint_mcmc, transit_config_for_eclipse
    from joint_plotting import plot_combined_joint_fit, plot_stacked_joint_fit
    from submission import export_joint_submission_components, prepare_joint_posterior


def resolve_eccentricity_prior(transit: dict, default_eccentricity: float, eclipse_number) -> dict:
    """Validate fixed, uniform, or bounded Gaussian eccentricity priors."""
    prior = dict(transit.get("ecc_prior", {}))
    initial = float(
        prior.get("initial", transit.get("ecc_initial", default_eccentricity))
    )
    if not prior:
        prior = {"type": "fixed", "value": initial}
    prior_type = str(prior.get("type", "fixed")).strip().lower()
    if prior_type == "normal":
        prior_type = "gaussian"
    if prior_type == "fixed":
        value = float(prior.get("value", initial))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"Eclipse {eclipse_number} fixed eccentricity must satisfy 0 <= ecc < 1")
        return {
            "type": "fixed", "initial": value, "fixed": True,
            "bounds": (value, value), "value": value,
        }
    if prior_type == "uniform":
        lower = float(prior["lower"])
        upper = float(prior["upper"])
        resolved = {"type": "uniform", "lower": lower, "upper": upper}
    elif prior_type == "gaussian":
        mean = float(prior["mean"])
        sigma = float(prior["sigma"])
        lower = float(prior.get("lower", 0.0))
        upper = float(prior.get("upper", 0.999999))
        if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                f"Eclipse {eclipse_number} Gaussian eccentricity prior requires finite mean and sigma > 0"
            )
        resolved = {
            "type": "gaussian", "mean": mean, "sigma": sigma,
            "lower": lower, "upper": upper,
        }
    else:
        raise ValueError(
            f"Eclipse {eclipse_number} ecc_prior.type must be fixed, uniform, or gaussian"
        )
    if not 0.0 <= lower < upper < 1.0:
        raise ValueError(
            f"Eclipse {eclipse_number} eccentricity bounds must satisfy 0 <= lower < upper < 1"
        )
    if not lower <= initial <= upper:
        raise ValueError(f"Eclipse {eclipse_number} ecc_initial is outside its prior bounds")
    return {
        **resolved, "initial": initial, "fixed": False, "bounds": (lower, upper),
    }


def resolve_eclipse_specs(config: dict) -> list[dict]:
    named_parameters = isinstance(config.get("parameters"), dict)
    specs = []
    for eclipse_cfg in config["eclipses"]:
        eclipse_number = eclipse_cfg["eclipse_number"]
        model_type = eclipse_cfg["detrending"]["model_type"]
        requires_centroids = "_2nd_order_centroid" in model_type
        data = load_photometry_dataset(
            eclipse_cfg["data"]["path"], require_centroids=requires_centroids
        )
        preprocess = eclipse_cfg.get("preprocessing", config.get("preprocessing", {}))
        data = preprocess_dataset(
            data,
            normalize=preprocess.get("normalize", "median"),
            sigma_clip=preprocess.get("sigma_clip", 4.0),
        )
        parameter_names = detrending_parameter_names(model_type)
        defaults = default_initial_guess(model_type)
        if named_parameters:
            missing = [
                name for name in parameter_names
                if name not in eclipse_cfg["detrending"]
            ]
            if missing:
                raise ValueError(
                    f"Eclipse {eclipse_number} detrending model '{model_type}' is "
                    f"missing parameter specification(s): {', '.join(missing)}"
                )
            resolved_detrending = [
                _configured_parameter(
                    name, eclipse_cfg["detrending"][name], default
                )
                for name, default in zip(parameter_names, defaults)
            ]
            guess = [parameter["initial"] for parameter in resolved_detrending]
        else:
            guess = eclipse_cfg["detrending"].get("initial_guess")
            if guess is None:
                guess = defaults
        expected_parameter_count = detec_parameter_count(model_type)
        if len(guess) != expected_parameter_count:
            raise ValueError(
                f"Eclipse {eclipse_number} detrending model '{model_type}' "
                f"expects {expected_parameter_count} initial values; got {len(guess)}"
            )
        transit = eclipse_cfg["transit"]
        if named_parameters:
            if "t_secondary_mjd" not in transit:
                raise ValueError(
                    f"Eclipse {eclipse_number} must define transit.t_secondary_mjd "
                    "using a fixed, uniform, or gaussian parameter specification"
                )
            timing = _configured_parameter(
                "t_secondary_mjd",
                transit["t_secondary_mjd"],
                float(data["time_mjd"][len(data["time_mjd"]) // 2]),
            )
            t0_mjd = float(timing["initial"])
            fix_mid_eclipse_time = not timing["sampled"]
            t0_prior = None
            if timing["sampled"]:
                t0_bounds = (float(timing["lower"]), float(timing["upper"]))
            else:
                t0_bounds = (t0_mjd, t0_mjd)
            shared_ecc = config.get("parameters", {}).get("eclipse", {}).get(
                "ecc", {"type": "fixed", "value": 0.0}
            )
            default_ecc = float(
                shared_ecc.get("value", shared_ecc.get("initial", 0.0))
            )
            eccentricity = resolve_eccentricity_prior(
                transit, default_ecc, eclipse_number
            )
            local_parameters = {
                "eclipse": {"t_secondary_mjd": transit["t_secondary_mjd"]},
                "detrending": {
                    name: eclipse_cfg["detrending"][name]
                    for name in parameter_names
                },
            }
            if "ecc_prior" in transit:
                local_parameters["eclipse"]["ecc"] = transit["ecc_prior"]
            if isinstance(eclipse_cfg.get("noise"), dict):
                local_parameters["noise"] = eclipse_cfg["noise"]
        else:
            t0_mjd = float(transit["expected_midtime_mjd"])
            fix_mid_eclipse_time = bool(
                transit.get(
                    "fix_mid_eclipse_time",
                    config.get("fit", {}).get("fix_mid_eclipse_time", False),
                )
            )
            t0_prior = transit.get("expected_midtime_mjd_prior")
            t0_bounds = (
                (float(t0_prior["lower"]), float(t0_prior["upper"]))
                if t0_prior is not None else (t0_mjd, t0_mjd)
            )
            eccentricity = resolve_eccentricity_prior(
                transit,
                config["transit"].get("fixed", {}).get("ecc", 0.0),
                eclipse_number,
            )
            local_parameters = eclipse_cfg.get("parameters", {})
        if not named_parameters and t0_prior is None and not fix_mid_eclipse_time:
            raise ValueError(
                f"Eclipse {eclipse_number} must define "
                "expected_midtime_mjd_prior when its midtime is sampled"
            )
        if (not named_parameters and t0_prior is not None
                and t0_prior.get("type", "uniform") != "uniform"):
            raise ValueError("A sampled joint-fit midtime prior must be uniform")
        if (t0_prior is not None
                and not float(t0_prior["lower"]) <= t0_mjd <= float(t0_prior["upper"])):
            raise ValueError(f"Eclipse {eclipse_number} expected_midtime_mjd is outside its prior")
        sigma_initial = eclipse_cfg.get("fit", {}).get(
            "sigma_initial", config.get("fit", {}).get("sigma_initial")
        )
        if sigma_initial is None:
            sigma_initial = float(np.nanmedian(data["flux_err"]))
        if not np.isfinite(sigma_initial) or float(sigma_initial) <= 0.0:
            raise ValueError(f"Eclipse {eclipse_number} sigma_initial must be positive")
        specs.append(
            {
                "eclipse_number": str(eclipse_number),
                "data": data,
                "data_path": eclipse_cfg["data"]["path"],
                "t0_mjd_initial": t0_mjd,
                "t0_mjd_bounds": t0_bounds,
                "fix_mid_eclipse_time": fix_mid_eclipse_time,
                "ecc_initial": eccentricity["initial"],
                "ecc_bounds": eccentricity["bounds"],
                "ecc_prior": {
                    key: value for key, value in eccentricity.items()
                    if key not in {"initial", "fixed", "bounds"}
                },
                "fix_eccentricity": eccentricity["fixed"],
                "detrending": eclipse_cfg["detrending"],
                "detrending_prior_bounds": eclipse_cfg["detrending"].get("prior_bounds"),
                "detrending_initial_guess": list(guess),
                "detec_param_count": expected_parameter_count,
                "sigma_initial": float(sigma_initial),
                "sigma_bounds": eclipse_cfg.get("fit", {}).get(
                    "sigma_bounds",
                    eclipse_cfg.get("fit", {}).get("evidence", {}).get("sigma_bounds"),
                ),
                "parameters": local_parameters,
            }
        )
    return specs


def main():
    parser = argparse.ArgumentParser(description="Run a joint eclipse fit.")
    parser.add_argument("--config", required=True, help="Path to the joint-fit YAML file.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    planet_name = get_planet_name(config)
    output_root = Path(config["output"]["root_dir"])
    run_dir = build_step3_run_dir(output_root, planet_name, timestamp_tag())
    ensure_dir(run_dir)
    copy_yaml(args.config, run_dir, "config_input.yaml")

    eclipse_specs = resolve_eclipse_specs(config)
    resolved_config = dict(config)
    dump_yaml(resolved_config, run_dir / "config_resolved.yaml")

    datasets = [spec["data"] for spec in eclipse_specs]
    fit_result = run_joint_mcmc(datasets, eclipse_specs, resolved_config, run_dir)

    np.save(run_dir / "chain.npy", fit_result["chain"])
    np.save(run_dir / "lnprobchain.npy", fit_result["lnprobchain"])
    np.save(run_dir / "final_state.npy", fit_result["final_state"])
    np.save(run_dir / "best_params.npy", fit_result["best"])
    np.save(run_dir / "posterior_samples.npy", fit_result["samples"])
    prepare_joint_posterior(resolved_config, fit_result, run_dir)

    summary = {
        "sampler": fit_result["sampler_name"],
        "best": fit_result["best"].tolist(),
        "summary": fit_result["summary"].tolist(),
        "labels": fit_result["labels"],
        "discard": fit_result["discard"],
        "evidence": fit_result["evidence"],
    }
    if fit_result["sampler_name"] == "emcee":
        summary.update(
            nwalkers=int(config["fit"]["nwalkers"]),
            nsteps=int(config["fit"]["nsteps"]),
        )
    if fit_result["evidence"] is not None:
        save_evidence(run_dir, fit_result["evidence"])
        print(
            f"Bayesian evidence: logZ = {fit_result['evidence']['logz']:.3f} "
            f"+/- {fit_result['evidence']['logz_err']:.3f}"
        )
    save_results_bundle(run_dir, summary, resolved_config)

    best = fit_result["best"]
    if config.get("parameters") is not None:
        fp, parts, _shared_values, _system = reconstruct_configured_joint(
            best, config, eclipse_specs, datasets
        )
    else:
        fp, parts = joint_split_theta(best, eclipse_specs)
    if "fp" in fit_result["labels"]:
        fp_summary = fit_result["summary"][fit_result["labels"].index("fp")]
    else:
        fp_summary = np.asarray([fp, 0.0, 0.0])
    for spec, dataset in zip(eclipse_specs, datasets):
        part = parts[spec["eclipse_number"]]
        t0_hours = mjd_to_observation_hours(part["t0_mjd"], dataset["time_mjd"][0])
        expected_t0_hours = mjd_to_observation_hours(spec["t0_mjd_initial"], dataset["time_mjd"][0])
        theta = np.concatenate(([t0_hours, fp], part["detec"], [part["sigF"]]))
        eclipse_transit = part.get("transit_config")
        if eclipse_transit is None:
            eclipse_transit = transit_config_for_eclipse(config, part["ecc"])
        eclipse_transit["time_ref_mjd"] = float(dataset["time_mjd"][0])
        astro = transit_model(dataset["time_hours"], theta[0], theta[1], eclipse_transit)
        from pipeline.utils.models import signal

        model = signal(dataset["time_hours"], dataset["centroid_x"], dataset["centroid_y"], theta, spec["detrending"]["model_type"], eclipse_transit, spec["detrending"])
        detec_model = model / astro
        residuals = dataset["flux"] / detec_model - astro

        eclipse_dir = ensure_dir(run_dir / f"eclipse{spec['eclipse_number']}")
        if config.get("plots", {}).get("corrected_flux", True):
            plot_corrected_flux(
                dataset["time_hours"],
                dataset["flux"],
                dataset["flux_err"],
                detec_model,
                astro,
                residuals,
                theta[0],
                f"eclipse{spec['eclipse_number']}",
                eclipse_dir / "corrected_flux.png",
                expected_t_s=expected_t0_hours,
            )
        if config.get("plots", {}).get("rednoise", True):
            plot_rednoise(residuals, eclipse_dir / "rednoise.png")
        if config.get("plots", {}).get("binned_fit", True):
            plot_binned_fit(
                dataset["time_hours"],
                dataset["flux"],
                dataset["flux_err"],
                detec_model,
                astro,
                eclipse_dir / "binned_fit.png",
                bins=int(config.get("plots", {}).get("n_bins", 100)),
                expected_t_s=expected_t0_hours,
                fitted_t_s=t0_hours,
                eclipse_depth_summary=fp_summary,
            )

    if (fit_result["sampler_name"] == "emcee"
            and config.get("plots", {}).get("walker", True)):
        plot_walkers(fit_result["chain"], fit_result["labels"], run_dir / "walkers.png")
    if config.get("plots", {}).get("corner", True):
        plot_corner(fit_result["samples"], fit_result["labels"], run_dir / "corner.png")
    if config.get("plots", {}).get("combined_binned_fit", True):
        plot_combined_joint_fit(
            datasets, eclipse_specs, config, fit_result, run_dir / "combined_binned_fit.png"
        )
    if config.get("plots", {}).get("stacked_binned_fit", True):
        plot_stacked_joint_fit(
            datasets, eclipse_specs, config, fit_result, run_dir / "stacked_binned_fit.png"
        )
    if config.get("submission", {}).get("enabled", False):
        export_joint_submission_components(
            datasets, eclipse_specs, config, fit_result, run_dir
        )


if __name__ == "__main__":
    main()
