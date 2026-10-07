from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.utils import config
from pipeline.utils.config import copy_yaml, dump_yaml, load_yaml, timestamp_tag
from pipeline.utils.io import (
    load_photometry_dataset,
    preprocess_dataset,
    save_corrected_photometry_h5,
    save_results_bundle,
)
from pipeline.utils.mcmc import (
    gp_config,
    predict_individual_gp,
    reconstruct_configured_individual,
    resolve_individual_eccentricity,
    run_individual_mcmc,
    split_individual_theta,
    transit_config_with_eccentricity,
)
from pipeline.utils.paths import build_step2_run_dir, ensure_dir, get_planet_name
from pipeline.utils.models import transit_model
from pipeline.utils.plotting import plot_binned_fit, plot_corner, plot_corrected_flux, plot_gaussian_process_fit, plot_rednoise, plot_walkers
from pipeline.utils.evidence import save_evidence


def build_resolved_config(config: dict, data_path: str) -> dict:
    resolved = dict(config)
    resolved.setdefault("data", {})["path"] = data_path
    resolved.setdefault("fit", {})
    resolved.setdefault("plots", {})
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Run an individual eclipse fit.")
    parser.add_argument("--config", required=True, help="Path to the individual-fit YAML file.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    planet_name = get_planet_name(config)
    eclipse_number = config["eclipse_number"]
    run_label = config["output"]["run_label"]
    output_root = Path(config["output"]["root_dir"])
    run_dir = build_step2_run_dir(output_root, config["output"]["run_label"], planet_name, eclipse_number, timestamp_tag())
    ensure_dir(run_dir)
    copy_yaml(args.config, run_dir, "config_input.yaml")

    model_type = config["detrending"]["model_type"]
    data = load_photometry_dataset(
        config["data"]["path"],
        require_centroids="_2nd_order_centroid" in model_type,
    )
    preprocess = config.get("preprocessing", {})
    time_coordinate = str(config.get("data", {}).get("time_coordinate", "mjd")).lower()
    if time_coordinate == "phase_hours":
        expected_phase_time = config.get("transit", {}).get("expected_midtime_hours")
        if expected_phase_time is None or not np.isclose(float(expected_phase_time), 0.0):
            raise ValueError(
                "phase_hours data require transit.expected_midtime_hours: 0.0"
            )
        if "expected_midtime_mjd" in config.get("transit", {}):
            raise ValueError(
                "Remove transit.expected_midtime_mjd when using phase_hours"
            )
        if isinstance(config.get("parameters"), dict):
            eclipse_parameters = config["parameters"].get("eclipse", {})
            if "t_secondary_phase_hours" not in eclipse_parameters:
                raise ValueError(
                    "phase_hours data require parameters.eclipse.t_secondary_phase_hours"
                )
            if "t_secondary_mjd" in eclipse_parameters:
                raise ValueError(
                    "Remove parameters.eclipse.t_secondary_mjd when using phase_hours; "
                    "use t_secondary_phase_hours instead"
                )
            if "t0_hours" not in eclipse_parameters:
                raise ValueError(
                    "phase_hours data require parameters.eclipse.t0_hours"
                )
            if "t0_mjd" in eclipse_parameters:
                raise ValueError(
                    "Remove parameters.eclipse.t0_mjd when using phase_hours; "
                    "use t0_hours instead"
                )
    phase_reference_mjd = 0.0 if time_coordinate == "phase_hours" else None
    data = preprocess_dataset(
        data,
        normalize=preprocess.get("normalize", "median"),
        sigma_clip=preprocess.get("sigma_clip", 4.0),
        time_coordinate=time_coordinate,
        phase_reference_mjd=phase_reference_mjd,
    )
    resolved_config = build_resolved_config(config, config["data"]["path"])
    dump_yaml(resolved_config, run_dir / "config_resolved.yaml")

    fit_result = run_individual_mcmc(data, resolved_config, run_dir)
    np.save(run_dir / "chain.npy", fit_result["chain"])
    np.save(run_dir / "lnprobchain.npy", fit_result["lnprobchain"])
    np.save(run_dir / "final_state.npy", fit_result["final_state"])
    np.save(run_dir / "best_params.npy", fit_result["best"])

    summary = {
        "sampler": fit_result["sampler_name"],
        "best": fit_result["best"].tolist(),
        "summary": fit_result["summary"].tolist(),
        "labels": fit_result["labels"],
        "discard": fit_result["discard"],
        "evidence": fit_result["evidence"]
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
    gaussian_process = gp_config(config)
    if config.get("parameters") is not None:
        theta, gp_parameters, fitted_transit_config, parameter_values, _system = (
            reconstruct_configured_individual(
                best, config, data["flux_err"], data["time_ref_mjd"]
            )
        )
        eccentricity_best = parameter_values["ecc"]
    else:
        fix_mid_eclipse_time = bool(config.get("fit", {}).get("fix_mid_eclipse_time", False))
        eccentricity = resolve_individual_eccentricity(config)
        theta, gp_parameters, eccentricity_best = split_individual_theta(
            best, config["detrending"]["model_type"], gp_config(config),
            fix_mid_eclipse_time=fix_mid_eclipse_time,
            fixed_mid_eclipse_time=config.get("fit", {}).get("dt_s_initial_hours"),
            fix_eccentricity=eccentricity["fixed"],
            fixed_eccentricity=eccentricity["initial"],
        )
        fitted_transit_config = transit_config_with_eccentricity(
            config["transit"], eccentricity_best
        )
        fitted_transit_config["time_ref_mjd"] = float(data["time_ref_mjd"])
    t_s = float(theta[0])
    fp_best = float(theta[1])
    if "fp" in fit_result["labels"]:
        fp_summary_index = fit_result["labels"].index("fp")
        fp_summary = fit_result["summary"][fp_summary_index]
    else:
        fp_summary = np.asarray([fp_best, 0.0, 0.0], dtype=float)
    fp_median, fp_upper, fp_lower = map(float, fp_summary)
    print("ts_best: ", t_s)
    if config.get("parameters") is not None:
        if time_coordinate == "phase_hours":
            print(
                "t_secondary_phase_hours_best: ",
                parameter_values["t_secondary_phase_hours"],
            )
        else:
            print("t_secondary_mjd_best: ", parameter_values["t_secondary_mjd"])
    print("fp_best: ", fp_best)
    print("fp_median: ", fp_median)
    print("ecc_best: ", eccentricity_best)
    if "ecc" in fit_result["labels"]:
        ecc_summary = fit_result["summary"][fit_result["labels"].index("ecc")]
        print(
            f"ecc_median: {ecc_summary[0]:.8g} "
            f"+{ecc_summary[1]:.8g} -{ecc_summary[2]:.8g}"
        )
    print(
        f"fp_median uncertainty: +{fp_upper:.8g} -{fp_lower:.8g} "
        f"({fp_median * 1e6:.1f} +{fp_upper * 1e6:.1f} "
        f"-{fp_lower * 1e6:.1f} ppm)"
    )

    # Use the posterior-median eclipse depth for every plotted model and
    # diagnostic so the curve matches the median quoted in its annotation.
    plot_theta = np.asarray(theta, dtype=float).copy()
    plot_theta[1] = fp_median
    from pipeline.utils.models import signal

    model = signal(data["time_hours"], data["centroid_x"], data["centroid_y"], plot_theta, config["detrending"]["model_type"], fitted_transit_config, config["detrending"])
    astro = transit_model(data["time_hours"], t_s, fp_median, fitted_transit_config)
    detec = model / astro
    save_corrected_data = config.get("output", {}).get("save_corrected_data_h5", False)
    if not isinstance(save_corrected_data, (bool, np.bool_)):
        raise ValueError("output.save_corrected_data_h5 must be true or false")
    if save_corrected_data:
        corrected_path = save_corrected_photometry_h5(
            run_dir / "corrected_data.h5",
            data["time_mjd"],
            data["flux"],
            data["flux_err"],
            detec,
        )
        print(f"Saved corrected photometry: {corrected_path}")
    if gaussian_process.get("enabled", False):
        gp_prediction = predict_individual_gp(
            data["time_hours"], data["flux"] - model, plot_theta[-1],
            gp_parameters, data["flux_err"], gaussian_process,
        )
        np.save(run_dir / "gp_prediction.npy", gp_prediction)
        np.save(run_dir / "gp_full_model.npy", model + gp_prediction)
        if config.get("plots", {}).get("gaussian_process", True):
            plot_gaussian_process_fit(
                data["time_hours"], data["flux"], data["flux_err"], detec,
                model, gp_prediction, run_dir / "gaussian_process_fit.png",
            )
    residuals = data["flux"] / detec - astro
    expected_t_s = (
        float(config["transit"].get("expected_midtime_hours", 0.0))
        if time_coordinate == "phase_hours"
        else (
            float(config["transit"]["expected_midtime_mjd"])
            - float(data["time_ref_mjd"])
        ) * 24.0
    )

    if config.get("plots", {}).get("corrected_flux", True):
        plot_corrected_flux(
            data["time_hours"],
            data["flux"],
            data["flux_err"],
            detec,
            astro,
            residuals,
            t_s,
            f"eclipse{eclipse_number}",
            run_dir / "corrected_flux.png",
            expected_t_s=expected_t_s,
        )
    if config.get("plots", {}).get("rednoise", True):
        plot_rednoise(residuals, run_dir / "rednoise.png")
    if (fit_result["sampler_name"] == "emcee"
            and config.get("plots", {}).get("walker", True)):
        plot_walkers(fit_result["chain"], fit_result["labels"], run_dir / "walkers.png")
    if config.get("plots", {}).get("corner", True):
        plot_corner(fit_result["samples"], fit_result["labels"], run_dir / "corner.png")
    if config.get("plots", {}).get("binned_fit", True):
        deterministic_output = (
            run_dir / "binned_fit_deterministic.png"
            if gaussian_process.get("enabled", False)
            else run_dir / "binned_fit.png"
        )
        plot_binned_fit(
            data["time_hours"],
            data["flux"],
            data["flux_err"],
            detec,
            astro,
            deterministic_output,
            bins=int(config.get("plots", {}).get("n_bins", 100)),
            expected_t_s=expected_t_s,
            fitted_t_s=t_s,
            eclipse_depth_summary=fp_summary,
        )
        if gaussian_process.get("enabled", False):
            plot_binned_fit(
                data["time_hours"],
                data["flux"] - gp_prediction,
                data["flux_err"],
                detec,
                astro,
                run_dir / "binned_fit_gp_corrected.png",
                bins=int(config.get("plots", {}).get("n_bins", 100)),
                expected_t_s=expected_t_s,
                fitted_t_s=t_s,
                eclipse_depth_summary=fp_summary,
            )


if __name__ == "__main__":
    main()
