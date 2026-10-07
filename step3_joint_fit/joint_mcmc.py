from __future__ import annotations

import emcee
import numpy as np

from pipeline.utils.mcmc import _configured_parameter, burn_in_steps, summarize_chain
from pipeline.utils.models import default_initial_guess, detrending_parameter_names, signal
from pipeline.utils.evidence import run_dynesty_fit, run_nested_evidence


def _uniform_bounds(prior: dict, label: str) -> tuple[float, float]:
    if prior.get("type", "uniform") != "uniform":
        raise ValueError(f"{label} currently supports only a uniform prior")
    lower = float(prior["lower"])
    upper = float(prior["upper"])
    if not lower < upper:
        raise ValueError(f"{label} must have lower < upper")
    return lower, upper


def resolve_joint_fp_bounds(config: dict) -> tuple[float, float]:
    """Resolve the nested fp prior, falling back to legacy fit.fp_bounds."""
    fit_cfg = config.get("fit", {})
    fp_prior = fit_cfg.get("priors", {}).get("fp")
    if fp_prior is not None:
        return _uniform_bounds(fp_prior, "fit.priors.fp")
    bounds = fit_cfg.get("fp_bounds", [0.0, 1.0])
    return _uniform_bounds(
        {"type": "uniform", "lower": bounds[0], "upper": bounds[1]},
        "fit.fp_bounds",
    )


def resolve_joint_sigma_min(config: dict) -> float:
    fit_cfg = config.get("fit", {})
    return float(fit_cfg.get("priors", {}).get("sigma_min", fit_cfg.get("sigma_min", 0.0)))


def mjd_to_observation_hours(mjd: float, first_data_time: float) -> float:
    """Map MJD to hours from the first sample for MJD- or JD/BJD-based data."""
    first_mjd = first_data_time - 2_400_000.5 if first_data_time > 1_000_000 else first_data_time
    return (float(mjd) - float(first_mjd)) * 24.0


def transit_config_for_eclipse(
    config: dict, eccentricity: float, time_ref_mjd: float | None = None,
) -> dict:
    transit_cfg = dict(config["transit"])
    fixed = dict(transit_cfg.get("fixed", transit_cfg))
    fixed["ecc"] = float(eccentricity)
    transit_cfg["fixed"] = fixed
    if time_ref_mjd is not None:
        transit_cfg["time_ref_mjd"] = float(time_ref_mjd)
    return transit_cfg


def make_joint_labels(eclipse_specs: list[dict]) -> list[str]:
    labels = ["fp"]
    for spec in eclipse_specs:
        prefix = f"e{spec['eclipse_number']}"
        if not spec.get("fix_mid_eclipse_time", False):
            labels.append(f"{prefix}_t0_mjd")
        if not spec.get("fix_eccentricity", False):
            labels.append(f"{prefix}_ecc")
        labels.extend([f"{prefix}_c_{i}" for i in range(1, spec["detec_param_count"] + 1)])
        labels.append(f"{prefix}_sigF")
    return labels


def build_joint_theta0(config: dict, eclipse_specs: list[dict]) -> np.ndarray:
    fp_initial = float(config.get("fit", {}).get("fp_initial", 8.5e-5))
    fp_bounds = resolve_joint_fp_bounds(config)
    if not fp_bounds[0] <= fp_initial <= fp_bounds[1]:
        raise ValueError(f"fit.fp_initial={fp_initial} is outside the resolved bounds {fp_bounds}")
    theta = [fp_initial]
    for spec in eclipse_specs:
        if not spec["t0_mjd_bounds"][0] <= spec["t0_mjd_initial"] <= spec["t0_mjd_bounds"][1]:
            raise ValueError(
                f"Eclipse {spec['eclipse_number']} expected_midtime_mjd is outside its prior"
            )
        if (not spec.get("fix_eccentricity", False)
                and not spec["ecc_bounds"][0] <= spec["ecc_initial"] <= spec["ecc_bounds"][1]):
            raise ValueError(f"Eclipse {spec['eclipse_number']} ecc_initial is outside its prior")
        if not spec.get("fix_mid_eclipse_time", False):
            theta.append(spec["t0_mjd_initial"])
        if not spec.get("fix_eccentricity", False):
            theta.append(spec["ecc_initial"])
        theta.extend(spec["detrending_initial_guess"])
        theta.append(spec["sigma_initial"])
    return np.asarray(theta, dtype=float)


def joint_evidence_bounds(config, eclipse_specs, labels):
    """Build proper uniform-prior bounds in joint parameter order."""
    bounds = [resolve_joint_fp_bounds(config)]
    fit_cfg = config.get("fit", {})
    global_sigma_bounds = fit_cfg.get("sigma_bounds", fit_cfg.get("evidence", {}).get("sigma_bounds"))
    for spec in eclipse_specs:
        if not spec.get("fix_mid_eclipse_time", False):
            bounds.append(spec["t0_mjd_bounds"])
        if not spec.get("fix_eccentricity", False):
            bounds.append(spec["ecc_bounds"])
        detrending_bounds = spec.get("detrending_prior_bounds")
        if detrending_bounds is None or len(detrending_bounds) != spec["detec_param_count"]:
            raise ValueError(
                f"Sampling eclipse {spec['eclipse_number']} requires "
                f"detrending.prior_bounds with {spec['detec_param_count']} pairs"
            )
        bounds.extend(tuple(map(float, item)) for item in detrending_bounds)
        sigma_bounds = spec.get("sigma_bounds")
        if sigma_bounds is None:
            sigma_bounds = global_sigma_bounds
        if sigma_bounds is None:
            raise ValueError(
                f"Sampling eclipse {spec['eclipse_number']} requires "
                "fit.sigma_bounds, locally or at the joint-fit level"
            )
        bounds.append(tuple(map(float, sigma_bounds)))
    if len(bounds) != len(labels):
        raise ValueError(
            f"Resolved {len(bounds)} sampling bounds for {len(labels)} parameters"
        )
    return bounds


def joint_sampling_priors(config, eclipse_specs, labels):
    """Return prior-transform metadata in the same order as the joint vector."""
    bounds = joint_evidence_bounds(config, eclipse_specs, labels)
    priors = [
        {"type": "uniform", "lower": bounds[0][0], "upper": bounds[0][1]}
    ]
    bound_index = 1
    for spec in eclipse_specs:
        if not spec.get("fix_mid_eclipse_time", False):
            lower, upper = bounds[bound_index]
            priors.append({"type": "uniform", "lower": lower, "upper": upper})
            bound_index += 1
        if not spec.get("fix_eccentricity", False):
            lower, upper = bounds[bound_index]
            eccentricity_prior = dict(spec["ecc_prior"])
            eccentricity_prior.update(lower=lower, upper=upper)
            priors.append(eccentricity_prior)
            bound_index += 1
        for _ in range(spec["detec_param_count"] + 1):
            lower, upper = bounds[bound_index]
            priors.append({"type": "uniform", "lower": lower, "upper": upper})
            bound_index += 1
    if len(priors) != len(labels):
        raise ValueError(f"Resolved {len(priors)} priors for {len(labels)} parameters")
    return priors


def joint_split_theta(theta: np.ndarray, eclipse_specs: list[dict]) -> tuple[float, dict]:
    theta = np.asarray(theta, dtype=float)
    fp = float(theta[0])
    idx = 1
    parts = {}
    for spec in eclipse_specs:
        n = spec["detec_param_count"]
        if spec.get("fix_mid_eclipse_time", False):
            t0_mjd = float(spec["t0_mjd_initial"])
        else:
            t0_mjd = float(theta[idx])
            idx += 1
        if spec.get("fix_eccentricity", False):
            ecc = float(spec["ecc_initial"])
        else:
            ecc = float(theta[idx])
            idx += 1
        parts[spec["eclipse_number"]] = {
            "t0_mjd": t0_mjd,
            "ecc": ecc,
            "detec": theta[idx : idx + n],
            "sigF": float(theta[idx + n]),
        }
        idx += n + 1
    if idx != len(theta):
        raise ValueError("Joint parameter vector has an unexpected length")
    return fp, parts


def make_joint_log_prob(
    config: dict, eclipse_specs: list[dict], include_eccentricity_prior: bool = True,
):
    fp_bounds = resolve_joint_fp_bounds(config)
    sigma_min = resolve_joint_sigma_min(config)

    def log_prob(theta, datasets):
        fp, parts = joint_split_theta(theta, eclipse_specs)
        if not np.isfinite(fp) or not fp_bounds[0] <= fp <= fp_bounds[1]:
            return -np.inf

        total_log_likelihood = 0.0
        for spec, dataset in zip(eclipse_specs, datasets):
            part = parts[spec["eclipse_number"]]
            if (not spec.get("fix_mid_eclipse_time", False)
                    and not spec["t0_mjd_bounds"][0] <= part["t0_mjd"] <= spec["t0_mjd_bounds"][1]):
                return -np.inf
            if not spec.get("fix_eccentricity", False):
                if not spec["ecc_bounds"][0] <= part["ecc"] <= spec["ecc_bounds"][1]:
                    return -np.inf
                if include_eccentricity_prior and spec["ecc_prior"]["type"] == "gaussian":
                    mean = spec["ecc_prior"]["mean"]
                    sigma = spec["ecc_prior"]["sigma"]
                    total_log_likelihood += -0.5 * ((part["ecc"] - mean) / sigma) ** 2
            if not np.isfinite(part["sigF"]) or part["sigF"] <= sigma_min:
                return -np.inf
            detrending_bounds = spec.get("detrending_prior_bounds")
            if detrending_bounds is not None:
                if len(detrending_bounds) != len(part["detec"]):
                    return -np.inf
                for value, bounds in zip(part["detec"], detrending_bounds):
                    if not bounds[0] <= value <= bounds[1]:
                        return -np.inf
            sigma_bounds = spec.get("sigma_bounds")
            if sigma_bounds is None:
                fit_cfg = config.get("fit", {})
                sigma_bounds = fit_cfg.get("sigma_bounds", fit_cfg.get("evidence", {}).get("sigma_bounds"))
            if sigma_bounds is not None and not sigma_bounds[0] <= part["sigF"] <= sigma_bounds[1]:
                return -np.inf

            t0_hours = mjd_to_observation_hours(part["t0_mjd"], dataset["time_mjd"][0])
            full_theta = np.concatenate(([t0_hours, fp], part["detec"], [part["sigF"]]))
            model = signal(
                dataset["time_hours"],
                dataset["centroid_x"],
                dataset["centroid_y"],
                full_theta,
                spec["detrending"]["model_type"],
                {
                    **transit_config_for_eclipse(config, part["ecc"]),
                    "time_ref_mjd": float(dataset["time_mjd"][0]),
                },
                spec["detrending"],
            )
            if not np.all(np.isfinite(model)):
                return -np.inf
            sigma2 = part["sigF"] ** 2
            residuals = dataset["flux"] - model
            total_log_likelihood += -0.5 * np.sum(
                residuals**2 / sigma2 + np.log(2.0 * np.pi * sigma2)
            )
        return total_log_likelihood

    return log_prob


def initialize_joint_walkers(theta0, nwalkers, config, eclipse_specs):
    """Initialize walkers near YAML values while respecting every hard bound."""
    theta0 = np.asarray(theta0, dtype=float)
    labels = make_joint_labels(eclipse_specs)
    bounds = np.asarray(joint_evidence_bounds(config, eclipse_specs, labels), dtype=float)
    if np.any(theta0 < bounds[:, 0]) or np.any(theta0 > bounds[:, 1]):
        raise ValueError("One or more joint-fit initial values are outside their prior bounds")
    widths = bounds[:, 1] - bounds[:, 0]
    scales = np.maximum(widths * 1e-4, np.finfo(float).eps)
    low = np.maximum(bounds[:, 0], theta0 - scales)
    high = np.minimum(bounds[:, 1], theta0 + scales)
    return np.random.uniform(low, high, size=(nwalkers, theta0.size))


def build_configured_joint_system(config, eclipse_specs, datasets):
    """Build shared astrophysical and per-eclipse named parameter definitions."""
    named = config.get("parameters")
    if not isinstance(named, dict):
        return None
    fixed = config["transit"].get("fixed", config["transit"])
    u = list(fixed.get("u", [0.1, 0.1]))
    shared_defaults = {
        "fp": float(config.get("fit", {}).get("fp_initial", 8.5e-5)),
        "t0_mjd": float(fixed.get("t0_mjd", datasets[0]["time_mjd"][0])),
        "period_days": float(fixed.get("period_days", fixed.get("per_days", 1.0))),
        "rp": float(fixed.get("rp", 0.1)), "a": float(fixed.get("a", 10.0)),
        "inc": float(fixed.get("inc", 90.0)), "ecc": float(fixed.get("ecc", 0.0)),
        "w": float(fixed.get("w", 90.0)), "u1": float(u[0]), "u2": float(u[1]),
    }
    shared_cfg = named.get("eclipse", {})
    shared = [
        _configured_parameter(name, shared_cfg.get(name), default)
        for name, default in shared_defaults.items()
    ]
    per_eclipse = []
    for spec, dataset in zip(eclipse_specs, datasets):
        configured = spec.get("parameters", {})
        eclipse_cfg = configured.get("eclipse", {})
        timing_spec = eclipse_cfg.get("t_secondary_mjd")
        if timing_spec is None:
            timing_spec = (
                {"type": "fixed", "value": spec["t0_mjd_initial"]}
                if spec.get("fix_mid_eclipse_time", False)
                else {
                    "type": "uniform", "initial": spec["t0_mjd_initial"],
                    "lower": spec["t0_mjd_bounds"][0], "upper": spec["t0_mjd_bounds"][1],
                }
            )
        local = [_configured_parameter(
            "t_secondary_mjd", timing_spec, spec["t0_mjd_initial"]
        )]
        eccentricity_spec = eclipse_cfg.get("ecc")
        if eccentricity_spec is not None:
            local.append(_configured_parameter(
                "ecc", eccentricity_spec, spec["ecc_initial"]
            ))
        names = detrending_parameter_names(spec["detrending"]["model_type"])
        defaults = spec.get("detrending_initial_guess") or default_initial_guess(
            spec["detrending"]["model_type"]
        )
        det_cfg = configured.get("detrending", {})
        legacy_bounds = spec.get("detrending_prior_bounds")
        for index, (name, default) in enumerate(zip(names, defaults)):
            parameter_spec = det_cfg.get(name)
            if parameter_spec is None and legacy_bounds is not None:
                parameter_spec = {
                    "type": "uniform", "initial": default,
                    "lower": legacy_bounds[index][0], "upper": legacy_bounds[index][1],
                }
            local.append(_configured_parameter(name, parameter_spec, default))
        sigma_spec = configured.get("noise", {}).get("sigma")
        if sigma_spec is None:
            sigma_bounds = spec.get("sigma_bounds") or config.get("fit", {}).get("sigma_bounds")
            if sigma_bounds is not None:
                sigma_spec = {
                    "type": "uniform", "initial": spec["sigma_initial"],
                    "lower": sigma_bounds[0], "upper": sigma_bounds[1],
                }
        local.append(_configured_parameter(
            "sigma", sigma_spec, spec["sigma_initial"]
        ))
        per_eclipse.append({"spec": spec, "parameters": local, "detrending_names": names})
    labels = [p["name"] for p in shared if p["sampled"]]
    ordered_sampled = [p for p in shared if p["sampled"]]
    for item in per_eclipse:
        prefix = f"e{item['spec']['eclipse_number']}_"
        for parameter in item["parameters"]:
            if parameter["sampled"]:
                labels.append(prefix + parameter["name"])
                ordered_sampled.append(parameter)
    return {"shared": shared, "per_eclipse": per_eclipse, "sampled": ordered_sampled, "labels": labels}


def reconstruct_configured_joint(theta, config, eclipse_specs, datasets, system=None):
    system = system or build_configured_joint_system(config, eclipse_specs, datasets)
    theta = np.asarray(theta, dtype=float)
    if theta.size != len(system["sampled"]):
        raise ValueError("Configured joint parameter vector has an unexpected length")
    cursor = iter(theta)
    shared_values = {
        p["name"]: float(next(cursor)) if p["sampled"] else float(p["value"])
        for p in system["shared"]
    }
    parts = {}
    for item, dataset in zip(system["per_eclipse"], datasets):
        values = {
            p["name"]: float(next(cursor)) if p["sampled"] else float(p["value"])
            for p in item["parameters"]
        }
        transit_cfg = dict(config["transit"])
        fixed = dict(transit_cfg.get("fixed", transit_cfg))
        eclipse_eccentricity = values.get("ecc", shared_values["ecc"])
        fixed.update({
            "t0_mjd": shared_values["t0_mjd"],
            "t0_hours": mjd_to_observation_hours(shared_values["t0_mjd"], dataset["time_mjd"][0]),
            "period_days": shared_values["period_days"], "rp": shared_values["rp"],
            "a": shared_values["a"], "inc": shared_values["inc"],
            "ecc": eclipse_eccentricity, "w": shared_values["w"],
            "u": [shared_values["u1"], shared_values["u2"]],
        })
        transit_cfg["fixed"] = fixed
        transit_cfg["time_ref_mjd"] = float(dataset["time_mjd"][0])
        parts[item["spec"]["eclipse_number"]] = {
            "t0_mjd": values["t_secondary_mjd"], "ecc": eclipse_eccentricity,
            "detec": np.asarray([values[n] for n in item["detrending_names"]]),
            "sigF": values["sigma"], "transit_config": transit_cfg, "values": values,
        }
    return shared_values["fp"], parts, shared_values, system


def make_configured_joint_log_prob(config, eclipse_specs, datasets, include_gaussian_priors=True):
    system = build_configured_joint_system(config, eclipse_specs, datasets)

    def log_prob(theta, supplied_datasets):
        fp, parts, shared, _ = reconstruct_configured_joint(
            theta, config, eclipse_specs, supplied_datasets, system
        )
        lp = 0.0
        values_in_order = list(theta)
        for parameter, value in zip(system["sampled"], values_in_order):
            if not parameter["lower"] <= value <= parameter["upper"]:
                return -np.inf
            if include_gaussian_priors and parameter["type"] == "gaussian":
                lp += -0.5 * ((value - parameter["mean"]) / parameter["sigma"]) ** 2
        for spec, dataset in zip(eclipse_specs, supplied_datasets):
            part = parts[spec["eclipse_number"]]
            t_secondary_hours = mjd_to_observation_hours(part["t0_mjd"], dataset["time_mjd"][0])
            full_theta = np.concatenate(([t_secondary_hours, fp], part["detec"], [part["sigF"]]))
            try:
                model = signal(
                    dataset["time_hours"], dataset["centroid_x"], dataset["centroid_y"],
                    full_theta, spec["detrending"]["model_type"], part["transit_config"],
                    spec["detrending"],
                )
            except (ValueError, RuntimeError, FloatingPointError):
                return -np.inf
            if not np.all(np.isfinite(model)) or part["sigF"] <= 0:
                return -np.inf
            sigma2 = part["sigF"] ** 2
            residuals = dataset["flux"] - model
            lp += -0.5 * np.sum(residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))
        return lp
    return log_prob


def configured_joint_setup(config, eclipse_specs, datasets):
    system = build_configured_joint_system(config, eclipse_specs, datasets)
    theta0 = np.asarray([p["initial"] for p in system["sampled"]])
    bounds = [(p["lower"], p["upper"]) for p in system["sampled"]]
    priors = [
        {key: p[key] for key in ("type", "lower", "upper", "mean", "sigma") if key in p}
        for p in system["sampled"]
    ]
    return theta0, system["labels"], bounds, priors, system


def run_joint_mcmc(datasets, eclipse_specs, config, output_dir):
    fit_cfg = config.get("fit", {})
    sampler_name = str(fit_cfg.get("sampler", "emcee")).strip().lower()
    if sampler_name not in {"emcee", "dynesty"}:
        raise ValueError("fit.sampler must be either 'emcee' or 'dynesty'")
    seed = fit_cfg.get("random_seed")
    if seed is not None:
        np.random.seed(int(seed))

    if config.get("parameters") is not None:
        theta0, labels, sampling_bounds, sampling_priors, configured_system = (
            configured_joint_setup(config, eclipse_specs, datasets)
        )
        log_probability = make_configured_joint_log_prob(
            config, eclipse_specs, datasets, include_gaussian_priors=True
        )
        nested_log_likelihood = make_configured_joint_log_prob(
            config, eclipse_specs, datasets, include_gaussian_priors=False
        )
        fixed_names = [p["name"] for p in configured_system["shared"] if not p["sampled"]]
        print("Fixed shared parameters:", ", ".join(fixed_names) or "none")
        print("Sampled parameters:", ", ".join(labels) or "none")
    else:
        theta0 = build_joint_theta0(config, eclipse_specs)
        labels = make_joint_labels(eclipse_specs)
        sampling_bounds = joint_evidence_bounds(config, eclipse_specs, labels)
        sampling_priors = joint_sampling_priors(config, eclipse_specs, labels)
        log_probability = make_joint_log_prob(config, eclipse_specs, include_eccentricity_prior=True)
        nested_log_likelihood = make_joint_log_prob(
            config, eclipse_specs, include_eccentricity_prior=False
        )
    ndim = len(theta0)
    if ndim == 0:
        raise ValueError("At least one joint parameter must be uniform or gaussian")
    evidence_enabled = (
        sampler_name == "dynesty"
        or fit_cfg.get("evidence", {}).get("enabled", True)
    )
    if sampler_name == "dynesty":
        return run_dynesty_fit(
            nested_log_likelihood, labels, sampling_bounds, config, theta0,
            args=(datasets,), prior_specs=sampling_priors,
        )

    nwalkers = int(fit_cfg.get("nwalkers", 70))
    nsteps = int(fit_cfg.get("nsteps", 5000))
    if nwalkers < 2 * ndim:
        raise ValueError(f"nwalkers must be at least 2 * ndim ({2 * ndim})")
    bounds_array = np.asarray(sampling_bounds, dtype=float)
    if np.any(theta0 < bounds_array[:, 0]) or np.any(theta0 > bounds_array[:, 1]):
        raise ValueError("One or more joint-fit initial values are outside their prior bounds")
    widths = bounds_array[:, 1] - bounds_array[:, 0]
    scales = np.maximum(widths * 1e-4, np.finfo(float).eps)
    pos = np.random.uniform(
        np.maximum(bounds_array[:, 0], theta0 - scales),
        np.minimum(bounds_array[:, 1], theta0 + scales),
        size=(nwalkers, ndim),
    )
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(datasets,)
    )
    pos2, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)
    discard = burn_in_steps(nsteps, config.get("fit", {}).get("burn_in", "auto"))
    lnprobchain = sampler.get_log_prob(discard=discard).swapaxes(0, 1)
    chain = sampler.get_chain(discard=discard).swapaxes(0, 1)
    summary, best, samples = summarize_chain(chain, lnprobchain)
    evidence = None
    if evidence_enabled:
        evidence = run_nested_evidence(
            nested_log_likelihood,
            labels,
            sampling_bounds,
            config,
            args=(datasets,),
            prior_specs=sampling_priors,
        )
    return {
        "sampler": sampler,
        "sampler_name": "emcee",
        "chain": chain,
        "lnprobchain": lnprobchain,
        "final_state": pos2,
        "final_prob": prob,
        "summary": summary,
        "best": best,
        "samples": samples,
        "labels": labels,
        "theta0": theta0,
        "discard": discard,
        "evidence": evidence,
    }
