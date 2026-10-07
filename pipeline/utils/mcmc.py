from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import emcee
import numpy as np

from .models import (
    default_initial_guess,
    detrending_parameter_names,
    detec_parameter_count,
    detec_model,
    signal,
    strip_centroid_suffix,
    transit_model,
)
from .evidence import run_dynesty_fit, run_nested_evidence


def summarize_chain(chain: np.ndarray, lnprobchain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = chain.reshape((-1, chain.shape[-1]))
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    summary = np.array([(p50, p84 - p50, p50 - p16) for p16, p50, p84 in zip(*percentiles)], dtype=float)
    best_index = np.unravel_index(np.argmax(lnprobchain), lnprobchain.shape)
    best = chain[best_index]
    return summary, best, samples


def burn_in_steps(nsteps: int, burn_in: int | str | None = "auto") -> int:
    if isinstance(burn_in, int):
        return max(0, min(burn_in, nsteps - 1))
    return max(0, min(5000, nsteps - 1))


def gp_config(config: dict) -> dict:
    return config.get("fit", {}).get("gaussian_process", {})


def gp_enabled(config: dict) -> bool:
    return bool(gp_config(config).get("enabled", False))


def resolve_individual_eccentricity(config: dict) -> dict:
    """Resolve a fixed, uniform, or bounded Gaussian eccentricity prior."""
    transit_cfg = config.get("transit", {})
    default = float(transit_cfg.get("fixed", {}).get("ecc", 0.0))
    initial = float(transit_cfg.get("ecc_initial", default))
    prior = dict(transit_cfg.get("ecc_prior", {"type": "fixed", "value": initial}))
    prior_type = str(prior.get("type", "fixed")).strip().lower()
    if prior_type == "normal":
        prior_type = "gaussian"
    if prior_type == "fixed":
        value = float(prior.get("value", initial))
        if not 0.0 <= value < 1.0:
            raise ValueError("Fixed eccentricity must satisfy 0 <= ecc < 1")
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
            raise ValueError("Gaussian eccentricity prior requires finite mean and sigma > 0")
        resolved = {
            "type": "gaussian", "mean": mean, "sigma": sigma,
            "lower": lower, "upper": upper,
        }
    else:
        raise ValueError("transit.ecc_prior.type must be fixed, uniform, or gaussian")
    if not 0.0 <= lower < upper < 1.0:
        raise ValueError("Eccentricity bounds must satisfy 0 <= lower < upper < 1")
    if not lower <= initial <= upper:
        raise ValueError("transit.ecc_initial is outside its prior bounds")
    return {
        **resolved, "initial": initial, "fixed": False, "bounds": (lower, upper),
    }


def transit_config_with_eccentricity(transit_cfg: dict, eccentricity: float) -> dict:
    """Return a transit configuration with the requested eccentricity."""
    resolved = dict(transit_cfg)
    fixed = dict(resolved.get("fixed", resolved))
    fixed["ecc"] = float(eccentricity)
    resolved["fixed"] = fixed
    return resolved


def gp_components(gaussian_process: dict) -> list[dict]:
    """Return configured kernel components, preserving the legacy single-kernel form."""
    components = gaussian_process.get("kernels")
    if components is None:
        return [gaussian_process]
    if not isinstance(components, list) or not components:
        raise ValueError("gaussian_process.kernels must be a non-empty list")
    return components


def gp_parameter_spec(component: dict) -> tuple[tuple[str, str, str], ...]:
    """Return (label, initial-config key, bounds-config key) for a GP term."""
    kernel_name = str(component.get("kernel", "matern32")).lower()
    if kernel_name == "matern32":
        return (
            ("sigma", "log_sigma_initial", "log_sigma_bounds"),
            ("rho", "log_rho_initial", "log_rho_bounds"),
        )
    if kernel_name in {"sho", "shoterm"}:
        return (
            ("sigma", "log_sigma_initial", "log_sigma_bounds"),
            ("period", "log_period_initial", "log_period_bounds"),
            ("Q", "log_Q_initial", "log_Q_bounds"),
        )
    if kernel_name in {"rotation", "two_sho", "double_sho"}:
        return (
            ("sigma", "log_sigma_initial", "log_sigma_bounds"),
            ("period", "log_period_initial", "log_period_bounds"),
            ("Q0", "log_Q0_initial", "log_Q0_bounds"),
            ("dQ", "log_dQ_initial", "log_dQ_bounds"),
            ("f", "log_f_initial", "log_f_bounds"),
        )
    raise ValueError(f"Unsupported celerite kernel: {kernel_name}")


def gp_parameter_count(gaussian_process: dict) -> int:
    return sum(len(gp_parameter_spec(component)) for component in gp_components(gaussian_process))


def iter_gp_component_parameters(gp_parameters, gaussian_process):
    """Yield each component paired with its named log-parameter values."""
    gp_parameters = np.asarray(gp_parameters, dtype=float)
    expected = gp_parameter_count(gaussian_process)
    if gp_parameters.size != expected:
        raise ValueError(
            f"The GP parameter count does not match the configured kernels "
            f"(expected {expected}, got {gp_parameters.size})"
        )
    offset = 0
    for component in gp_components(gaussian_process):
        names = [item[0] for item in gp_parameter_spec(component)]
        values = gp_parameters[offset : offset + len(names)]
        yield component, dict(zip(names, values))
        offset += len(names)


def make_labels(
    model_type: str, gaussian_process: dict | None = None,
    fix_mid_eclipse_time: bool = False, fix_eccentricity: bool = True,
) -> list[str]:
    n = detec_parameter_count(model_type)
    labels = ([] if fix_mid_eclipse_time else ["dt_s"])
    labels += ["fp"]
    if not fix_eccentricity:
        labels.append("ecc")
    labels += [f"c_{i}" for i in range(1, n + 1)] + ["sigF"]
    if gaussian_process and gaussian_process.get("enabled", False):
        components = gp_components(gaussian_process)
        for index, component in enumerate(components, start=1):
            name = str(component.get("name", f"kernel{index}")).replace(" ", "_")
            labels.extend(
                f"log_gp_{name}_{parameter_name}"
                for parameter_name, _initial_key, _bounds_key in gp_parameter_spec(component)
            )
    return labels


def split_individual_theta(
    theta, model_type, gaussian_process=None, fix_mid_eclipse_time=False,
    fixed_mid_eclipse_time=None, fix_eccentricity=True, fixed_eccentricity=0.0,
):
    theta = np.asarray(theta, dtype=float)
    n_detec = detec_parameter_count(model_type)
    sampled_size = n_detec + 2
    sampled_size += 0 if fix_mid_eclipse_time else 1
    sampled_size += 0 if fix_eccentricity else 1
    idx = 0
    if fix_mid_eclipse_time:
        if fixed_mid_eclipse_time is None:
            raise ValueError("A fixed mid-eclipse time value is required")
        mid_eclipse_time = float(fixed_mid_eclipse_time)
    else:
        mid_eclipse_time = float(theta[idx])
        idx += 1
    fp = float(theta[idx])
    idx += 1
    if fix_eccentricity:
        eccentricity = float(fixed_eccentricity)
    else:
        eccentricity = float(theta[idx])
        idx += 1
    detrending = theta[idx : idx + n_detec]
    idx += n_detec
    sigF = float(theta[idx])
    idx += 1
    deterministic = np.concatenate(([mid_eclipse_time, fp], detrending, [sigF]))
    if gaussian_process and gaussian_process.get("enabled", False):
        n_gp_parameters = gp_parameter_count(gaussian_process)
        if theta.size != sampled_size + n_gp_parameters:
            raise ValueError("GP-enabled individual parameter vector has an unexpected length")
        return deterministic, theta[idx:], eccentricity
    if theta.size != sampled_size:
        raise ValueError("Individual parameter vector has an unexpected length")
    return deterministic, None, eccentricity


def _build_gp(time, sigF, gp_parameters, flux_err, config):
    try:
        from celerite2 import GaussianProcess, terms
    except ImportError as exc:
        raise ImportError(
            "celerite2 is required when fit.gaussian_process.enabled is true"
        ) from exc

    kernel = None
    for component, parameters in iter_gp_component_parameters(gp_parameters, config):
        kernel_name = str(component.get("kernel", "matern32")).lower()
        if kernel_name == "matern32":
            term = terms.Matern32Term(
                sigma=np.exp(parameters["sigma"]),
                rho=np.exp(parameters["rho"]),
            )
        elif kernel_name in {"sho", "shoterm"}:
            term = terms.SHOTerm(
                sigma=np.exp(parameters["sigma"]),
                rho=np.exp(parameters["period"]),
                Q=np.exp(parameters["Q"]),
            )
        elif kernel_name in {"rotation", "two_sho", "double_sho"}:
            term = terms.RotationTerm(
                sigma=np.exp(parameters["sigma"]),
                period=np.exp(parameters["period"]),
                Q0=np.exp(parameters["Q0"]),
                dQ=np.exp(parameters["dQ"]),
                f=np.exp(parameters["f"]),
            )
        else:  # gp_parameter_spec normally catches this first
            raise ValueError(f"Unsupported celerite kernel: {kernel_name}")
        kernel = term if kernel is None else kernel + term
    gp = GaussianProcess(kernel, mean=0.0)
    if config.get("include_flux_errors", False):
        diagonal_error = np.sqrt(np.asarray(flux_err, dtype=float) ** 2 + sigF**2)
    else:
        diagonal_error = np.full_like(np.asarray(time, dtype=float), sigF)
    gp.compute(np.asarray(time, dtype=float), yerr=diagonal_error, quiet=True)
    return gp


def _gp_log_likelihood(time, residuals, sigF, gp_parameters, flux_err, config):
    gp = _build_gp(time, sigF, gp_parameters, flux_err, config)
    return float(gp.log_likelihood(np.asarray(residuals, dtype=float)))


def predict_individual_gp(time, residuals, sigF, gp_parameters, flux_err, config):
    """Return the conditioned GP mean at the observation times."""
    gp = _build_gp(time, sigF, gp_parameters, flux_err, config)
    return np.asarray(
        gp.predict(
            np.asarray(residuals, dtype=float),
            t=np.asarray(time, dtype=float),
            return_cov=False,
        ),
        dtype=float,
    )


def _log_prior_individual(theta, priors):
    t_s, fp = theta[0], theta[1]
    sigF = theta[-1]
    if not np.isfinite(t_s) or not np.isfinite(fp) or not np.isfinite(sigF):
        return -np.inf
    dt_bounds = priors.get("dt_s_bounds_hours", (-1.0, 1.0))
    fp_bounds = priors.get("fp_bounds", (0.0, 1.0))
    sigma_min = priors.get("sigma_min", 0.0)
    eclipse = priors.get("dt_s_initial_hours", 5.0)
    if not eclipse + dt_bounds[0] <= t_s <= eclipse + dt_bounds[1]:
        return -np.inf
    if not fp_bounds[0] <= fp <= fp_bounds[1]:
        return -np.inf
    if not sigF > sigma_min:
        return -np.inf
    sigma_bounds = priors.get("sigma_bounds")
    if sigma_bounds is not None and not sigma_bounds[0] <= sigF <= sigma_bounds[1]:
        return -np.inf
    return 0.0


def resolve_individual_priors(config: dict) -> dict:
    """Resolve legacy top-level fit bounds with optional nested-prior overrides."""
    fit_cfg = config.get("fit", {})
    resolved = {
        key: fit_cfg[key]
        for key in (
            "dt_s_initial_hours", "dt_s_bounds_hours", "fp_bounds", "sigma_min",
            "fix_mid_eclipse_time",
        )
        if key in fit_cfg
    }
    sigma_bounds = fit_cfg.get("sigma_bounds", fit_cfg.get("evidence", {}).get("sigma_bounds"))
    if sigma_bounds is not None:
        resolved["sigma_bounds"] = tuple(map(float, sigma_bounds))
    resolved.update(fit_cfg.get("priors", {}))
    return resolved


def make_individual_log_prob(
    model_type, transit_cfg, priors, gaussian_process=None, detrending_config=None,
    eccentricity=None, include_eccentricity_prior=True,
):
    gaussian_process = gaussian_process or {}
    eccentricity = eccentricity or {
        "type": "fixed", "initial": transit_cfg.get("fixed", {}).get("ecc", 0.0),
        "fixed": True,
    }

    def log_prob(theta, time, flux, flux_err, centroid_x, centroid_y):
        deterministic_theta, gp_parameters, eccentricity_value = split_individual_theta(
            theta, model_type, gaussian_process,
            fix_mid_eclipse_time=bool(priors.get("fix_mid_eclipse_time", False)),
            fixed_mid_eclipse_time=priors.get("dt_s_initial_hours"),
            fix_eccentricity=bool(eccentricity["fixed"]),
            fixed_eccentricity=eccentricity["initial"],
        )
        lp = _log_prior_individual(deterministic_theta, priors)
        if not np.isfinite(lp):
            return -np.inf
        if not eccentricity["fixed"]:
            if not eccentricity["bounds"][0] <= eccentricity_value <= eccentricity["bounds"][1]:
                return -np.inf
            if include_eccentricity_prior and eccentricity["type"] == "gaussian":
                lp += -0.5 * (
                    (eccentricity_value - eccentricity["mean"]) / eccentricity["sigma"]
                ) ** 2
        detrending_theta = deterministic_theta[2:-1]
        detrending_bounds = (detrending_config or {}).get("prior_bounds")
        if detrending_bounds is not None:
            if len(detrending_bounds) != detrending_theta.size:
                return -np.inf
            for value, bounds in zip(detrending_theta, detrending_bounds):
                if not bounds[0] <= value <= bounds[1]:
                    return -np.inf
        sigF = deterministic_theta[-1]
        model = signal(
            time, centroid_x, centroid_y, deterministic_theta, model_type,
            transit_config_with_eccentricity(transit_cfg, eccentricity_value),
            detrending_config,
        )
        if not np.all(np.isfinite(model)):
            return -np.inf
        resid = flux - model
        if gaussian_process.get("enabled", False):
            for component, parameters in iter_gp_component_parameters(
                gp_parameters, gaussian_process
            ):
                for parameter_name, _initial_key, bounds_key in gp_parameter_spec(component):
                    bounds = component.get(bounds_key, [-10.0, 10.0])
                    if parameter_name == "sigma":
                        bounds = component.get(bounds_key, [-20.0, 0.0])
                    if not bounds[0] <= parameters[parameter_name] <= bounds[1]:
                        return -np.inf
            try:
                gp_log_like = _gp_log_likelihood(
                    time, resid, sigF, gp_parameters, flux_err,
                    gaussian_process,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                return -np.inf
            return lp + gp_log_like if np.isfinite(gp_log_like) else -np.inf
        sigma2 = sigF ** 2
        return lp - 0.5 * np.sum((resid ** 2) / sigma2 + np.log(2.0 * np.pi * sigma2))

    return log_prob


def make_individual_initial_theta(config, flux_err, model_type):
    fit_cfg = config.get("fit", {})
    detr_cfg = config.get("detrending", {})
    theta = []
    if not bool(fit_cfg.get("fix_mid_eclipse_time", False)):
        theta.append(float(fit_cfg.get("dt_s_initial_hours", 0.0)))
    theta.append(float(fit_cfg.get("fp_initial", 8.5e-5)))
    eccentricity = resolve_individual_eccentricity(config)
    if not eccentricity["fixed"]:
        theta.append(float(eccentricity["initial"]))
    theta.extend(detr_cfg.get("initial_guess") or default_initial_guess(model_type))
    sigma0 = fit_cfg.get("sigma_initial")
    if sigma0 is None:
        sigma0 = float(np.nanmedian(flux_err))
        if not np.isfinite(sigma0) or sigma0 <= 0.0:
            sigma0 = 1e-3
    theta.append(float(sigma0))
    gaussian_process = gp_config(config)
    if gaussian_process.get("enabled", False):
        for component in gp_components(gaussian_process):
            for parameter_name, initial_key, _bounds_key in gp_parameter_spec(component):
                default = np.log(1e-4) if parameter_name == "sigma" else np.log(1.0)
                theta.append(float(component.get(initial_key, default)))
    return np.asarray(theta, dtype=float)


def individual_evidence_bounds(config, model_type, labels):
    """Build proper uniform-prior bounds in the individual parameter order."""
    fit_cfg = config.get("fit", {})
    priors = resolve_individual_priors(config)
    dt_offsets = priors.get("dt_s_bounds_hours", (-1.0, 1.0))
    dt_center = float(priors.get("dt_s_initial_hours", fit_cfg.get("dt_s_initial_hours", 0.0)))
    bounds = []
    if not bool(fit_cfg.get("fix_mid_eclipse_time", False)):
        bounds.append(
            (dt_center + float(dt_offsets[0]), dt_center + float(dt_offsets[1]))
        )
    bounds.append(tuple(map(float, priors.get("fp_bounds", (0.0, 1.0)))))
    eccentricity = resolve_individual_eccentricity(config)
    if not eccentricity["fixed"]:
        bounds.append(tuple(map(float, eccentricity["bounds"])))

    detrending_bounds = config.get("detrending", {}).get("prior_bounds")
    n_detec = detec_parameter_count(model_type)
    if detrending_bounds is None or len(detrending_bounds) != n_detec:
        raise ValueError(
            f"Sampling requires detrending.prior_bounds with {n_detec} "
            "[lower, upper] pairs, one for every detrending coefficient"
        )
    bounds.extend(tuple(map(float, item)) for item in detrending_bounds)

    sigma_bounds = fit_cfg.get("sigma_bounds", fit_cfg.get("evidence", {}).get("sigma_bounds"))
    if sigma_bounds is None:
        raise ValueError(
            "Sampling requires fit.sigma_bounds: [lower, upper]"
        )
    bounds.append(tuple(map(float, sigma_bounds)))

    gaussian_process = gp_config(config)
    if gaussian_process.get("enabled", False):
        for component in gp_components(gaussian_process):
            for _name, _initial_key, bounds_key in gp_parameter_spec(component):
                component_bounds = component.get(bounds_key)
                if component_bounds is None:
                    raise ValueError(
                        f"Sampling requires {bounds_key} for every GP component"
                    )
                bounds.append(tuple(map(float, component_bounds)))
    if len(bounds) != len(labels):
        raise ValueError(
            f"Resolved {len(bounds)} sampling bounds for {len(labels)} parameters"
        )
    return bounds


def individual_sampling_priors(config, model_type, labels):
    """Return prior-transform metadata in individual parameter order."""
    bounds = individual_evidence_bounds(config, model_type, labels)
    eccentricity = resolve_individual_eccentricity(config)
    priors = []
    bound_index = 0
    if not bool(config.get("fit", {}).get("fix_mid_eclipse_time", False)):
        lower, upper = bounds[bound_index]
        priors.append({"type": "uniform", "lower": lower, "upper": upper})
        bound_index += 1
    lower, upper = bounds[bound_index]
    priors.append({"type": "uniform", "lower": lower, "upper": upper})
    bound_index += 1
    if not eccentricity["fixed"]:
        lower, upper = bounds[bound_index]
        eccentricity_prior = {
            key: value for key, value in eccentricity.items()
            if key not in {"initial", "fixed", "bounds"}
        }
        eccentricity_prior.update(lower=lower, upper=upper)
        priors.append(eccentricity_prior)
        bound_index += 1
    while bound_index < len(bounds):
        lower, upper = bounds[bound_index]
        priors.append({"type": "uniform", "lower": lower, "upper": upper})
        bound_index += 1
    if len(priors) != len(labels):
        raise ValueError(f"Resolved {len(priors)} priors for {len(labels)} parameters")
    return priors


def _configured_parameter(name, specification, default):
    specification = dict(specification or {"type": "fixed", "value": default})
    prior_type = str(specification.get("type", "fixed")).strip().lower()
    if prior_type == "normal":
        prior_type = "gaussian"
    if prior_type == "fixed":
        value = float(specification.get("value", default))
        return {"name": name, "type": "fixed", "value": value, "initial": value, "sampled": False}
    if prior_type not in {"uniform", "gaussian"}:
        raise ValueError(f"Parameter {name}.type must be fixed, uniform, or gaussian")
    lower = float(specification["lower"])
    upper = float(specification["upper"])
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError(f"Parameter {name} requires finite bounds with lower < upper")
    if prior_type == "uniform":
        initial = float(specification.get("initial", default))
        resolved = {"name": name, "type": "uniform", "lower": lower, "upper": upper}
    else:
        mean = float(specification["mean"])
        sigma = float(specification["sigma"])
        if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"Gaussian parameter {name} requires finite mean and sigma > 0")
        initial = float(specification.get("initial", mean))
        resolved = {
            "name": name, "type": "gaussian", "mean": mean, "sigma": sigma,
            "lower": lower, "upper": upper,
        }
    if not lower <= initial <= upper:
        raise ValueError(f"Initial value for {name} is outside [{lower}, {upper}]")
    return {**resolved, "initial": initial, "sampled": True}


def build_individual_parameter_system(config, flux_err, time_ref_mjd):
    """Build named full and sampled parameter definitions for the new YAML schema."""
    configured = config.get("parameters")
    if not isinstance(configured, dict):
        return None
    transit_cfg = config.get("transit", {})
    fixed = transit_cfg.get("fixed", transit_cfg)
    fit_cfg = config.get("fit", {})
    eclipse_cfg = configured.get("eclipse", {})
    detrending_cfg = configured.get("detrending", {})
    noise_cfg = configured.get("noise", {})
    def hours_spec_to_mjd(specification):
        converted = dict(specification)
        for key in ("value", "initial", "mean", "lower", "upper"):
            if key in converted:
                converted[key] = float(time_ref_mjd) + float(converted[key]) / 24.0
        return converted

    phase_timing = (
        "t_secondary_phase_hours" in eclipse_cfg
        and "t_secondary_mjd" not in eclipse_cfg
    )
    if "t_secondary_mjd" not in eclipse_cfg and "t_secondary_hours" in eclipse_cfg:
        eclipse_cfg = dict(eclipse_cfg)
        eclipse_cfg["t_secondary_mjd"] = hours_spec_to_mjd(eclipse_cfg["t_secondary_hours"])
    if "t0_mjd" not in eclipse_cfg and "t0_hours" in eclipse_cfg:
        eclipse_cfg = dict(eclipse_cfg)
        eclipse_cfg["t0_mjd"] = hours_spec_to_mjd(eclipse_cfg["t0_hours"])
    t0_default = float(
        fixed.get(
            "t0_mjd",
            float(time_ref_mjd) + float(fixed.get("t0_hours", 0.0)) / 24.0,
        )
    )
    if phase_timing:
        timing_name = "t_secondary_phase_hours"
        t_secondary_default = float(transit_cfg.get("expected_midtime_hours", 0.0))
    else:
        timing_name = "t_secondary_mjd"
        t_secondary_default = float(
            transit_cfg.get(
                "expected_midtime_mjd",
                float(time_ref_mjd) + float(fit_cfg.get("dt_s_initial_hours", 0.0)) / 24.0,
            )
        )
    u = list(fixed.get("u", [0.1, 0.1]))
    eclipse_defaults = {
        timing_name: t_secondary_default,
        "fp": float(fit_cfg.get("fp_initial", 8.5e-5)),
        "t0_mjd": t0_default,
        "period_days": float(fixed.get("period_days", fixed.get("per_days", 1.0))),
        "rp": float(fixed.get("rp", 0.1)), "a": float(fixed.get("a", 10.0)),
        "inc": float(fixed.get("inc", 90.0)), "ecc": float(fixed.get("ecc", 0.0)),
        "w": float(fixed.get("w", 90.0)), "u1": float(u[0]), "u2": float(u[1]),
    }
    full = []
    for name, default in eclipse_defaults.items():
        full.append(_configured_parameter(name, eclipse_cfg.get(name), default))
    model_type = config["detrending"]["model_type"]
    detec_names = detrending_parameter_names(model_type)
    legacy_initial = config["detrending"].get("initial_guess") or default_initial_guess(model_type)
    if len(legacy_initial) != len(detec_names):
        raise ValueError("detrending.initial_guess does not match the selected model")
    for name, default in zip(detec_names, legacy_initial):
        full.append(_configured_parameter(name, detrending_cfg.get(name), default))
    sigma_default = fit_cfg.get("sigma_initial")
    if sigma_default is None:
        sigma_default = float(np.nanmedian(flux_err))
    full.append(_configured_parameter("sigma", noise_cfg.get("sigma"), sigma_default))
    sampled = [parameter for parameter in full if parameter["sampled"]]
    return {"full": full, "sampled": sampled, "detrending_names": detec_names}


def reconstruct_configured_individual(
    sampled_theta, config, flux_err, time_ref_mjd, parameter_system=None,
):
    system = parameter_system or build_individual_parameter_system(config, flux_err, time_ref_mjd)
    if system is None:
        raise ValueError("The named parameter system is not configured")
    sampled_theta = np.asarray(sampled_theta, dtype=float)
    gp = gp_config(config)
    n_gp = gp_parameter_count(gp) if gp.get("enabled", False) else 0
    model_values = sampled_theta[:-n_gp] if n_gp else sampled_theta
    gp_values = sampled_theta[-n_gp:] if n_gp else None
    if model_values.size != len(system["sampled"]):
        raise ValueError("Configured individual parameter vector has an unexpected length")
    sampled_map = dict(zip((p["name"] for p in system["sampled"]), model_values))
    values = {
        p["name"]: float(sampled_map[p["name"]]) if p["sampled"] else float(p["value"])
        for p in system["full"]
    }
    transit_cfg = dict(config["transit"])
    fixed = dict(transit_cfg.get("fixed", transit_cfg))
    fixed.update({
        "t0_mjd": values["t0_mjd"],
        "t0_hours": (values["t0_mjd"] - float(time_ref_mjd)) * 24.0,
        "period_days": values["period_days"],
        "rp": values["rp"], "a": values["a"], "inc": values["inc"],
        "ecc": values["ecc"], "w": values["w"], "u": [values["u1"], values["u2"]],
    })
    transit_cfg["fixed"] = fixed
    transit_cfg["time_ref_mjd"] = float(time_ref_mjd)
    detrending = [values[name] for name in system["detrending_names"]]
    if "t_secondary_phase_hours" in values:
        t_secondary_hours = values["t_secondary_phase_hours"]
    else:
        t_secondary_hours = (
            values["t_secondary_mjd"] - float(time_ref_mjd)
        ) * 24.0
    deterministic = np.asarray(
        [
            t_secondary_hours,
            values["fp"], *detrending, values["sigma"],
        ],
        dtype=float,
    )
    return deterministic, gp_values, transit_cfg, values, system


def _configured_individual_log_probability(config, data, include_gaussian_priors=True):
    model_type = config["detrending"]["model_type"]
    gaussian_process = gp_config(config)
    parameter_system = build_individual_parameter_system(
        config, data["flux_err"], data.get("time_ref_mjd", data["time_mjd"][0])
    )

    def log_probability(theta, time, flux, flux_err, centroid_x, centroid_y):
        deterministic, gp_parameters, transit_cfg, values, system = reconstruct_configured_individual(
            theta, config, flux_err,
            data.get("time_ref_mjd", data["time_mjd"][0]), parameter_system
        )
        lp = 0.0
        for parameter in system["sampled"]:
            value = values[parameter["name"]]
            if not parameter["lower"] <= value <= parameter["upper"]:
                return -np.inf
            if include_gaussian_priors and parameter["type"] == "gaussian":
                lp += -0.5 * ((value - parameter["mean"]) / parameter["sigma"]) ** 2
        try:
            model = signal(
                time, centroid_x, centroid_y, deterministic, model_type, transit_cfg,
                config["detrending"],
            )
        except (ValueError, RuntimeError, FloatingPointError):
            return -np.inf
        if not np.all(np.isfinite(model)):
            return -np.inf
        residuals = flux - model
        sigma = values["sigma"]
        if sigma <= 0.0:
            return -np.inf
        if gaussian_process.get("enabled", False):
            for component, component_values in iter_gp_component_parameters(
                gp_parameters, gaussian_process
            ):
                for parameter_name, _initial_key, bounds_key in gp_parameter_spec(component):
                    lower, upper = component[bounds_key]
                    if not lower <= component_values[parameter_name] <= upper:
                        return -np.inf
            try:
                likelihood = _gp_log_likelihood(
                    time, residuals, sigma, gp_parameters, flux_err, gaussian_process
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                return -np.inf
            return lp + likelihood if np.isfinite(likelihood) else -np.inf
        sigma2 = sigma**2
        return lp - 0.5 * np.sum(residuals**2 / sigma2 + np.log(2.0 * np.pi * sigma2))
    return log_probability


def run_configured_individual_fit(data, config):
    time_ref_mjd = data.get("time_ref_mjd", data["time_mjd"][0])
    system = build_individual_parameter_system(config, data["flux_err"], time_ref_mjd)
    gaussian_process = gp_config(config)
    labels = [parameter["name"] for parameter in system["sampled"]]
    theta0 = [parameter["initial"] for parameter in system["sampled"]]
    bounds = [(parameter["lower"], parameter["upper"]) for parameter in system["sampled"]]
    prior_specs = [
        {key: parameter[key] for key in ("type", "lower", "upper", "mean", "sigma") if key in parameter}
        for parameter in system["sampled"]
    ]
    if gaussian_process.get("enabled", False):
        for index, component in enumerate(gp_components(gaussian_process), start=1):
            component_name = str(component.get("name", f"kernel{index}")).replace(" ", "_")
            for parameter_name, initial_key, bounds_key in gp_parameter_spec(component):
                labels.append(f"log_gp_{component_name}_{parameter_name}")
                theta0.append(float(component[initial_key]))
                component_bounds = tuple(map(float, component[bounds_key]))
                bounds.append(component_bounds)
                prior_specs.append({"type": "uniform", "lower": component_bounds[0], "upper": component_bounds[1]})
    return np.asarray(theta0), labels, bounds, prior_specs


def run_individual_mcmc(data, config, output_dir):
    import numpy as np

    fit_cfg = config.get("fit", {})
    sampler_name = str(fit_cfg.get("sampler", "emcee")).strip().lower()
    if sampler_name not in {"emcee", "dynesty"}:
        raise ValueError("fit.sampler must be either 'emcee' or 'dynesty'")
    model_type = config["detrending"]["model_type"]
    gaussian_process = gp_config(config)
    seed = fit_cfg.get("random_seed")
    if seed is not None:
        np.random.seed(int(seed))

    if config.get("parameters") is not None:
        theta0, labels, sampling_bounds, sampling_priors = run_configured_individual_fit(
            data, config
        )
        configured_system = build_individual_parameter_system(
            config, data["flux_err"], data.get("time_ref_mjd", data["time_mjd"][0])
        )
        fixed_names = [p["name"] for p in configured_system["full"] if not p["sampled"]]
        print("Fixed parameters:", ", ".join(fixed_names) or "none")
        print("Sampled parameters:", ", ".join(labels) or "none")
        log_probability = _configured_individual_log_probability(
            config, data, include_gaussian_priors=True
        )
        nested_log_likelihood = _configured_individual_log_probability(
            config, data, include_gaussian_priors=False
        )
    else:
        transit_cfg = dict(config["transit"])
        transit_cfg["time_ref_mjd"] = float(data.get("time_ref_mjd", data["time_mjd"][0]))
        priors = resolve_individual_priors(config)
        eccentricity = resolve_individual_eccentricity(config)
        theta0 = make_individual_initial_theta(config, data["flux_err"], model_type)
        fix_mid_eclipse_time = bool(fit_cfg.get("fix_mid_eclipse_time", False))
        labels = make_labels(
            model_type, gaussian_process, fix_mid_eclipse_time, eccentricity["fixed"]
        )
        sampling_bounds = individual_evidence_bounds(config, model_type, labels)
        sampling_priors = individual_sampling_priors(config, model_type, labels)
        log_probability = make_individual_log_prob(
            model_type, transit_cfg, priors, gaussian_process, config["detrending"],
            eccentricity=eccentricity, include_eccentricity_prior=True,
        )
        nested_log_likelihood = make_individual_log_prob(
            model_type, transit_cfg, priors, gaussian_process, config["detrending"],
            eccentricity=eccentricity, include_eccentricity_prior=False,
        )
    print('theta0: ', theta0)
    ndim = len(theta0)
    if ndim == 0:
        raise ValueError("At least one parameter must be uniform or gaussian to run a fit")
    evidence_enabled = (
        sampler_name == "dynesty"
        or fit_cfg.get("evidence", {}).get("enabled", True)
    )
    log_probability_args = (
        data["time_hours"], data["flux"], data["flux_err"],
        data["centroid_x"], data["centroid_y"],
    )
    if sampler_name == "dynesty":
        return run_dynesty_fit(
            nested_log_likelihood, labels, sampling_bounds, config, theta0,
            args=log_probability_args, prior_specs=sampling_priors,
        )

    nwalkers = int(fit_cfg.get("nwalkers", 70))
    nsteps = int(fit_cfg.get("nsteps", 5000))
    bounds_array = np.asarray(sampling_bounds, dtype=float)
    if np.any(theta0 < bounds_array[:, 0]) or np.any(theta0 > bounds_array[:, 1]):
        raise ValueError("One or more initial values are outside their prior bounds")
    widths = bounds_array[:, 1] - bounds_array[:, 0]
    scales = np.maximum(widths * 1e-4, np.finfo(float).eps)
    low = np.maximum(bounds_array[:, 0], theta0 - scales)
    high = np.minimum(bounds_array[:, 1], theta0 + scales)
    pos = np.random.uniform(low, high, size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=log_probability_args,
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
            args=log_probability_args,
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





