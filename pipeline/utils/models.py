from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np

try:
    import batman
except Exception:  # pragma: no cover
    batman = None


SUPPORTED_MODELS = {
    "offset": 1,
    "exp": 3,
    "linear": 2,
    "polynomial": 4,
    "double_exp": 4,
    "exp+linear": 5,
    "exp+polynomial": 7,
    "linear+polynomial": 6,
}


def strip_centroid_suffix(model_type: str) -> str:
    return model_type.replace("_2nd_order_centroid", "")


def split_gaussian_suffix(model_type: str) -> tuple[str, int]:
    """Return the unchanged base model name and number of additive Gaussians."""
    model_type = strip_centroid_suffix(model_type)
    match = re.fullmatch(r"(.+)\+(?:(\d+))?gaussian", model_type)
    if match is None:
        return model_type, 0
    count = int(match.group(2) or 1)
    if count < 1:
        raise ValueError("The Gaussian component count must be at least one")
    return match.group(1), count


def split_additive_suffixes(model_type: str) -> tuple[str, int, int]:
    """Parse optional Gaussian and fixed-peak flare additions."""
    model_type = strip_centroid_suffix(model_type)
    if model_type in {"flare", "2flare"}:
        return "offset", 0, 1 if model_type == "flare" else 2
    flare_match = re.fullmatch(r"(.+)\+(?:(\d+))?flare", model_type)
    flare_count = 0
    if flare_match is not None:
        model_type = flare_match.group(1)
        flare_count = int(flare_match.group(2) or 1)
        if flare_count not in (1, 2):
            raise ValueError("The stellar flare component supports one or two peaks")
    base, gaussian_count = split_gaussian_suffix(model_type)
    return base, gaussian_count, flare_count


def resolve_transit_config(transit_cfg: dict, time_ref_mjd: float) -> dict:
    resolved = dict(transit_cfg)
    fixed = dict(transit_cfg.get("fixed", transit_cfg))
    if "t0_hours" not in fixed:
        if "t0_mjd" in fixed:
            fixed["t0_hours"] = (float(fixed["t0_mjd"]) - float(time_ref_mjd)) * 24.0
        elif "t0_days" in fixed:
            fixed["t0_hours"] = float(fixed["t0_days"]) * 24.0
    if "period_hours" not in fixed:
        if "period_days" in fixed:
            fixed["period_hours"] = float(fixed["period_days"]) * 24.0
        elif "per_days" in fixed:
            fixed["period_hours"] = float(fixed["per_days"]) * 24.0
    resolved["fixed"] = fixed
    resolved["time_ref_mjd"] = float(time_ref_mjd)
    return resolved


def detec_parameter_count(model_type: str) -> int:
    base, gaussian_count, flare_count = split_additive_suffixes(model_type)
    if base not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported detrending model: {model_type}")
    flare_parameters = 5 * flare_count
    return SUPPORTED_MODELS[base] + 3 * gaussian_count + flare_parameters + (6 if "_2nd_order_centroid" in model_type else 0)


def default_initial_guess(model_type: str) -> list[float]:
    base, gaussian_count, flare_count = split_additive_suffixes(model_type)
    seeds = {
        "offset": [1.0],
        "exp": [0.0, 0.0, 1.0],
        "linear": [0.0, 1.0],
        "polynomial": [0.0, 0.0, 0.0, 1.0],
        "double_exp": [0.5, 0.0, 0.5, 0.0],
        "exp+linear": [0.0, 0.0, 1.0, 0.0, 1.0],
        "exp+polynomial": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "linear+polynomial": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    if base not in seeds:
        raise ValueError(f"Unsupported detrending model: {model_type}")
    guess = list(seeds[base])
    guess.extend([0.0, 0.0, 1.0] * gaussian_count)
    if flare_count == 1:
        guess.extend([0.0, 0.0, 0.0, 0.1, 1.0])
    elif flare_count == 2:
        guess.extend([0.0, 0.0, 0.0, 0.1, 1.0] * 2)
    if "_2nd_order_centroid" in model_type:
        guess = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] + guess
    return guess


def default_labels(model_type: str) -> list[str]:
    n = detec_parameter_count(model_type)
    labels = [f"c_{i}" for i in range(1, n + 1)]
    return labels


def detrending_parameter_names(model_type: str) -> list[str]:
    """Return stable, descriptive names in the exact order consumed by detec_model."""
    base, gaussian_count, flare_count = split_additive_suffixes(model_type)
    base_names = {
        "offset": ["offset"],
        "exp": ["exp_amplitude", "exp_rate", "exp_offset"],
        "linear": ["linear_slope", "linear_intercept"],
        "polynomial": ["poly_cubic", "poly_quadratic", "poly_linear", "poly_intercept"],
        "double_exp": ["exp1_amplitude", "exp1_rate", "exp2_amplitude", "exp2_rate"],
        "exp+linear": ["exp_amplitude", "exp_rate", "exp_offset", "linear_slope", "linear_intercept"],
        "exp+polynomial": ["exp_amplitude", "exp_rate", "exp_offset", "poly_cubic", "poly_quadratic", "poly_linear", "poly_intercept"],
        "linear+polynomial": ["linear_slope", "linear_intercept", "poly_cubic", "poly_quadratic", "poly_linear", "poly_intercept"],
    }
    names = list(base_names[base])
    for index in range(1, gaussian_count + 1):
        names.extend([f"gaussian{index}_amplitude", f"gaussian{index}_center", f"gaussian{index}_sigma"])
    for index in range(1, flare_count + 1):
        names.extend([
            f"flare{index}_f0", f"flare{index}_amplitude", f"flare{index}_t_peak",
            f"flare{index}_tau_rise", f"flare{index}_tau_decay",
        ])
    if "_2nd_order_centroid" in model_type:
        names = [
            "centroid_constant", "centroid_x", "centroid_y", "centroid_x2",
            "centroid_xy", "centroid_y2",
        ] + names
    if len(names) != detec_parameter_count(model_type):
        raise RuntimeError(f"Internal parameter-name mismatch for {model_type}")
    return names


def transit_model(time_hours: np.ndarray, t_s_hours: float, fp: float, transit_cfg: dict) -> np.ndarray:
    if batman is None:
        raise ImportError("batman-package is required for the transit model")

    fixed = transit_cfg.get("fixed", transit_cfg)
    params = batman.TransitParams()
    if "t0_hours" in fixed:
        params.t0 = float(fixed["t0_hours"])
    elif "t0_days" in fixed:
        params.t0 = float(fixed["t0_days"]) * 24.0
    else:
        params.t0 = float(fixed["t0_mjd"]) * 24.0

    if "period_hours" in fixed:
        params.per = float(fixed["period_hours"])
    elif "period_days" in fixed:
        params.per = float(fixed["period_days"]) * 24.0
    else:
        params.per = float(fixed["per_days"]) * 24.0
    params.rp = float(fixed["rp"])
    params.a = float(fixed["a"])
    params.inc = float(fixed["inc"])
    params.ecc = float(fixed.get("ecc", 0.0))
    params.w = float(fixed.get("w", 90.0))
    params.limb_dark = str(fixed.get("limb_dark", "quadratic"))
    params.u = list(fixed.get("u", [0.1, 0.1]))
    params.fp = float(fp)
    params.t_secondary = float(t_s_hours)
    model = batman.TransitModel(params, np.asarray(time_hours, dtype=float), transittype="secondary")
    return np.asarray(model.light_curve(params), dtype=float)


def exponential_func(time, a, b, c):
    return a * np.exp(-b * time) + c


def linear_slope(time, m, b):
    return m * time + b


def poly_3rd_degree(time, a, b, c, d):
    return a * time**3 + b * time**2 + c * time + d


def double_exponential(time, c1, c2, c3, c4):
    return c1 * np.exp(-c2 * time) + c3 * np.exp(-c4 * time)


def gaussian_func(time, amplitude, center, sigma):
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.full_like(np.asarray(time, dtype=float), np.nan)
    return amplitude * np.exp(-0.5 * ((np.asarray(time, dtype=float) - center) / sigma) ** 2)


def stellar_flare_func(time, f0, amplitude, t_peak, tau_rise, tau_decay):
    """Asymmetric exponential flare with a fixed peak time."""
    time = np.asarray(time, dtype=float)
    if not np.isfinite(tau_rise) or not np.isfinite(tau_decay) or tau_rise <= 0.0 or tau_decay <= 0.0:
        return np.full_like(time, np.nan)
    return np.where(
        time < t_peak,
        f0 + amplitude * np.exp(-(t_peak - time) / tau_rise),
        f0 + amplitude * np.exp(-(time - t_peak) / tau_decay),
    )


def detec_model_poly(xdata, ydata, c1, c2, c3, c4, c5, c6):
    x = np.asarray(xdata, dtype=float)
    y = np.asarray(ydata, dtype=float)
    pos = np.vstack((np.ones_like(x), x, y, x**2, x * y, y**2))
    coeffs = np.array([c1, c2, c3, c4, c5, c6], dtype=float)
    return np.dot(coeffs[np.newaxis, :], pos).reshape(-1)


def detec_model(time, centroid_x, centroid_y, theta, model_type, model_config=None):
    theta = np.asarray(theta, dtype=float)
    base_model, gaussian_count, flare_count = split_additive_suffixes(model_type)
    centroid_model = "_2nd_order_centroid" in model_type
    detec_centroid = None

    if centroid_model:
        detec_centroid = detec_model_poly(
            centroid_x, centroid_y, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
        )
        theta = theta[6:]

    base_parameter_count = SUPPORTED_MODELS.get(base_model)
    if base_parameter_count is None:
        raise ValueError(f"Unsupported detrending model: {model_type}")
    flare_parameter_count = 5 * flare_count
    expected = base_parameter_count + 3 * gaussian_count + flare_parameter_count
    if (gaussian_count or flare_count) and theta.size != expected:
        raise ValueError(
            f"Detrending model '{model_type}' expects {expected} parameters; got {theta.size}"
        )
    base_theta = theta[:base_parameter_count]
    gaussian_end = base_parameter_count + 3 * gaussian_count
    gaussian_theta = theta[base_parameter_count:gaussian_end]
    flare_theta = theta[gaussian_end:]

    if base_model == "offset":
        detec = np.full_like(time, base_theta[0], dtype=float)
    elif base_model == "exp+linear":
        detec = exponential_func(time, *base_theta[0:3]) * linear_slope(time, *base_theta[3:5])
    elif base_model == "exp":
        detec = exponential_func(time, *base_theta[0:3])
    elif base_model == "linear":
        detec = linear_slope(time, *base_theta[0:2])
    elif base_model == "polynomial":
        detec = poly_3rd_degree(time, *base_theta[0:4])
    elif base_model == "exp+polynomial":
        detec = exponential_func(time, *base_theta[0:3]) * poly_3rd_degree(time, *base_theta[3:7])
    elif base_model == "linear+polynomial":
        detec = linear_slope(time, *base_theta[0:2]) * poly_3rd_degree(time, *base_theta[2:6])
    elif base_model == "double_exp":
        detec = double_exponential(time, *base_theta[0:4])
    else:
        raise ValueError(f"Unsupported detrending model: {model_type}")

    for index in range(gaussian_count):
        amplitude, center, sigma = gaussian_theta[3 * index : 3 * index + 3]
        detec = detec + gaussian_func(time, amplitude, center, sigma)

    if flare_count == 1:
        f0, amplitude, t_peak, tau_rise, tau_decay = flare_theta
        detec = detec + stellar_flare_func(
            time, f0, amplitude, t_peak, tau_rise, tau_decay,
        )
    elif flare_count == 2:
        for index in range(2):
            f0, amplitude, t_peak, tau_rise, tau_decay = flare_theta[
                5 * index : 5 * index + 5
            ]
            detec = detec + stellar_flare_func(
                time, f0, amplitude, t_peak, tau_rise, tau_decay,
            )

    if detec_centroid is not None:
        detec = detec * detec_centroid
    return np.asarray(detec, dtype=float)


def signal(time, centroid_x, centroid_y, theta, model_type, transit_cfg, detrending_config=None):
    theta = np.asarray(theta, dtype=float)
    t_s = theta[0]
    fp = theta[1]
    astro = transit_model(time, t_s, fp, transit_cfg)
    detec = detec_model(
        time, centroid_x, centroid_y, theta[2:-1], model_type, detrending_config
    )
    return astro * detec


def split_theta(theta: np.ndarray, model_type: str) -> tuple[float, float, np.ndarray, float]:
    theta = np.asarray(theta, dtype=float)
    return float(theta[0]), float(theta[1]), theta[2:-1], float(theta[-1])
