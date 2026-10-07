from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.utils.models import detrending_parameter_names, detec_model, transit_model
try:
    from step3_joint_fit.joint_mcmc import joint_split_theta, mjd_to_observation_hours, reconstruct_configured_joint, transit_config_for_eclipse
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from joint_mcmc import joint_split_theta, mjd_to_observation_hours, reconstruct_configured_joint, transit_config_for_eclipse


def _flare_peaks_bjd_to_hours(theta, model_type, time_ref_mjd):
    """Convert fitted BJD flare peaks for the legacy hours-based plot model."""
    converted = np.asarray(theta, dtype=float).copy()
    parameter_names = detrending_parameter_names(model_type)
    if len(parameter_names) != converted.size:
        raise ValueError(
            f"Detrending model {model_type!r} expects {len(parameter_names)} "
            f"parameters, but received {converted.size}"
        )
    for index, name in enumerate(parameter_names):
        if name.endswith("_t_peak"):
            converted[index] = (converted[index] - float(time_ref_mjd)) * 24.0
    return converted


def _binned(x: np.ndarray, y: np.ndarray, yerr: np.ndarray, bins: int):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), max(2, bins) + 1)
    which = np.digitize(x, edges) - 1
    xb, yb, eb = [], [], []
    for index in range(len(edges) - 1):
        keep = which == index
        if not np.any(keep):
            continue
        weights = np.divide(1.0, yerr[keep] ** 2, out=np.zeros(np.sum(keep)), where=yerr[keep] > 0)
        xb.append(np.nanmedian(x[keep]))
        if np.sum(weights) > 0:
            yb.append(np.sum(weights * y[keep]) / np.sum(weights))
            eb.append(np.sqrt(1.0 / np.sum(weights)))
        else:
            yb.append(np.nanmedian(y[keep]))
            eb.append(np.nanstd(y[keep]) / np.sqrt(np.sum(keep)))
    return np.asarray(xb), np.asarray(yb), np.asarray(eb)


def _plot_window(x, y, yerr, xlim):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    if xlim is not None:
        finite &= (x >= float(xlim[0])) & (x <= float(xlim[1]))
    return x[finite], y[finite], yerr[finite]


def _joint_plot_products(datasets, eclipse_specs, config, fit_result, xlim):
    """Return detrended, phase-aligned data and fitted eclipse curves."""
    if config.get("parameters") is not None:
        fp, parts, shared_values, _system = reconstruct_configured_joint(
            fit_result["best"], config, eclipse_specs, datasets
        )
    else:
        fp, parts = joint_split_theta(fit_result["best"], eclipse_specs)
        shared_values = None
    fixed = config["transit"].get("fixed", config["transit"])
    period_hours = (
        float(shared_values["period_days"]) * 24.0 if shared_values is not None
        else float(fixed.get("period_hours", float(fixed["period_days"]) * 24.0))
    )

    products = []
    for spec, dataset in zip(eclipse_specs, datasets):
        part = parts[spec["eclipse_number"]]
        t_secondary_hours = mjd_to_observation_hours(
            part["t0_mjd"], dataset["time_mjd"][0]
        )
        detrending_parameters = _flare_peaks_bjd_to_hours(
            part["detec"], spec["detrending"]["model_type"],
            dataset["time_mjd"][0],
        )
        detrending = detec_model(
            dataset["time_hours"], dataset["centroid_x"], dataset["centroid_y"],
            detrending_parameters, spec["detrending"]["model_type"], spec["detrending"],
        )
        valid = (
            np.isfinite(detrending) & (detrending != 0)
            & np.isfinite(dataset["flux"]) & np.isfinite(dataset["flux_err"])
        )
        phase = 0.5 + (
            dataset["time_hours"][valid] - t_secondary_hours
        ) / period_hours
        corrected_flux = dataset["flux"][valid] / detrending[valid]
        corrected_error = dataset["flux_err"][valid] / np.abs(detrending[valid])
        transit_config = part.get("transit_config")
        if transit_config is None:
            transit_config = transit_config_for_eclipse(config, part["ecc"])
        transit_config["time_ref_mjd"] = float(dataset["time_mjd"][0])
        fitted_at_data = transit_model(
            dataset["time_hours"][valid], t_secondary_hours, fp, transit_config
        )
        residual_rms = np.sqrt(np.nanmean((corrected_flux - fitted_at_data) ** 2))
        if xlim is None:
            grid = np.linspace(np.nanmin(phase), np.nanmax(phase), 600)
        else:
            grid = np.linspace(float(xlim[0]), float(xlim[1]), 600)
        fitted_grid = transit_model(
            (grid - 0.5) * period_hours, 0.0, fp, transit_config
        )
        products.append({
            "eclipse_number": spec["eclipse_number"],
            "phase": phase,
            "flux": corrected_flux,
            "error": corrected_error,
            "grid": grid,
            "model": fitted_grid,
            "residual_rms": residual_rms,
        })
    return fp, products


def plot_combined_joint_fit(datasets, eclipse_specs, config, fit_result, output_path):
    xlim = config.get("plots", {}).get("combined_xlim")
    fp, products = _joint_plot_products(
        datasets, eclipse_specs, config, fit_result, xlim
    )
    windowed = [
        _plot_window(item["phase"], item["flux"], item["error"], xlim)
        for item in products
    ]
    x = np.concatenate([item[0] for item in windowed])
    y = np.concatenate([item[1] for item in windowed])
    yerr = np.concatenate([item[2] for item in windowed])
    order = np.argsort(x)
    x, y, yerr = x[order], y[order], yerr[order]
    xb, yb, eb = _binned(x, y, yerr, int(config.get("plots", {}).get("combined_n_bins", 100)))

    depth = (
        fit_result["summary"][fit_result["labels"].index("fp")]
        if "fp" in fit_result["labels"] else np.asarray([fp, 0.0, 0.0])
    )
    depth_text = (
        rf"Eclipse depth = {depth[0] * 1e6:.1f}"
        rf"$^{{+{depth[1] * 1e6:.1f}}}_{{-{depth[2] * 1e6:.1f}}}$ ppm"
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(
        x, y, yerr=yerr, fmt=".", ms=3, color="0.65", alpha=0.18,
        label="All corrected flux", zorder=1,
    )
    ax.errorbar(
        xb, yb, yerr=eb, fmt="o", ms=5, color="tab:blue",
        label="Combined binned flux", zorder=2,
    )
    common_grid = products[0]["grid"]
    combined_model = np.nanmedian(
        np.vstack([
            np.interp(common_grid, item["grid"], item["model"])
            for item in products
        ]),
        axis=0,
    )
    ax.plot(
        common_grid, combined_model, color="black", lw=2.5,
        label="Combined fitted eclipse", zorder=10,
    )
    ax.axvline(0.5, color="0.35", ls="--", lw=1.2, zorder=0)
    ax.text(0.03, 0.05, depth_text, transform=ax.transAxes, fontsize=13,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"})
    ax.set_xlabel("Orbital phase (aligned on each fitted eclipse midtime)")
    ax.set_ylabel("Detrended normalized flux")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.legend(loc="best")
    fig.savefig(Path(output_path), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_stacked_joint_fit(datasets, eclipse_specs, config, fit_result, output_path):
    """Plot each phase-aligned, binned eclipse with a configurable offset."""
    plot_cfg = config.get("plots", {})
    xlim = plot_cfg.get("stacked_xlim", plot_cfg.get("combined_xlim"))
    _fp, products = _joint_plot_products(
        datasets, eclipse_specs, config, fit_result, xlim
    )
    bins = int(plot_cfg.get("stacked_n_bins", plot_cfg.get("combined_n_bins", 100)))
    vertical_offset = float(plot_cfg.get("stacked_vertical_offset", 0.0015))
    if not np.isfinite(vertical_offset) or vertical_offset < 0:
        raise ValueError("plots.stacked_vertical_offset must be finite and non-negative")

    height = max(7.0, 1.05 * len(products))
    fig, ax = plt.subplots(figsize=(12, height))
    for index, item in enumerate(products):
        offset = index * vertical_offset
        phase, flux, error = _plot_window(
            item["phase"], item["flux"], item["error"], xlim
        )
        xb, yb, eb = _binned(phase, flux, error, bins)
        ax.errorbar(
            xb, yb + offset, yerr=eb, fmt="o", ms=4.2,
            color="black", ecolor="0.55", elinewidth=0.8, capsize=0,
            zorder=2,
        )
        ax.plot(
            item["grid"], item["model"] + offset,
            color="tab:red", lw=2.0, zorder=10,
        )
        text_x = item["grid"][-1] - 0.01 * np.ptp(item["grid"])
        ax.text(
            text_x, 1.0 + offset,
            f"Eclipse {item['eclipse_number']}  |  "
            f"residual RMS = {item['residual_rms'] * 1e6:.1f} ppm",
            ha="right", va="bottom", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            zorder=11,
        )

    ax.axvline(0.5, color="0.35", ls="--", lw=1.2, zorder=0)
    ax.set_xlabel("Orbital phase (aligned on each fitted eclipse midtime)")
    ax.set_ylabel("Detrended normalized flux + offset")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_title(
        f"Joint-fit eclipses (vertical offset = {vertical_offset:g})"
    )
    fig.savefig(Path(output_path), bbox_inches="tight", dpi=200)
    plt.close(fig)
