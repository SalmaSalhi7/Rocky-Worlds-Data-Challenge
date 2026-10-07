from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def eclipse_detection_sigma(depth, upper_error, lower_error) -> float:
    """Approximate distance from zero using the asymmetric error toward zero."""
    depth = float(depth)
    upper_error = float(upper_error)
    lower_error = float(lower_error)
    toward_zero_error = lower_error if depth >= 0.0 else upper_error
    if not np.isfinite(depth) or not np.isfinite(toward_zero_error) or toward_zero_error <= 0.0:
        return np.nan
    return abs(depth) / toward_zero_error


def plot_corrected_flux(
    time, flux, flux_err, detec, astro, residuals, fitted_t_s, eclipse_label,
    output_path, expected_t_s=None,
):
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    detec = np.asarray(detec, dtype=float)
    astro = np.asarray(astro, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    corrected = np.divide(flux, detec, out=np.full_like(flux, np.nan), where=np.isfinite(detec) & (detec != 0))

    ax[0].errorbar(time, flux, yerr=flux_err, fmt="o", ms=3, color="grey", alpha=0.6, label="Data")
    ax[0].errorbar(time, corrected, yerr=np.divide(flux_err, np.abs(detec), out=np.full_like(flux_err, np.nan), where=np.isfinite(detec) & (detec != 0)), fmt="o", ms=3, color="red", alpha=0.6, label="Corrected")
    ax[0].plot(time, astro, color="black", lw=2.5, label="Astro model")
    ax[0].plot(time, detec, color="tab:gray", lw=2.0, label="Detrending model")
    if expected_t_s is not None:
        ax[0].axvline(expected_t_s, color="tab:orange", ls="--", lw=1.8, label="Expected mid-eclipse")
        ax[1].axvline(expected_t_s, color="tab:orange", ls="--", lw=1.8)
    ax[0].axvline(fitted_t_s, color="tab:blue", ls="-", lw=1.8, label="Fitted mid-eclipse")
    ax[1].axvline(fitted_t_s, color="tab:blue", ls="-", lw=1.8)
    ax[0].set_title(str(eclipse_label))
    ax[0].legend(loc="best")

    ax[1].scatter(time, residuals, color="black", s=12, alpha=0.6)
    ax[1].axhline(0.0, color="red", lw=1.5)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Residuals")
    ax[0].set_ylabel("Normalized flux")

    _save(fig, output_path)


def plot_rednoise(residuals, output_path):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 4:
        return
    bins = np.arange(1, min(200, residuals.size) + 1)
    rms = []
    stderr = []
    for n in bins:
        nb = residuals.size // n
        if nb < 2:
            continue
        trimmed = residuals[: nb * n].reshape(nb, n)
        binned = np.nanmean(trimmed, axis=1)
        rms.append(np.nanstd(binned))
        stderr.append(np.nanstd(residuals) / np.sqrt(n))
    if not rms:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(bins[: len(rms)], rms, label="Binned RMS")
    ax.loglog(bins[: len(stderr)], stderr, label="White-noise expectation")
    ax.set_xlabel("Bin size")
    ax.set_ylabel("RMS")
    ax.legend(loc="best")
    _save(fig, output_path)


def plot_gaussian_process_fit(
    time, flux, flux_err, detrending_model, eclipse_detrending_model,
    gp_prediction, output_path,
):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    detrending_model = np.asarray(detrending_model, dtype=float)
    eclipse_detrending_model = np.asarray(eclipse_detrending_model, dtype=float)
    gp_prediction = np.asarray(gp_prediction, dtype=float)
    full_model = eclipse_detrending_model + gp_prediction
    residuals = flux - full_model
    finite = residuals[np.isfinite(residuals)]
    rms_ppm = float(np.sqrt(np.mean(finite**2)) * 1e6) if finite.size else np.nan

    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].errorbar(
        time, flux, yerr=flux_err, fmt="o", ms=3, color="0.45", alpha=0.45,
        label="Flux data",
    )
    ax[0].plot(time, detrending_model, color="0.35", ls=":", lw=2.0,
               label="Detrending baseline only")
    ax[0].plot(time, eclipse_detrending_model, color="tab:orange", ls="--", lw=2.0,
               label="Detrending × fitted eclipse")
    ax[0].plot(time, full_model, color="tab:blue", lw=2.2,
               label="Detrending × fitted eclipse + GP")
    ax[0].text(
        0.03, 0.06, f"Post-GP residual RMS = {rms_ppm:.1f} ppm",
        transform=ax[0].transAxes, fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    ax[0].set_ylabel("Normalized flux")
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    ax[1].scatter(time, residuals, color="black", s=12, alpha=0.55)
    ax[1].axhline(0.0, color="tab:red", lw=1.5)
    ax[1].set_xlabel("Time (hours)")
    ax[1].set_ylabel("Residuals")
    _save(fig, output_path)


def plot_walkers(chain, labels, output_path):
    chain = np.asarray(chain)
    nwalkers, nsteps, ndim = chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(12, 2.0 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chain[:, :, i].T, alpha=0.25, lw=0.6)
        axes[i].set_ylabel(labels[i] if i < len(labels) else f"p{i}")
    axes[-1].set_xlabel("Step")
    _save(fig, output_path)


def plot_corner(samples, labels, output_path):
    try:
        import corner
    except Exception:
        return
    fig = corner.corner(samples, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84], plot_datapoints=False, title_fmt=".6f")
    _save(fig, output_path)


def plot_binned_fit(
    time, flux, flux_err, detec, astro, output_path, bins=100,
    expected_t_s=None, fitted_t_s=None, eclipse_depth_summary=None,
):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    detec = np.asarray(detec, dtype=float)
    astro = np.asarray(astro, dtype=float)
    if time.size < 4:
        return
    bins = max(2, min(int(bins), time.size))
    edges = np.linspace(np.nanmin(time), np.nanmax(time), bins + 1)
    idx = np.digitize(time, edges) - 1
    xs, ys, yerr = [], [], []
    for i in range(bins):
        mask = idx == i
        if not mask.any():
            continue
        xs.append(np.nanmedian(time[mask]))
        ys.append(np.nanmedian(flux[mask] / detec[mask]))
        yerr.append(np.nanmean(flux_err[mask]))
    if not xs:
        return
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})
    corrected_residuals = flux / detec - astro
    finite_residuals = corrected_residuals[np.isfinite(corrected_residuals)]
    residual_rms_ppm = (
        float(np.sqrt(np.mean(finite_residuals**2))) * 1e6
        if finite_residuals.size else np.nan
    )
    ax[0].errorbar(time, flux / detec, yerr=flux_err / np.abs(detec), fmt="o", ms=3, alpha=0.25, label="Corrected data")
    ax[0].plot(time, astro, color="black", lw=2.5, label="Astro model")
    ax[0].errorbar(xs, ys, yerr=yerr, fmt="o", ms=4, color="tab:red", alpha=0.8, label="Binned data")
    if expected_t_s is not None:
        ax[0].axvline(expected_t_s, color="tab:orange", ls="--", lw=1.8, label="Expected mid-eclipse")
        ax[1].axvline(expected_t_s, color="tab:orange", ls="--", lw=1.8)
    if fitted_t_s is not None:
        ax[0].axvline(fitted_t_s, color="tab:blue", ls="-", lw=1.8, label="Fitted mid-eclipse")
        ax[1].axvline(fitted_t_s, color="tab:blue", ls="-", lw=1.8)
    annotation_lines = [rf"Residual RMS = {residual_rms_ppm:.1f} ppm"]
    if eclipse_depth_summary is not None:
        depth, upper_error, lower_error = np.asarray(eclipse_depth_summary, dtype=float)
        detection_sigma = eclipse_detection_sigma(depth, upper_error, lower_error)
        depth_label = (
            rf"Eclipse depth = {depth * 1e6:.1f}"
            rf"$^{{+{upper_error * 1e6:.1f}}}_{{-{lower_error * 1e6:.1f}}}$ ppm"
        )
        annotation_lines.insert(0, depth_label)
        significance_label = (
            rf"Detection significance = {detection_sigma:.2f}$\sigma$"
            if np.isfinite(detection_sigma)
            else "Detection significance = unavailable"
        )
        annotation_lines.insert(1, significance_label)
    ax[0].text(
        0.03, 0.06, "\n".join(annotation_lines), transform=ax[0].transAxes, fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax[1].scatter(time, corrected_residuals, color="black", s=12, alpha=0.5)
    ax[1].axhline(0.0, color="red", lw=1.5)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Residuals")
    _save(fig, output_path)
