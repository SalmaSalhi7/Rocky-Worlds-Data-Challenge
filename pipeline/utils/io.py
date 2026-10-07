from __future__ import annotations

from pathlib import Path
import json
import os

import h5py
import numpy as np

from .config import dump_yaml


def inject_eclipse(data: dict[str, np.ndarray], config: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Inject a multiplicative secondary eclipse into a preprocessed light curve."""
    injection = config.get("injection", {})
    enabled = injection.get("inject_eclipse", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("injection.inject_eclipse must be true or false")
    if not enabled:
        return data, {"inject_eclipse": False}

    missing = [name for name in ("depth_ppm", "time_mjd") if name not in injection]
    if missing:
        raise ValueError(
            "An enabled eclipse injection requires injection."
            + " and injection.".join(missing)
        )
    depth_ppm = float(injection["depth_ppm"])
    time_mjd_injection = float(injection["time_mjd"])
    if not np.isfinite(depth_ppm) or depth_ppm < 0.0:
        raise ValueError("injection.depth_ppm must be a finite, non-negative number")
    if not np.isfinite(time_mjd_injection):
        raise ValueError("injection.time_mjd must be finite")

    time_mjd = np.asarray(data["time_mjd"], dtype=float)
    if not np.nanmin(time_mjd) <= time_mjd_injection <= np.nanmax(time_mjd):
        raise ValueError(
            f"injection.time_mjd={time_mjd_injection} is outside the data range "
            f"[{np.nanmin(time_mjd)}, {np.nanmax(time_mjd)}] MJD"
        )

    # Import locally to keep the general I/O helpers usable without batman.
    from .models import resolve_transit_config, transit_model

    time_ref = float(data.get("time_ref_mjd", time_mjd[0]))
    transit_cfg = resolve_transit_config(config["transit"], time_ref)
    injection_time_hours = (time_mjd_injection - time_ref) * 24.0
    depth_fraction = depth_ppm * 1.0e-6
    if depth_fraction >= 1.0:
        raise ValueError("injection.depth_ppm must be less than 1,000,000 ppm")
    # BATMAN's secondary-eclipse parameter is the planet/star flux ratio and
    # its uneclipsed baseline is 1 + fp. Convert the requested fractional
    # drop to fp, then normalize so the out-of-eclipse injected profile is 1.
    planet_flux_ratio = depth_fraction / (1.0 - depth_fraction)
    profile = transit_model(
        data["time_hours"], injection_time_hours, planet_flux_ratio, transit_cfg
    ) / (1.0 + planet_flux_ratio)

    injected = dict(data)
    injected["flux"] = np.asarray(data["flux"], dtype=float) * profile
    injected["injected_eclipse_model"] = profile
    metadata = {
        "inject_eclipse": True,
        "depth_ppm": depth_ppm,
        "depth_fraction": depth_fraction,
        "planet_flux_ratio": planet_flux_ratio,
        "time_mjd": time_mjd_injection,
        "time_hours": injection_time_hours,
    }
    return injected, metadata


def find_specdata_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.glob("*SpecData.h5"))
        if matches:
            return matches[0]
        matches = sorted(path.glob("*.h5"))
        if matches:
            return matches[0]
        matches = sorted(path.glob("*.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate a data file at or under: {path}")


def load_photometry_dataset(
    path: str | Path, require_centroids: bool = True,
) -> dict[str, np.ndarray]:
    file_path = find_specdata_file(path)
    suffix = file_path.suffix.lower()

    if suffix in {".h5", ".hdf5"}:
        with h5py.File(file_path, "r") as fh:
            def _get(name: str) -> np.ndarray:
                if name not in fh:
                    raise KeyError(f"Missing dataset '{name}' in {file_path}")
                return np.asarray(fh[name])

            flux_pair = next(
                (
                    (flux_name, error_name)
                    for flux_name, error_name in (
                        ("flux", "flux_err"),
                        ("aplev", "aperr"),
                    )
                    if flux_name in fh and error_name in fh
                ),
                None,
            )
            if flux_pair is None:
                available = ", ".join(sorted(fh.keys())) or "none"
                raise KeyError(
                    f"Could not find a complete flux/error pair in {file_path}. "
                    "Expected either ('flux', 'flux_err') or ('aplev', 'aperr'). "
                    f"Available top-level datasets: {available}"
                )
            flux_name, error_name = flux_pair
            time = _get("time")
            data = {
                "time": time,
                "flux": _get(flux_name),
                "flux_err": _get(error_name),
            }
            for centroid_name in ("centroid_x", "centroid_y"):
                if centroid_name in fh:
                    data[centroid_name] = np.asarray(fh[centroid_name])
                elif require_centroids:
                    raise KeyError(f"Missing dataset '{centroid_name}' in {file_path}")
                else:
                    data[centroid_name] = np.zeros_like(time, dtype=float)
        return data

    if suffix == ".csv":
        import csv

        with file_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"No rows found in {file_path}")
        columns = rows[0].keys()
        wanted = {
            "time": ["time", "int_mid_BJD_TDB"],
            "flux": ["flux", "amps", "aplev"],
            "flux_err": ["flux_err", "amps_err", "aperr"],
            "centroid_x": ["centroid_x", "dxs"],
            "centroid_y": ["centroid_y", "dys"],
        }
        out = {}
        for key, candidates in wanted.items():
            for candidate in candidates:
                if candidate in columns:
                    out[key] = np.asarray([float(row[candidate]) for row in rows], dtype=float)
                    break
            else:
                if key in {"centroid_x", "centroid_y"} and not require_centroids:
                    out[key] = np.zeros(len(rows), dtype=float)
                else:
                    raise KeyError(f"Could not find a column for '{key}' in {file_path}")
        return out

    raise ValueError(f"Unsupported data format: {file_path}")


def sigma_clip_mask(values: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=bool)
    median = np.nanmedian(values[finite])
    mad = np.nanmedian(np.abs(values[finite] - median))
    if not np.isfinite(mad) or mad == 0.0:
        std = np.nanstd(values[finite])
        if not np.isfinite(std) or std == 0.0:
            return np.zeros_like(values, dtype=bool)
        return np.abs(values - median) > sigma * std
    robust_std = 1.4826 * mad
    return np.abs(values - median) > sigma * robust_std


def preprocess_dataset(
    data: dict[str, np.ndarray], normalize: str = "median",
    sigma_clip: float | None = 4.0, time_coordinate: str = "mjd",
    phase_reference_mjd: float | None = None,
) -> dict[str, np.ndarray]:
    time = np.asarray(data["time"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    flux_err = np.asarray(data["flux_err"], dtype=float)
    centroid_x = np.asarray(data["centroid_x"], dtype=float)
    centroid_y = np.asarray(data["centroid_y"], dtype=float)

    keep = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & np.isfinite(centroid_x) & np.isfinite(centroid_y)
    if sigma_clip is not None:
        keep &= ~sigma_clip_mask(flux, sigma=sigma_clip)

    time = time[keep]
    flux = flux[keep]
    flux_err = flux_err[keep]
    centroid_x = centroid_x[keep]
    centroid_y = centroid_y[keep]

    if normalize == "median":
        scale = float(np.nanmedian(flux))
    elif normalize == "mean":
        scale = float(np.nanmean(flux))
    elif normalize in {"none", "", None}:
        scale = 1.0
    else:
        raise ValueError(f"Unsupported normalization mode: {normalize}")

    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    time_coordinate = str(time_coordinate).strip().lower()
    if time_coordinate == "mjd":
        time_mjd = time
        time_hours = (time - time[0]) * 24.0
        time_ref_mjd = float(time[0])
    elif time_coordinate == "phase_hours":
        if phase_reference_mjd is None or not np.isfinite(float(phase_reference_mjd)):
            raise ValueError("phase_hours data require a finite phase_reference_mjd")
        time_hours = time
        time_ref_mjd = float(phase_reference_mjd)
        time_mjd = time_ref_mjd + time_hours / 24.0
    else:
        raise ValueError("time_coordinate must be 'mjd' or 'phase_hours'")

    return {
        "time_mjd": time_mjd,
        "time_hours": time_hours,
        "time_ref_mjd": time_ref_mjd,
        "time_coordinate": time_coordinate,
        "flux": flux / scale,
        "flux_err": flux_err / scale,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "scale": scale,
        "n_points": int(time.size),
    }


def save_jsonable(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def save_corrected_photometry_h5(
    path: str | Path,
    time_mjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    detrending_model: np.ndarray,
) -> Path:
    """Save detrending-corrected photometry using the fitted deterministic model."""
    path = Path(path)
    time_mjd = np.asarray(time_mjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    detrending_model = np.asarray(detrending_model, dtype=float)
    if not (time_mjd.shape == flux.shape == flux_err.shape == detrending_model.shape):
        raise ValueError("Time, flux, errors, and detrending model must have matching shapes")
    valid_model = np.isfinite(detrending_model) & (detrending_model != 0.0)
    corrected_flux = np.divide(
        flux, detrending_model, out=np.full_like(flux, np.nan), where=valid_model
    )
    corrected_flux_err = np.divide(
        flux_err, np.abs(detrending_model),
        out=np.full_like(flux_err, np.nan), where=valid_model,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        time_dataset = handle.create_dataset("time", data=time_mjd)
        flux_dataset = handle.create_dataset("flux", data=corrected_flux)
        error_dataset = handle.create_dataset("flux_err", data=corrected_flux_err)
        time_dataset.attrs["units"] = "MJD"
        flux_dataset.attrs["description"] = "Normalized flux divided by the fitted detrending model"
        error_dataset.attrs["description"] = "Flux uncertainty divided by the absolute fitted detrending model"
        handle.attrs["correction"] = "deterministic detrending model; Gaussian-process prediction not subtracted"
    return path


def save_results_bundle(output_dir: str | Path, summary: dict, config: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_dir / "config_resolved.yaml")
    save_jsonable(summary, output_dir / "summary.json")
