from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validate_uniform_bounds(labels, bounds):
    """Validate proper finite uniform priors and return a float array."""
    labels = list(labels)
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (len(labels), 2):
        raise ValueError(
            f"Evidence bounds must have shape ({len(labels)}, 2); got {bounds.shape}"
        )
    for label, (lower, upper) in zip(labels, bounds):
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise ValueError(
                f"Bayesian evidence requires a finite proper prior for {label}; "
                f"got [{lower}, {upper}]"
            )
    return bounds


def _normalized_prior_specs(labels, bounds, prior_specs=None):
    """Validate sampler priors, defaulting to the legacy uniform bounds."""
    if prior_specs is None:
        return [
            {"type": "uniform", "lower": float(lower), "upper": float(upper)}
            for lower, upper in bounds
        ]
    if len(prior_specs) != len(labels):
        raise ValueError(
            f"Expected {len(labels)} prior specifications; got {len(prior_specs)}"
        )
    normalized = []
    for label, bound, prior in zip(labels, bounds, prior_specs):
        prior = dict(prior)
        prior_type = str(prior.get("type", "uniform")).lower()
        lower = float(prior.get("lower", bound[0]))
        upper = float(prior.get("upper", bound[1]))
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise ValueError(f"Prior bounds for {label} must be finite with lower < upper")
        if prior_type == "uniform":
            normalized.append({"type": "uniform", "lower": lower, "upper": upper})
        elif prior_type in {"gaussian", "normal"}:
            mean = float(prior["mean"])
            sigma = float(prior["sigma"])
            if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"Gaussian prior for {label} requires finite mean and sigma > 0")
            normalized.append({
                "type": "gaussian", "mean": mean, "sigma": sigma,
                "lower": lower, "upper": upper,
            })
        else:
            raise ValueError(f"Unsupported prior type '{prior_type}' for {label}")
    return normalized


def _run_nested(log_likelihood, labels, bounds, config, args=(), prior_specs=None):
    """Run dynesty once and return the sampler, results, and prior metadata."""
    try:
        import dynesty
    except ImportError as exc:
        raise ImportError(
            "dynesty is required to compute Bayesian evidence; install it with "
            "`pip install dynesty`"
        ) from exc

    labels = list(labels)
    bounds = validate_uniform_bounds(labels, bounds)
    prior_specs = _normalized_prior_specs(labels, bounds, prior_specs)
    evidence_cfg = config.get("fit", {}).get("evidence", {})
    seed = evidence_cfg.get("random_seed", config.get("fit", {}).get("random_seed"))
    rng = np.random.default_rng(None if seed is None else int(seed))

    def prior_transform(unit_cube):
        from scipy.special import ndtr, ndtri

        unit_cube = np.asarray(unit_cube, dtype=float)
        transformed = np.empty_like(unit_cube)
        for index, (unit_value, prior) in enumerate(zip(unit_cube, prior_specs)):
            lower = prior["lower"]
            upper = prior["upper"]
            if prior["type"] == "uniform":
                transformed[index] = lower + (upper - lower) * unit_value
            else:
                mean = prior["mean"]
                sigma = prior["sigma"]
                lower_cdf = ndtr((lower - mean) / sigma)
                upper_cdf = ndtr((upper - mean) / sigma)
                probability = lower_cdf + unit_value * (upper_cdf - lower_cdf)
                probability = np.clip(
                    probability, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
                )
                transformed[index] = mean + sigma * ndtri(probability)
        return transformed

    def finite_log_likelihood(theta):
        value = float(log_likelihood(np.asarray(theta, dtype=float), *args))
        return value if np.isfinite(value) else -1e300

    sampler = dynesty.NestedSampler(
        finite_log_likelihood,
        prior_transform,
        len(labels),
        nlive=int(evidence_cfg.get("nlive", 500)),
        bound=str(evidence_cfg.get("bound", "multi")),
        sample=str(evidence_cfg.get("sample", "rwalk")),
        rstate=rng,
    )
    sampler.run_nested(
        dlogz=float(evidence_cfg.get("dlogz", 0.1)),
        maxiter=evidence_cfg.get("maxiter"),
        maxcall=evidence_cfg.get("maxcall"),
        print_progress=bool(evidence_cfg.get("progress", True)),
    )
    return sampler, sampler.results, labels, bounds, prior_specs, rng


def _evidence_summary(results, labels, bounds, prior_specs, config):
    evidence_cfg = config.get("fit", {}).get("evidence", {})
    return {
        "logz": float(results.logz[-1]),
        "logz_err": float(results.logzerr[-1]),
        "information": float(results.information[-1]),
        "ncall": int(np.sum(results.ncall)),
        "nlive": int(evidence_cfg.get("nlive", 500)),
        "dlogz": float(evidence_cfg.get("dlogz", 0.1)),
        "sampler": "dynesty",
        "prior_type": "mixed" if any(p["type"] != "uniform" for p in prior_specs) else "uniform",
        "labels": labels,
        "bounds": bounds.tolist(),
        "priors": prior_specs,
    }


def run_nested_evidence(log_likelihood, labels, bounds, config, args=(), prior_specs=None):
    """Compute log evidence with dynesty using normalized configured priors."""
    _sampler, results, labels, bounds, prior_specs, _rng = _run_nested(
        log_likelihood, labels, bounds, config, args=args, prior_specs=prior_specs
    )
    return _evidence_summary(results, labels, bounds, prior_specs, config)


def run_dynesty_fit(
    log_likelihood, labels, bounds, config, theta0, args=(), prior_specs=None,
):
    """Use one nested run for posterior inference and Bayesian evidence."""
    from dynesty import utils as dyfunc

    sampler, results, labels, bounds, prior_specs, rng = _run_nested(
        log_likelihood, labels, bounds, config, args=args, prior_specs=prior_specs
    )
    raw_samples = np.asarray(results.samples, dtype=float)
    raw_logl = np.asarray(results.logl, dtype=float)
    weights = np.exp(np.asarray(results.logwt) - float(results.logz[-1]))
    weights /= np.sum(weights)
    samples = np.asarray(dyfunc.resample_equal(raw_samples, weights, rstate=rng))

    percentiles = np.asarray(
        [dyfunc.quantile(values, [0.16, 0.5, 0.84], weights=weights)
         for values in raw_samples.T],
        dtype=float,
    ).T
    summary = np.array(
        [(p50, p84 - p50, p50 - p16)
         for p16, p50, p84 in zip(*percentiles)],
        dtype=float,
    )
    best_index = int(np.nanargmax(raw_logl))
    best = raw_samples[best_index]
    evidence = _evidence_summary(results, labels, bounds, prior_specs, config)
    return {
        "sampler": sampler,
        "sampler_name": "dynesty",
        # These retain the pipeline's saved-array interface but are raw nested
        # samples, not walker trajectories.
        "chain": raw_samples[np.newaxis, :, :],
        "lnprobchain": raw_logl[np.newaxis, :],
        "final_state": best,
        "final_prob": float(raw_logl[best_index]),
        "summary": summary,
        "best": best,
        "samples": samples,
        "labels": labels,
        "theta0": np.asarray(theta0, dtype=float),
        "discard": 0,
        "evidence": evidence,
    }


def save_evidence(output_dir, evidence):
    path = Path(output_dir) / "evidence.json"
    with path.open("w") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")
    return path
