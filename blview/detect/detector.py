"""Drives edge detection over a whole time-height block."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d

from ..config import Config
from ..preprocess import overlap_function
from ..model import Layer, ProcessedProfiles
from .haar import haar_covariance, haar_noise, log_field
from .layers import (
    aggregate_edges,
    classify_profile,
    detect_clouds,
    refine_edge_heights,
)


def detect_layers(
    processed: ProcessedProfiles,
    config: Config | None = None,
    chunk: int = 512,
) -> list[list[Layer]]:
    """Detect and classify layers for every profile.

    Returns one list of :class:`~blview.model.Layer` per timestamp, in the same
    order as ``processed.time``.  Screened profiles (precipitation, fog, low
    SNR) get an empty list -- they are *flagged, not deleted*, so the quicklook
    still shows the backscatter, but no layer is claimed from data known to be
    contaminated.

    The wavelet transforms are computed a chunk of profiles at a time so that
    memory stays bounded on a long ingest; the transform is purely vertical, so
    chunking in time changes nothing about the result.
    """
    config = config or Config()
    cfg = config.detect
    r = processed.range_
    dz = processed.range_resolution
    overlap_full = config.preprocess.overlap_full_m
    screened = processed.screened_mask()
    n_time = processed.n_time
    fine_scale = min(cfg.scales_m)

    sigma_single = (
        processed.sigma_single if processed.sigma_single is not None else processed.sigma
    )

    # --- overlap-correction bias -----------------------------------------
    # The modelled overlap function is wrong by an unknown amount that grows
    # as the correction grows.  That error is correlated between neighbouring
    # gates, so it does not behave like noise: summing per-gate variances would
    # let it average away, and the leftover ramp from an imperfect correction
    # would be detected as a shallow layer with high significance.  Instead the
    # error profile is pushed through the same Haar transform as the data, and
    # the resulting bias is added in quadrature to the transform's noise.
    overlap, _usable = overlap_function(r, config.preprocess)
    overlap_error_fraction = config.preprocess.overlap_uncertainty * (1.0 - overlap)
    # A fractional error f in beta is an error f / ln(10) in log10(beta).
    err_log = (overlap_error_fraction / np.log(10.0))[None, :]
    overlap_bias_log = {
        float(scale): np.abs(haar_covariance(err_log, dz, scale))[0]
        for scale in cfg.scales_m
    }

    # --- pass 1: cloud detection at full time resolution -----------------
    # Clouds are found before anything else, because where they are decides
    # what the aerosol analysis is allowed to look at.
    clouds_per_profile: list[list] = [[] for _ in range(n_time)]
    for start in range(0, n_time, chunk):
        stop = min(start + chunk, n_time)
        block = slice(start, stop)
        if screened[block].all():
            continue
        beta_full = processed.beta[block]
        w_fine_linear = haar_covariance(beta_full, dz, fine_scale)
        for k in range(stop - start):
            i = start + k
            if screened[i]:
                continue
            clouds_per_profile[i] = detect_clouds(
                beta_full[k], sigma_single[block][k],
                processed.beta_smooth[block][k], processed.sigma[block][k],
                w_fine_linear[k], r, cfg,
            )

    # A cloud present in only a few profiles is still smeared across the whole
    # temporal averaging window of beta_smooth, so the ceiling that protects the
    # aerosol analysis is the running *minimum* over that window -- not just
    # this profile's own cloud.
    ceiling_per_profile = np.array(
        [min((c.base for c in cs), default=np.inf) for cs in clouds_per_profile]
    )
    smooth_profiles = int(processed.attrs.get("smooth_time_profiles", 1) or 1)
    if smooth_profiles > 1 and n_time > 1:
        aerosol_ceiling = minimum_filter1d(
            ceiling_per_profile, size=min(smooth_profiles, n_time), mode="nearest"
        )
    else:
        aerosol_ceiling = ceiling_per_profile

    # --- pass 2: aerosol edges and classification ------------------------
    out: list[list[Layer]] = [[] for _ in range(n_time)]
    for start in range(0, n_time, chunk):
        stop = min(start + chunk, n_time)
        block = slice(start, stop)
        if screened[block].all():
            continue

        beta_s = processed.beta_smooth[block]
        sigma_s = processed.sigma[block]

        contaminated = r[None, :] >= (
            aerosol_ceiling[block] - cfg.cloud_mask_margin_m
        )[:, None]
        beta_clear = np.where(contaminated, np.nan, beta_s)
        sigma_clear = np.where(contaminated, np.nan, sigma_s)

        # Aerosol edges are found in log space (see haar.log_field): in linear
        # space the near-surface gradient dwarfs every elevated edge.
        log_beta, log_sigma = log_field(beta_clear, sigma_clear)
        transforms = {
            float(scale): (
                haar_covariance(log_beta, dz, scale),
                np.sqrt(
                    haar_noise(log_sigma, dz, scale) ** 2
                    + overlap_bias_log[scale] ** 2
                ),
            )
            for scale in cfg.scales_m
        }
        # The linear transform is computed at every dilation too: it localises
        # edges without the log transform's upward bias (refine_edge_heights),
        # and cloud bases are magnitude features that belong in linear space.
        overlap_err_linear = overlap_error_fraction[None, :] * np.abs(
            np.nan_to_num(beta_clear, nan=0.0)
        )
        linear = {
            float(scale): (
                haar_covariance(beta_clear, dz, scale),
                np.sqrt(
                    haar_noise(sigma_clear, dz, scale) ** 2
                    + haar_covariance(overlap_err_linear, dz, scale) ** 2
                ),
            )
            for scale in cfg.scales_m
        }

        for k in range(stop - start):
            i = start + k
            if screened[i]:
                continue
            clouds = clouds_per_profile[i]
            ceiling = float(aerosol_ceiling[i])
            edges = aggregate_edges(
                {scale: (w[k], sw[k]) for scale, (w, sw) in transforms.items()},
                r, cfg, ceiling,
            )
            edges = refine_edge_heights(
                edges,
                {scale: (w[k], sw[k]) for scale, (w, sw) in linear.items()},
                r, cfg,
            )
            out[i] = classify_profile(
                float(processed.time[i]), edges, clouds, beta_s[k], sigma_s[k],
                r, cfg, overlap_full,
            )
    return out
