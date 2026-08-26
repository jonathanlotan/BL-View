"""Preprocessing: corrections, noise characterisation and screening.

Stages, in order:

1. **Range correction** -- multiply by R^2, *only* if the adapter reports that
   the instrument has not already done it.  Vaisala CL-series firmware has;
   the generic CSV adapter has not.
2. **Background offset removal** -- estimate any residual detector offset from
   the far field and remove it.  Done in un-range-corrected space, because a
   constant detector offset appears in a range-corrected profile as an
   R^2-growing ramp, not as a constant.
3. **Overlap correction** -- divide by a modelled near-range overlap function.
   The real overlap function is instrument- and alignment-specific and is not
   published, so gates where the correction would need to amplify by more than
   a capped gain are masked out rather than fabricated.
4. **Noise characterisation** -- a range-dependent noise standard deviation,
   derived from the observed far-field scatter and propagated through the
   smoothing that follows.
5. **Screening** -- precipitation, fog and low-SNR profiles are *flagged*, so
   they still appear in the quicklook, but are excluded from layer detection.
6. **Smoothing** -- temporal and vertical averaging, with screened profiles
   excluded from the temporal average, producing the field detection runs on.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import Config, PreprocessConfig
from .model import ProcessedProfiles, ProfileSet, QualityFlag


# ------------------------------------------------------------------ helpers
def overlap_function(
    r: np.ndarray, cfg: PreprocessConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Modelled telescope overlap and the mask of usable gates.

    Returns ``(overlap, usable)`` where ``overlap`` rises from 0 at
    ``overlap_start_m`` to 1 at ``overlap_full_m`` as a power law, and
    ``usable`` is False for gates where ``1 / overlap`` would exceed
    ``overlap_max_gain``.

    Dividing by a small, *guessed* overlap value amplifies noise into
    structure that looks exactly like a shallow layer.  Masking those gates is
    the only honest option when the true overlap function is unknown
    (ASSUMPTIONS.md #P2).
    """
    span = max(cfg.overlap_full_m - cfg.overlap_start_m, 1e-6)
    x = np.clip((r - cfg.overlap_start_m) / span, 0.0, 1.0)
    overlap = x ** cfg.overlap_shape
    usable = overlap >= (1.0 / cfg.overlap_max_gain)
    return overlap, usable


def lowest_usable_height(cfg: PreprocessConfig) -> float:
    """Height of the lowest gate that survives the overlap gain cap."""
    span = cfg.overlap_full_m - cfg.overlap_start_m
    return cfg.overlap_start_m + span * (1.0 / cfg.overlap_max_gain) ** (
        1.0 / cfg.overlap_shape
    )


def _robust_std(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """1.4826 * median absolute deviation -- insensitive to real signal."""
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def _boxcar(x: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Un-normalised running sum via a boxcar mean (normalisation cancels)."""
    if size <= 1:
        return x
    return uniform_filter1d(x, size=size, axis=axis, mode="constant", cval=0.0)


def _odd(n: int) -> int:
    n = max(int(n), 1)
    return n if n % 2 == 1 else n + 1


# ------------------------------------------------------------------ stages
def estimate_noise_scale(
    beta_rc: np.ndarray, r: np.ndarray, cfg: PreprocessConfig
) -> np.ndarray:
    """Per-profile noise scale ``sigma0`` such that ``sigma(R) = sigma0 * R^2``.

    Detector noise is (to first order) constant in raw photon-count units.  A
    range-corrected profile has been multiplied by R^2, so its noise grows as
    R^2 too.  The scale is measured in the far field, where there is nothing
    but noise, by de-range-correcting and taking a robust scatter estimate.
    """
    top = cfg.noise_ref_top_m if cfg.noise_ref_top_m is not None else np.inf
    far = (r >= cfg.noise_ref_bottom_m) & (r <= top)
    if far.sum() < 10:                       # short profile: use the top decile
        far = r >= np.quantile(r, 0.9)

    de_corrected = beta_rc[:, far] / (r[far] ** 2)[None, :]
    sigma0 = _robust_std(de_corrected, axis=1)

    # A single profile gives a poor variance estimate; smooth along time.
    n = _odd(cfg.noise_time_smooth)
    if n > 1 and sigma0.size > 1:
        sigma0 = uniform_filter1d(sigma0, size=min(n, sigma0.size), mode="nearest")

    # Guard against a degenerate (all-constant) profile block.
    floor = np.nanmedian(sigma0)
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-30
    return np.where(np.isfinite(sigma0) & (sigma0 > 0), sigma0, floor)


def screen_profiles(
    beta: np.ndarray, sigma: np.ndarray, r: np.ndarray, cfg: PreprocessConfig
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Flag precipitation, fog and unusable profiles.

    Returns ``(flags, diagnostics)``.  Precipitation is tested **before** fog:
    a precipitating profile is also extinguished aloft, so the fog test alone
    would claim it.  What separates them is depth -- fog is a strong *shallow*
    return, precipitation fills the whole sub-cloud layer.
    """
    n_time = beta.shape[0]
    flags = np.zeros(n_time, dtype="int64")
    dz = float(np.median(np.diff(r))) if r.size > 1 else 1.0

    # --- fill depth: how far up surface-connected strong backscatter reaches
    start = int(np.searchsorted(r, cfg.precip_bottom_m))
    gap_gates = max(int(round(cfg.precip_gap_tolerance_m / dz)), 1)
    strong = np.nan_to_num(beta, nan=0.0) > cfg.precip_beta_threshold
    fill_depth = np.zeros(n_time)
    for i in range(n_time):
        run = strong[i, start:]
        gap = 0
        last = start - 1
        for j, hit in enumerate(run):
            if hit:
                gap = 0
                last = start + j
            else:
                gap += 1
                if gap > gap_gates:
                    break
        fill_depth[i] = max(r[last] - cfg.precip_bottom_m, 0.0) if last >= start else 0.0

    # --- near-surface peak and the extinction test above it
    near = (r >= cfg.precip_bottom_m) & (r <= cfg.fog_probe_m)
    peak_near = (
        np.nanmax(np.where(near[None, :], beta, np.nan), axis=1)
        if near.any() else np.zeros(n_time)
    )
    band = (r >= cfg.fog_extinction_height_m) & (
        r <= cfg.fog_extinction_height_m + cfg.fog_extinction_span_m
    )
    snr_upper = (
        np.nanmedian(np.where(band[None, :], beta / sigma, np.nan), axis=1)
        if band.any() else np.full(n_time, np.inf)
    )

    is_precip = fill_depth >= cfg.precip_min_depth_m
    is_fog = (
        ~is_precip
        & (np.nan_to_num(peak_near, nan=0.0) >= cfg.fog_beta_threshold)
        & (np.nan_to_num(snr_upper, nan=0.0) < cfg.fog_extinction_snr)
    )

    # --- overall usability
    usable_band = (r >= 200.0) & (r <= 1500.0)
    snr_mid = (
        np.nanmedian(np.where(usable_band[None, :], beta / sigma, np.nan), axis=1)
        if usable_band.any() else np.full(n_time, np.inf)
    )
    is_lowsnr = ~is_precip & ~is_fog & (np.nan_to_num(snr_mid, nan=0.0) < cfg.min_profile_snr)

    # Screening is deliberately conservative: dilate fog/precipitation by one
    # profile so the transition profiles either side are not half-contaminated.
    is_precip = _dilate(is_precip)
    is_fog = _dilate(is_fog) & ~is_precip

    flags |= np.where(is_precip, int(QualityFlag.PRECIPITATION), 0)
    flags |= np.where(is_fog, int(QualityFlag.FOG), 0)
    flags |= np.where(is_lowsnr, int(QualityFlag.LOW_SNR), 0)

    return flags, {
        "fill_depth_m": fill_depth,
        "peak_near": peak_near,
        "snr_upper": snr_upper,
        "snr_mid": snr_mid,
    }


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a boolean mask by one sample either side."""
    if mask.size < 2:
        return mask
    out = mask.copy()
    out[:-1] |= mask[1:]
    out[1:] |= mask[:-1]
    return out


def preprocess(profiles: ProfileSet, config: Config | None = None) -> ProcessedProfiles:
    """Run the full preprocessing chain over a :class:`ProfileSet`."""
    config = config or Config()
    cfg = config.preprocess
    r = profiles.range_
    beta = np.array(profiles.beta, dtype="float64", copy=True)
    dz = profiles.range_resolution
    notes: list[str] = []

    # --- 1. range correction --------------------------------------------
    if not profiles.range_corrected:
        beta = beta * (r ** 2)[None, :]
        notes.append("applied R^2 range correction")
    else:
        notes.append("range correction already applied by instrument")

    # --- 2. residual background offset ----------------------------------
    # A constant detector offset eps appears in a range-corrected profile as
    # eps * R^2, so it must be estimated and removed in de-range-corrected
    # space.  This is a no-op on data the instrument has already cleaned.
    top = cfg.noise_ref_top_m if cfg.noise_ref_top_m is not None else np.inf
    far = (r >= cfg.noise_ref_bottom_m) & (r <= top)
    if far.sum() >= 10:
        offset = np.nanmedian(beta[:, far] / (r[far] ** 2)[None, :], axis=1)
        n = _odd(cfg.noise_time_smooth)
        if n > 1 and offset.size > 1:
            offset = uniform_filter1d(offset, size=min(n, offset.size), mode="nearest")
        beta = beta - offset[:, None] * (r ** 2)[None, :]
        notes.append(f"removed residual background offset (median {np.nanmedian(offset):.3g})")

    # --- 3. overlap correction ------------------------------------------
    overlap, usable = overlap_function(r, cfg)
    beta = np.where(usable[None, :], beta / np.where(overlap > 0, overlap, np.nan), np.nan)
    notes.append(
        f"overlap-corrected below {cfg.overlap_full_m:.0f} m; "
        f"gates below {lowest_usable_height(cfg):.0f} m masked"
    )

    # --- 4. noise ---------------------------------------------------------
    sigma0 = estimate_noise_scale(beta, r, cfg)
    sigma_single = sigma0[:, None] * (r ** 2)[None, :]
    # The overlap correction divides the signal *and* the noise.
    sigma_single = np.where(
        usable[None, :], sigma_single / np.where(overlap > 0, overlap, np.nan), np.nan
    )
    # Detector noise alone tends to zero near the ground, which would make any
    # low-level wiggle look significant.  Real returns also carry
    # signal-proportional speckle; add it in quadrature.
    speckle = cfg.speckle_fraction * np.abs(np.nan_to_num(beta, nan=0.0))
    sigma_single = np.sqrt(sigma_single ** 2 + speckle ** 2)

    # --- 5. screening (on individual profiles) ---------------------------
    flags, diagnostics = screen_profiles(beta, sigma_single, r, cfg)
    flags |= np.asarray(profiles.quality, dtype="int64")   # keep adapter flags
    screened = (flags & int(QualityFlag.PRECIPITATION | QualityFlag.FOG
                            | QualityFlag.LOW_SNR | QualityFlag.INSTRUMENT_ALARM)) != 0

    # --- 6. smoothing -----------------------------------------------------
    dt = float(np.median(np.diff(profiles.time))) if profiles.time.size > 1 else 1.0
    n_t = _odd(round(cfg.smooth_time_s / dt)) if dt > 0 else 1
    n_t = min(n_t, max(profiles.n_time, 1))
    n_z = _odd(round(cfg.smooth_vertical_m / dz)) if dz > 0 else 1

    valid = np.isfinite(beta)
    weight = np.where(screened[:, None], 0.0, 1.0) * valid
    filled = np.where(valid, beta, 0.0)

    num = _boxcar(_boxcar(filled * weight, n_t, axis=0), n_z, axis=1)
    den = _boxcar(_boxcar(weight, n_t, axis=0), n_z, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta_smooth = np.where(den > 0, num / den, np.nan)

    # Effective independent-sample count (Kish n_eff = (sum w)^2 / sum w^2), so
    # sigma is propagated exactly rather than assuming a full box everywhere
    # (edges, masked gates and screened neighbours all contribute fewer).
    # _boxcar returns the *mean* over the window, so both den and num_sq carry
    # a 1/(n_t*n_z) factor; the ratio den^2/num_sq therefore has one factor of
    # (n_t*n_z) too many and must be scaled back up.  Without this correction
    # n_eff collapses to 1 everywhere and the smoothing appears to reduce no
    # noise at all.
    box = float(n_t * n_z)
    num_sq = _boxcar(_boxcar(weight ** 2, n_t, axis=0), n_z, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_eff = np.where(num_sq > 0, den ** 2 / num_sq * box, np.nan)
    sigma = sigma_single / np.sqrt(np.maximum(n_eff, 1.0))
    sigma = np.where(np.isfinite(beta_smooth), sigma, np.nan)

    # A smoothed profile built from too few contributors cannot be trusted.
    finite_eff = np.isfinite(n_eff)
    contributors = np.where(
        finite_eff.any(axis=1),
        np.nanmax(np.where(finite_eff, n_eff, -np.inf), axis=1),
        np.nan,
    )
    thin = ~np.isfinite(contributors) | (contributors < max(0.25 * box, 1.0))
    flags |= np.where(thin, int(QualityFlag.NO_DETECTION), 0)

    attrs = dict(profiles.attrs)
    attrs.update(
        {
            "preprocess_notes": "; ".join(notes),
            "smooth_time_profiles": int(n_t),
            "smooth_vertical_gates": int(n_z),
            "lowest_usable_height_m": float(lowest_usable_height(cfg)),
            "n_screened": int(screened.sum()),
        }
    )

    return ProcessedProfiles(
        time=profiles.time,
        range_=r,
        beta=beta,
        beta_smooth=beta_smooth,
        sigma=sigma,
        quality=flags,
        sigma_single=sigma_single,
        beta_raw=np.asarray(profiles.beta),
        cloud_base_reported=profiles.cloud_base_reported,
        attrs=attrs,
    )
