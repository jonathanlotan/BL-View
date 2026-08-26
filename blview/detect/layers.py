"""Per-profile layer detection: candidate edges, cloud, and classification.

Pipeline for one profile:

1. **Multi-scale edge candidates.** The Haar covariance transform is run at
   every dilation in ``DetectConfig.scales_m``; local extrema exceeding
   ``snr_k`` times the transform's own propagated noise become candidates.
   Candidates found at several dilations within a merge tolerance are the same
   physical edge: they are clustered, the height is taken from the *finest*
   dilation that saw it (best localisation), and the number of dilations that
   saw it becomes a persistence score feeding the confidence.

2. **Cloud.** Detected separately and first, by backscatter magnitude, because
   a cloud return is two orders of magnitude above aerosol and needs no
   gradient analysis to find. Cloud detection runs at **full time resolution**
   (averaging would dilute intermittent cumulus), aerosol detection on the
   smoothed field.

3. **Classification.** Working upward from the surface:

   * the lowest surface-connected top is the **mixing layer** top;
   * an elevated layer that has its own detected *base* (a negative extremum)
     with a backscatter minimum below it is decoupled -- **haze**;
   * an elevated layer sitting directly on the one below with no such base is
     contiguous with it -- the **residual layer**.

   Nothing is reported above the lowest cloud base: the beam is attenuated
   there and any "layer" found would be an artefact of the extinction, not
   aerosol structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..config import Config, DetectConfig
from ..model import Layer, LayerType, ProcessedProfiles
from .haar import haar_covariance, haar_noise, local_extrema, log_field

#: Guard used wherever a ratio's denominator could be zero or negative.
_TINY = 1e-30


@dataclass
class EdgeCandidate:
    """One aerosol-gradient edge, aggregated across dilation scales."""

    height: float
    sign: int                    #: +1 = layer top (signal falls), -1 = layer base
    strength: float              #: |W| / sigma_W at the dilation that saw it best
    amplitude: float             #: |W| itself, in log10(backscatter) units
    best_scale: float            #: dilation supplying ``height`` (the finest one)
    scales: list[float] = field(default_factory=list)

    @property
    def n_scales(self) -> int:
        return len(self.scales)


@dataclass
class CloudDetection:
    """One cloud return in one profile."""

    base: float
    top: Optional[float]
    peak_beta: float
    peak_height: float
    opaque: bool
    confidence: float


# --------------------------------------------------------------------- utils
def _band(r: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (r >= lo) & (r < hi)


def _band_mean(values: np.ndarray, r: np.ndarray, lo: float, hi: float) -> float:
    m = _band(r, lo, hi)
    if not m.any():
        return float("nan")
    v = values[m]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _contrast(inside: float, outside: float, noise: float) -> float:
    """Ratio of backscatter inside a layer to just outside it.

    ``outside`` is legitimately negative in clean air after background
    subtraction, so the denominator is floored at the local noise level: the
    honest statement is "at least this much brighter than the noise", not an
    infinite ratio.
    """
    if not np.isfinite(inside):
        return float("nan")
    floor = max(noise if np.isfinite(noise) else 0.0, _TINY)
    return float(inside / max(outside if np.isfinite(outside) else 0.0, floor))


# ------------------------------------------------------------- edge finding
def aggregate_edges(
    w_by_scale: dict[float, tuple[np.ndarray, np.ndarray]],
    r: np.ndarray,
    cfg: DetectConfig,
    ceiling: float,
) -> list[EdgeCandidate]:
    """Cluster per-scale extrema of one profile into physical edges."""
    search = (r >= cfg.min_layer_height_m) & (r <= min(cfg.max_layer_height_m, ceiling))
    if not search.any():
        return []

    raw: list[tuple[float, int, float, float, float]] = []   # sig, sign, height, |W|, scale
    for scale in sorted(w_by_scale):
        w, sw = w_by_scale[scale]
        with np.errstate(invalid="ignore", divide="ignore"):
            sig = w / sw
        finite = np.isfinite(w) & np.isfinite(sw) & (sw > 0)
        band = search & finite
        if not band.any():
            continue
        # Relative-strength cut is taken within this dilation only: |W| is not
        # comparable between dilations.
        w_max = np.nanmax(np.abs(np.where(band, w, np.nan)))
        if not np.isfinite(w_max) or w_max <= 0:
            continue
        floor = cfg.min_relative_strength * w_max

        for sign in (+1, -1):
            hits = (
                local_extrema(w, sign)
                & band
                & (sign * sig > cfg.snr_k)
                & (np.abs(w) >= floor)
                & (np.abs(w) >= cfg.min_edge_log_amplitude)
            )
            for j in np.flatnonzero(hits):
                raw.append(
                    (abs(float(sig[j])), sign, float(r[j]), abs(float(w[j])), float(scale))
                )

    # Strongest first, so a cluster is seeded by its most convincing member.
    raw.sort(key=lambda t: -t[0])
    clusters: list[EdgeCandidate] = []
    for strength, sign, height, amplitude, scale in raw:
        tol = max(cfg.scale_merge_tolerance_m, 0.5 * scale)
        match = None
        for c in clusters:
            if c.sign == sign and abs(c.height - height) <= tol:
                match = c
                break
        if match is None:
            clusters.append(
                EdgeCandidate(
                    height=height, sign=sign, strength=strength,
                    amplitude=amplitude, best_scale=scale, scales=[scale],
                )
            )
            continue
        if scale not in match.scales:
            match.scales.append(scale)
        match.strength = max(match.strength, strength)
        match.amplitude = max(match.amplitude, amplitude)
        # The finest dilation that saw the edge localises it best.
        if scale < match.best_scale:
            match.best_scale = scale
            match.height = height

    clusters.sort(key=lambda c: c.height)
    return clusters


def refine_edge_heights(
    edges: list[EdgeCandidate],
    linear_transforms: dict[float, tuple[np.ndarray, np.ndarray]],
    r: np.ndarray,
    cfg: DetectConfig,
) -> list[EdgeCandidate]:
    """Snap log-space edge heights onto the linear transform's extremum.

    Detecting in log space is what makes a weak elevated haze edge comparable
    to the huge near-surface gradient (see :mod:`blview.detect.haar`), but it
    biases heights *upward*: for a layer decaying onto a background, the
    steepest fractional change happens above the steepest absolute change.
    Measured against the synthetic truth the bias is tens of metres and grows
    with the depth of the transition.

    So: detect in log space, localise in linear space.  The edge's height is
    moved to the nearest extremum of the linear transform at the same dilation
    and of the same sign, provided that extremum is itself significant and
    within half a dilation.  Otherwise the log-space height stands.
    """
    for edge in edges:
        pair = linear_transforms.get(edge.best_scale)
        if pair is None:
            continue
        w, sw = pair
        window = _band(r, edge.height - 0.5 * edge.best_scale,
                       edge.height + 0.5 * edge.best_scale)
        if not window.any():
            continue
        signed = np.where(window, edge.sign * w, np.nan)
        if not np.isfinite(signed).any():
            continue
        j = int(np.nanargmax(signed))
        with np.errstate(invalid="ignore", divide="ignore"):
            significance = edge.sign * w[j] / sw[j] if sw[j] > 0 else 0.0
        # Only trust the linear position if it is a real feature there too;
        # for a weak elevated layer the linear transform can be pure noise.
        if np.isfinite(significance) and significance > 0.5 * cfg.snr_k:
            edge.height = float(r[j])
    return edges


# ---------------------------------------------------------- cloud detection
def detect_clouds(
    beta: np.ndarray,
    sigma: np.ndarray,
    beta_smooth: np.ndarray,
    sigma_smooth: np.ndarray,
    w_fine: np.ndarray,
    r: np.ndarray,
    cfg: DetectConfig,
) -> list[CloudDetection]:
    """Find cloud returns in one full-resolution profile."""
    n = r.size
    thr = cfg.cloud_beta_threshold
    strong = np.isfinite(beta) & (beta > thr) & (r >= cfg.min_layer_height_m)
    noise_floor = float(np.nanmedian(sigma)) if np.isfinite(sigma).any() else _TINY

    clouds: list[CloudDetection] = []
    j = 0
    while j < n:
        if not strong[j]:
            j += 1
            continue
        start = j
        while j < n and strong[j]:
            j += 1
        if j - start < cfg.cloud_min_gates:
            continue

        peak_j = start + int(np.nanargmax(beta[start:j]))
        peak = float(beta[peak_j])

        # "Sharp return": the peak must tower over the 300 m below the base.
        below = beta[_band(r, r[start] - 300.0, r[start])]
        ref = float(np.nanmedian(below)) if below.size and np.isfinite(below).any() else 0.0
        if peak < cfg.cloud_sharpness_ratio * max(ref, 3.0 * noise_floor, _TINY):
            continue

        base = _refine_cloud_base(r, w_fine, start, cfg)

        # Apparent top: where backscatter falls back under the threshold.
        k = peak_j
        while k < n and np.isfinite(beta[k]) and beta[k] > thr:
            k += 1
        apparent_top = float(r[k]) if k < n else None

        # Opacity: is there anything left above? Judged on the *smoothed*
        # field -- this is a low-SNR question about slowly-varying ambient air,
        # exactly the regime averaging is for.
        top: Optional[float] = None
        opaque = True
        if apparent_top is not None:
            probe = _band(
                r, apparent_top + 50.0, apparent_top + 50.0 + cfg.cloud_opaque_probe_m
            )
            if probe.any():
                with np.errstate(invalid="ignore", divide="ignore"):
                    snr = np.nanmedian(beta_smooth[probe] / sigma_smooth[probe])
                if np.isfinite(snr) and snr >= cfg.cloud_opaque_snr:
                    top = apparent_top
                    opaque = False

        confidence = float(
            np.clip(0.5 + 0.5 * np.log10(max(peak, _TINY) / thr), 0.5, 1.0)
        )
        clouds.append(
            CloudDetection(
                base=base, top=top, peak_beta=peak, peak_height=float(r[peak_j]),
                opaque=opaque, confidence=confidence,
            )
        )
    return clouds


def _refine_cloud_base(
    r: np.ndarray, w_fine: np.ndarray, crossing: int, cfg: DetectConfig
) -> float:
    """Snap a threshold crossing to the steepest rise nearby.

    The gate at which backscatter crosses the cloud threshold depends on the
    threshold; the strongest negative Haar covariance (steepest increase with
    height) does not.  Falls back to the crossing when no extremum is nearby.
    """
    scale = min(cfg.scales_m)
    window = _band(r, r[crossing] - scale, r[crossing] + scale)
    if not window.any():
        return float(r[crossing])
    w = np.where(window, w_fine, np.nan)
    if not np.isfinite(w).any():
        return float(r[crossing])
    return float(r[int(np.nanargmin(w))])


# ------------------------------------------------------------ classification
def _edge_confidence(
    edge: EdgeCandidate, contrast: float, cfg: DetectConfig, overlap_full_m: float
) -> float:
    """Blend edge significance, scale persistence and contrast into 0..1."""
    c_strength = edge.strength / (edge.strength + cfg.snr_k)
    c_persist = edge.n_scales / max(len(cfg.scales_m), 1)
    c_contrast = (
        float(np.clip(np.log(max(contrast, 1.0)) / np.log(3.0), 0.0, 1.0))
        if np.isfinite(contrast) else 0.0
    )
    conf = 0.5 * c_strength + 0.3 * c_persist + 0.2 * c_contrast
    if edge.height < overlap_full_m:
        conf *= cfg.overlap_confidence_penalty
    return float(np.clip(conf, 0.0, 1.0))


def classify_profile(
    time: float,
    edges: list[EdgeCandidate],
    clouds: list[CloudDetection],
    beta_smooth: np.ndarray,
    sigma_smooth: np.ndarray,
    r: np.ndarray,
    cfg: DetectConfig,
    overlap_full_m: float,
) -> list[Layer]:
    """Turn edges and clouds into typed layers for one timestamp."""
    layers: list[Layer] = []
    for c in clouds:
        layers.append(
            Layer(
                time=time, type=LayerType.CLOUD, base_height=c.base, top_height=c.top,
                confidence=c.confidence,
                meta={
                    "peak_beta": c.peak_beta,
                    "peak_height": c.peak_height,
                    "opaque": c.opaque,
                    "top_undetermined_reason": (
                        "beam extinguished within cloud" if c.opaque else None
                    ),
                },
            )
        )

    ceiling = min((c.base for c in clouds), default=float("inf"))
    limit = min(cfg.max_layer_height_m, ceiling)
    tops = [e for e in edges if e.sign > 0 and e.height < limit]
    bases = [e for e in edges if e.sign < 0 and e.height < limit]
    if not tops:
        return layers

    def probe_depth(edge: EdgeCandidate) -> float:
        return max(edge.best_scale, 100.0)

    # --- mixing layer: lowest surface-connected top --------------------
    ml_edge: EdgeCandidate | None = None
    for t in tops:
        d = probe_depth(t)
        inside = _band_mean(beta_smooth, r, cfg.min_layer_height_m, t.height)
        above = _band_mean(beta_smooth, r, t.height, t.height + d)
        noise = _band_mean(sigma_smooth, r, t.height, t.height + d)
        # Surface-connected means no detected base underneath it: an elevated
        # slab has one, a layer resting on the ground does not.
        has_base_below = any(
            cfg.min_layer_height_m < b.height < t.height - cfg.min_elevated_depth_m
            for b in bases
        )
        if has_base_below:
            continue
        if _contrast(inside, above, noise) >= cfg.surface_layer_min_contrast:
            ml_edge = t
            break

    prev_top = cfg.min_layer_height_m
    if ml_edge is not None:
        d = probe_depth(ml_edge)
        inside = _band_mean(beta_smooth, r, cfg.min_layer_height_m, ml_edge.height)
        above = _band_mean(beta_smooth, r, ml_edge.height, ml_edge.height + d)
        noise = _band_mean(sigma_smooth, r, ml_edge.height, ml_edge.height + d)
        contrast = _contrast(inside, above, noise)
        layers.append(
            Layer(
                time=time, type=LayerType.MIXING_LAYER,
                # The mixing layer reaches the ground; the lowest *measured*
                # gate is recorded in meta because nothing below it was seen.
                base_height=0.0, top_height=ml_edge.height,
                confidence=_edge_confidence(ml_edge, contrast, cfg, overlap_full_m),
                meta={
                    "edge_strength": ml_edge.strength,
                    "scales_detected": sorted(ml_edge.scales),
                    "contrast": contrast,
                    "lowest_measured_height_m": cfg.min_layer_height_m,
                },
            )
        )
        prev_top = ml_edge.height

    # --- elevated layers above it ---------------------------------------
    residual_done = False
    for t in tops:
        if t.height <= prev_top + cfg.min_elevated_depth_m:
            continue
        d = probe_depth(t)

        # Does this layer have its own base, i.e. is it decoupled from below?
        gap_bases = [
            b for b in bases
            if prev_top < b.height < t.height - cfg.min_elevated_depth_m
        ]
        layer_base = prev_top
        layer_type = LayerType.RESIDUAL_LAYER
        edge_for_conf = t
        if gap_bases:
            b = max(gap_bases, key=lambda e: e.strength)
            inside_candidate = _band_mean(beta_smooth, r, b.height, t.height)
            gap = _band(r, prev_top, b.height)
            gap_min = (
                float(np.nanmin(beta_smooth[gap]))
                if gap.any() and np.isfinite(beta_smooth[gap]).any()
                else float("nan")
            )
            decoupled = (
                np.isfinite(gap_min)
                and np.isfinite(inside_candidate)
                and gap_min < cfg.decoupling_ratio * inside_candidate
            )
            if decoupled and b.height >= cfg.haze_min_base_m:
                layer_base = b.height
                layer_type = LayerType.HAZE
                edge_for_conf = b if b.strength < t.strength else t

        if layer_type is LayerType.RESIDUAL_LAYER and residual_done:
            # A second *contiguous* elevated layer -- one with no detected base
            # of its own -- is almost always the decaying tail of the layer
            # below rather than a distinct structure.  Without a base there is
            # no evidence of decoupling, so nothing is claimed.
            continue

        if (
            layer_type is LayerType.RESIDUAL_LAYER
            and t.height - layer_base < cfg.min_residual_depth_m
        ):
            # Too thin to be a residual layer: this is the mixing layer's own
            # entrainment zone, not a separate structure above it.
            continue

        inside = _band_mean(beta_smooth, r, layer_base, t.height)
        above = _band_mean(beta_smooth, r, t.height, t.height + d)
        noise = _band_mean(sigma_smooth, r, t.height, t.height + d)
        inside_noise = _band_mean(sigma_smooth, r, layer_base, t.height)
        contrast = _contrast(inside, above, noise)
        snr = inside / inside_noise if np.isfinite(inside_noise) and inside_noise > 0 else 0.0

        if not np.isfinite(contrast) or contrast < cfg.elevated_layer_min_contrast:
            continue
        if not np.isfinite(snr) or snr < cfg.elevated_layer_min_snr:
            continue

        layers.append(
            Layer(
                time=time, type=layer_type,
                base_height=float(layer_base), top_height=float(t.height),
                confidence=_edge_confidence(edge_for_conf, contrast, cfg, overlap_full_m),
                meta={
                    "edge_strength": t.strength,
                    "scales_detected": sorted(t.scales),
                    "contrast": contrast,
                    "layer_snr": float(snr),
                    "decoupled": layer_type is LayerType.HAZE,
                },
            )
        )
        residual_done = residual_done or layer_type is LayerType.RESIDUAL_LAYER
        prev_top = t.height

    return layers
