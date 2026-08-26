"""Temporal continuity tracking for detected layers.

Per-profile detection is memoryless, so a layer top that sits a gate or two
higher on one profile than the next produces a quicklook that flickers, and a
layer that is briefly missed produces a hole. Tracking fixes both:

* detections are associated with existing tracks by **optimal** (Hungarian)
  assignment within a height-jump tolerance that grows with the time gap;
* tracks that never accumulate enough detections are discarded as flicker;
* short gaps *inside* a surviving track are filled by interpolation and marked
  ``interpolated=True`` with reduced confidence, so a consumer can always tell
  a filled point from a measured one;
* each track's heights are run through a running median, which removes
  single-profile spikes without rounding off genuine transitions the way a
  mean would.

Tracking is done independently per layer type, and on the height that is
physically meaningful for that type -- see :func:`tracking_height`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import Config, TrackConfig
from ..model import Layer, LayerType


def tracking_height(layer: Layer) -> float:
    """The height that identifies a layer of this type through time.

    Cloud and haze are identified by their **base** (that is the sharp,
    well-determined edge, and for haze it is the quantity of interest); the
    mixing and residual layers by their **top** (their base is the surface, or
    the layer below).
    """
    if layer.type in (LayerType.CLOUD, LayerType.HAZE):
        return layer.base_height
    return layer.top_height if layer.top_height is not None else layer.base_height


@dataclass
class _Track:
    track_id: int
    layer_type: LayerType
    indices: list[int] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    last_time: float = 0.0
    last_height: float = 0.0

    def add(self, index: int, layer: Layer, time: float) -> None:
        self.indices.append(index)
        self.layers.append(layer)
        self.last_time = time
        self.last_height = tracking_height(layer)


def _running_median(values: np.ndarray, width: int) -> np.ndarray:
    """Running median with edge clamping; ``width`` is forced odd."""
    n = values.size
    width = max(int(width), 1)
    if width % 2 == 0:
        width += 1
    if width <= 1 or n == 0:
        return values.copy()
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(padded[i: i + width])
    return out


def track_layers(
    per_profile: list[list[Layer]],
    times: np.ndarray,
    config: Config | None = None,
) -> list[list[Layer]]:
    """Assign track ids, drop flicker, fill gaps and smooth heights.

    Returns a new per-profile list; the input is not modified.
    """
    config = config or Config()
    cfg = config.track
    times = np.asarray(times, dtype="float64")

    tracks: list[_Track] = []
    next_id = 0
    for layer_type in LayerType:
        active: list[_Track] = []
        for i, layers in enumerate(per_profile):
            detections = [layer for layer in layers if layer.type is layer_type]
            now = float(times[i])

            # Retire tracks that have gone quiet.
            still_active = []
            for tr in active:
                if now - tr.last_time <= cfg.max_gap_s:
                    still_active.append(tr)
                else:
                    tracks.append(tr)
            active = still_active

            if not detections:
                continue
            if not active:
                for det in detections:
                    tr = _Track(next_id, layer_type)
                    next_id += 1
                    tr.add(i, det, now)
                    active.append(tr)
                continue

            # Optimal assignment within the jump tolerance.
            cost = np.empty((len(detections), len(active)))
            allowed = np.empty(cost.shape, dtype=bool)
            for a, det in enumerate(detections):
                h = tracking_height(det)
                for b, tr in enumerate(active):
                    gap_min = max(now - tr.last_time, 0.0) / 60.0
                    tolerance = cfg.max_jump_m + cfg.max_jump_m_per_min * gap_min
                    distance = abs(h - tr.last_height)
                    cost[a, b] = distance
                    allowed[a, b] = distance <= tolerance
            # Forbidden pairs get a cost the optimiser will never choose over a
            # legal one, then are filtered out of the result.
            big = float(cost.max() + 1e6) if cost.size else 1e6
            rows, cols = linear_sum_assignment(np.where(allowed, cost, big))

            matched_det: set[int] = set()
            for a, b in zip(rows, cols):
                if not allowed[a, b]:
                    continue
                active[b].add(i, detections[a], now)
                matched_det.add(a)
            for a, det in enumerate(detections):
                if a in matched_det:
                    continue
                tr = _Track(next_id, layer_type)
                next_id += 1
                tr.add(i, det, now)
                active.append(tr)
        tracks.extend(active)

    # ---------------------------------------------------------------- output
    out: list[list[Layer]] = [[] for _ in per_profile]
    for tr in tracks:
        if len(tr.indices) < cfg.min_track_profiles:
            continue                      # flicker, not a layer
        emitted = _finalise_track(tr, times, cfg)
        for index, layer in emitted:
            out[index].append(layer)

    for i, layers in enumerate(out):
        layers.sort(key=lambda l: (l.base_height, l.type.value))
        out[i] = _deduplicate(layers)
        _reconcile(out[i])
    return out


def _finalise_track(
    tr: _Track, times: np.ndarray, cfg: TrackConfig
) -> list[tuple[int, Layer]]:
    """Smooth a track's heights and optionally fill its internal gaps."""
    indices = list(tr.indices)
    tracked = np.array([tracking_height(layer) for layer in tr.layers])
    tops = np.array(
        [
            layer.top_height if layer.top_height is not None else np.nan
            for layer in tr.layers
        ]
    )
    bases = np.array([layer.base_height for layer in tr.layers])

    smooth_tracked = _running_median(tracked, cfg.smooth_profiles)
    smooth_tops = _running_median(np.nan_to_num(tops, nan=0.0), cfg.smooth_profiles)
    smooth_bases = _running_median(bases, cfg.smooth_profiles)

    emitted: list[tuple[int, Layer]] = []
    for k, index in enumerate(indices):
        src = tr.layers[k]
        base = float(smooth_bases[k])
        top = None if not np.isfinite(tops[k]) else float(smooth_tops[k])
        if tr.layer_type in (LayerType.CLOUD, LayerType.HAZE):
            base = float(smooth_tracked[k])
        else:
            top = float(smooth_tracked[k])
        if top is not None and top < base:
            top = base
        emitted.append((
            index,
            Layer(
                time=src.time, type=src.type, base_height=base, top_height=top,
                confidence=src.confidence, track_id=tr.track_id,
                interpolated=False, meta=dict(src.meta),
            ),
        ))

    if not cfg.fill_gaps:
        return emitted

    # Fill internal gaps by linear interpolation, clearly marked as filled.
    filled: list[tuple[int, Layer]] = []
    for k in range(len(indices) - 1):
        i0, i1 = indices[k], indices[k + 1]
        if i1 - i0 <= 1:
            continue
        if times[i1] - times[i0] > cfg.max_gap_s:
            continue
        left, right = emitted[k][1], emitted[k + 1][1]
        for index in range(i0 + 1, i1):
            frac = (index - i0) / (i1 - i0)

            def blend(a: Optional[float], b: Optional[float]) -> Optional[float]:
                if a is None or b is None:
                    return None
                return float(a + (b - a) * frac)

            filled.append((
                index,
                Layer(
                    time=float(times[index]), type=tr.layer_type,
                    base_height=float(blend(left.base_height, right.base_height)),
                    top_height=blend(left.top_height, right.top_height),
                    confidence=0.5 * min(left.confidence, right.confidence),
                    track_id=tr.track_id, interpolated=True,
                    meta={"filled_between": [i0, i1]},
                ),
            ))
    return emitted + filled


def _deduplicate(layers: list[Layer]) -> list[Layer]:
    """Keep at most one mixing layer and one residual layer per profile.

    Two tracks of the same type can be alive at once -- during the morning
    transition one track follows the rising mixing-layer top while another
    still sits on the residual-layer top -- and each fills its own gaps, so a
    profile can end up with a measured mixing layer *and* an interpolated one
    at a completely different height.  There is only ever one surface mixing
    layer, so the measured detection wins over an interpolated one, and
    confidence breaks the remaining ties.  Cloud and haze are left alone:
    several of each in one profile is real.
    """
    unique = {LayerType.MIXING_LAYER, LayerType.RESIDUAL_LAYER}
    best: dict[LayerType, Layer] = {}
    out: list[Layer] = []
    for layer in layers:
        if layer.type not in unique:
            out.append(layer)
            continue
        rank = (not layer.interpolated, layer.confidence)
        incumbent = best.get(layer.type)
        if incumbent is None or rank > (not incumbent.interpolated, incumbent.confidence):
            best[layer.type] = layer
    out.extend(best.values())
    out.sort(key=lambda l: (l.base_height, l.type.value))
    return out


def _reconcile(layers: list[Layer]) -> None:
    """Keep stacked layers consistent within one profile.

    The residual layer's base is the mixing-layer top, but the two are tracked
    and smoothed independently, so after smoothing they can disagree by a gate
    or two.  Re-stacking them keeps the quicklook from showing a sliver of gap
    or overlap that means nothing.
    """
    ml = next((l for l in layers if l.type is LayerType.MIXING_LAYER), None)
    rl = next((l for l in layers if l.type is LayerType.RESIDUAL_LAYER), None)
    if ml is not None and rl is not None and ml.top_height is not None:
        rl.base_height = float(ml.top_height)
        if rl.top_height is not None and rl.top_height < rl.base_height:
            rl.top_height = rl.base_height
