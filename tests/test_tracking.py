"""Temporal continuity: flicker removal, gap filling, de-duplication."""

import numpy as np
import pytest

from blview.config import Config
from blview.detect.tracking import track_layers, tracking_height
from blview.model import Layer, LayerType


def _times(n, step=30.0):
    return 1.756e9 + np.arange(n) * step


def _ml(t, height, confidence=0.9):
    return Layer(time=t, type=LayerType.MIXING_LAYER, base_height=0.0,
                 top_height=height, confidence=confidence)


def test_tracking_height_uses_the_meaningful_edge_per_type():
    assert tracking_height(_ml(0.0, 900.0)) == 900.0
    haze = Layer(0.0, LayerType.HAZE, 2000.0, 2800.0)
    assert tracking_height(haze) == 2000.0        # haze is identified by its base
    cloud = Layer(0.0, LayerType.CLOUD, 1500.0, None)
    assert tracking_height(cloud) == 1500.0


def test_a_steady_layer_becomes_one_track():
    times = _times(40)
    per_profile = [[_ml(t, 900.0 + 3.0 * i)] for i, t in enumerate(times)]
    out = track_layers(per_profile, times, Config())
    ids = {l.track_id for layers in out for l in layers}
    assert len(ids) == 1
    assert all(len(layers) == 1 for layers in out)


def test_flicker_shorter_than_the_minimum_is_discarded():
    """Two isolated detections are not a layer."""
    cfg = Config()
    times = _times(40)
    per_profile = [[] for _ in times]
    per_profile[10] = [_ml(times[10], 2500.0)]
    per_profile[11] = [_ml(times[11], 2505.0)]
    for i in range(40):
        per_profile[i].append(_ml(times[i], 900.0))
    out = track_layers(per_profile, times, cfg)
    heights = [l.top_height for layers in out for l in layers]
    assert all(abs(h - 900.0) < 60.0 for h in heights)


def test_a_short_gap_inside_a_track_is_filled_and_marked():
    times = _times(40)
    per_profile = [[_ml(t, 1000.0)] for t in times]
    for i in (18, 19, 20):
        per_profile[i] = []
    out = track_layers(per_profile, times, Config())
    for i in (18, 19, 20):
        assert len(out[i]) == 1
        filled = out[i][0]
        assert filled.interpolated is True
        assert filled.top_height == pytest.approx(1000.0, abs=1.0)
        # A filled point must be less trusted than a measured one.
        assert filled.confidence < out[5][0].confidence


def test_a_long_gap_is_not_bridged():
    """Beyond max_gap_s the track is closed, not silently interpolated across."""
    cfg = Config()
    times = _times(120)
    per_profile = [[_ml(t, 1000.0)] for t in times]
    gap = slice(40, 40 + int(cfg.track.max_gap_s / 30.0) + 10)
    for i in range(gap.start, gap.stop):
        per_profile[i] = []
    out = track_layers(per_profile, times, cfg)
    assert all(out[i] == [] for i in range(gap.start, gap.stop))
    ids = {l.track_id for layers in out for l in layers}
    assert len(ids) == 2          # closed and restarted


def test_a_jump_beyond_tolerance_starts_a_new_track():
    times = _times(60)
    per_profile = [[_ml(t, 500.0 if i < 30 else 3000.0)] for i, t in enumerate(times)]
    out = track_layers(per_profile, times, Config())
    ids = {l.track_id for layers in out for l in layers}
    assert len(ids) == 2


def test_running_median_removes_a_single_profile_spike():
    times = _times(40)
    per_profile = [[_ml(t, 1000.0)] for t in times]
    per_profile[20] = [_ml(times[20], 1400.0)]        # one bad profile
    out = track_layers(per_profile, times, Config())
    assert out[20][0].top_height == pytest.approx(1000.0, abs=30.0)


def test_only_one_mixing_layer_survives_per_profile():
    """Two coexisting tracks of a type that can only have one instance."""
    times = _times(40)
    per_profile = []
    for i, t in enumerate(times):
        entry = [_ml(t, 900.0, confidence=0.6)]
        if i % 2 == 0:                                # a competing track
            entry.append(_ml(t, 1500.0, confidence=0.95))
        per_profile.append(entry)
    out = track_layers(per_profile, times, Config())
    assert all(
        sum(1 for l in layers if l.type is LayerType.MIXING_LAYER) <= 1
        for layers in out
    )


def test_measured_detection_beats_an_interpolated_one():
    times = _times(60)
    per_profile = []
    for i, t in enumerate(times):
        entry = [_ml(t, 1500.0, confidence=0.95)] if i % 3 else []
        entry.append(_ml(t, 900.0, confidence=0.5))   # continuous, lower confidence
        per_profile.append(entry)
    out = track_layers(per_profile, times, Config())
    for i, layers in enumerate(out):
        ml = [l for l in layers if l.type is LayerType.MIXING_LAYER]
        assert len(ml) == 1
        if i % 3 == 0:            # the high-confidence track has no measurement here
            assert ml[0].interpolated is False


def test_multiple_clouds_in_one_profile_are_kept():
    """Cloud and haze can legitimately occur several times in one profile."""
    times = _times(40)
    per_profile = [
        [
            Layer(t, LayerType.CLOUD, 1200.0, None, confidence=0.9),
            Layer(t, LayerType.CLOUD, 3000.0, None, confidence=0.8),
        ]
        for t in times
    ]
    out = track_layers(per_profile, times, Config())
    assert all(len(layers) == 2 for layers in out)


def test_residual_layer_base_is_restacked_onto_the_mixing_top():
    times = _times(40)
    per_profile = [
        [
            _ml(t, 800.0),
            Layer(t, LayerType.RESIDUAL_LAYER, 795.0, 1600.0, confidence=0.9),
        ]
        for t in times
    ]
    out = track_layers(per_profile, times, Config())
    for layers in out:
        ml = next(l for l in layers if l.type is LayerType.MIXING_LAYER)
        rl = next(l for l in layers if l.type is LayerType.RESIDUAL_LAYER)
        assert rl.base_height == pytest.approx(ml.top_height)
