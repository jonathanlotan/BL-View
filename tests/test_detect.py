"""Cloud detection, classification and the multi-scale aggregation."""

import numpy as np
import pytest

from blview.config import Config, DetectConfig
from blview.detect.haar import haar_covariance, haar_noise, log_field
from blview.detect.layers import (
    aggregate_edges, classify_profile, detect_clouds,
)
from blview.model import LayerType

RANGES = (np.arange(770) + 0.5) * 10.0
DZ = 10.0


def _slab(r, bottom, top, width=60.0):
    lo = 0.5 * (1 + np.tanh((r - bottom) / (0.5 * width))) if bottom is not None else 1.0
    hi = 0.5 * (1 - np.tanh((r - top) / (0.5 * width))) if top is not None else 1.0
    return lo * hi


def _sigma(scale=3e-14):
    """Noise of the *smoothed* field: constant in photon units, so R^2 here."""
    return scale * RANGES ** 2 + 1e-9


def _edges(beta, cfg, ceiling=float("inf")):
    sigma = _sigma()
    value, error = log_field(beta[None, :], sigma[None, :])
    transforms = {
        float(s): (haar_covariance(value, DZ, s)[0], haar_noise(error, DZ, s)[0])
        for s in cfg.scales_m
    }
    return aggregate_edges(transforms, RANGES, cfg, ceiling)


# ------------------------------------------------------------------- clouds
def _cloud_profile(base, peak=5e-4, transparent=False, ambient=2e-6):
    beta = ambient * _slab(RANGES, None, 1200.0) + 1e-7
    cloud = peak * np.exp(-(((RANGES - (base + 40.0)) / 55.0) ** 2))
    beta = beta + cloud
    if transparent:
        # Signal recovers above: an aerosol layer the beam still reaches.
        beta = beta + 1.5e-6 * _slab(RANGES, base + 200.0, base + 900.0)
    return beta


def test_opaque_cloud_reports_base_but_no_top():
    """The beam is extinguished, so the top is genuinely not determinable."""
    cfg = DetectConfig()
    beta = _cloud_profile(1800.0)
    sigma = _sigma()
    clouds = detect_clouds(
        beta, sigma, beta, sigma, haar_covariance(beta[None, :], DZ, 60.0)[0],
        RANGES, cfg,
    )
    assert len(clouds) == 1
    assert clouds[0].base == pytest.approx(1800.0, abs=60.0)
    assert clouds[0].top is None
    assert clouds[0].opaque is True


def test_transparent_cloud_reports_a_top():
    """When there is still signal above, the top is reported."""
    cfg = DetectConfig()
    beta = _cloud_profile(1800.0, peak=3e-4, transparent=True)
    sigma = _sigma()
    clouds = detect_clouds(
        beta, sigma, beta, sigma, haar_covariance(beta[None, :], DZ, 60.0)[0],
        RANGES, cfg,
    )
    assert len(clouds) == 1
    assert clouds[0].opaque is False
    assert clouds[0].top is not None
    assert clouds[0].top > clouds[0].base


def test_single_gate_spike_is_not_a_cloud():
    """Rejects cosmic-ray-like artefacts."""
    cfg = DetectConfig()
    beta = np.full(RANGES.size, 1e-6)
    beta[300] = 1e-3
    sigma = _sigma()
    clouds = detect_clouds(
        beta, sigma, beta, sigma, haar_covariance(beta[None, :], DZ, 60.0)[0],
        RANGES, cfg,
    )
    assert clouds == []


def test_aerosol_alone_is_never_called_cloud():
    cfg = DetectConfig()
    beta = 8e-6 * _slab(RANGES, None, 1500.0) + 1e-7
    sigma = _sigma()
    clouds = detect_clouds(
        beta, sigma, beta, sigma, haar_covariance(beta[None, :], DZ, 60.0)[0],
        RANGES, cfg,
    )
    assert clouds == []


# ------------------------------------------------------------------- edges
def test_multiscale_aggregation_merges_one_edge_across_dilations():
    cfg = DetectConfig()
    beta = 4e-6 * _slab(RANGES, None, 1500.0) + 1e-7
    found = _edges(beta, cfg)
    tops = [e for e in found if e.sign > 0]
    assert len(tops) == 1
    assert tops[0].height == pytest.approx(1500.0, abs=80.0)
    # A real edge is seen by several dilations; that is the persistence score.
    assert tops[0].n_scales >= 3


def test_elevated_layer_yields_both_a_base_and_a_top():
    cfg = DetectConfig()
    beta = 1e-7 + 2e-6 * _slab(RANGES, 2000.0, 2800.0, width=100.0)
    found = _edges(beta, cfg)
    bases = [e for e in found if e.sign < 0]
    tops = [e for e in found if e.sign > 0]
    assert bases and tops
    assert min(b.height for b in bases) == pytest.approx(2000.0, abs=120.0)
    assert max(t.height for t in tops) == pytest.approx(2800.0, abs=120.0)


# ---------------------------------------------------------- classification
def _classify(beta, cfg=None):
    cfg = cfg or DetectConfig()
    sigma = _sigma()
    return classify_profile(
        0.0, _edges(beta, cfg), [], beta, sigma, RANGES, cfg,
        Config().preprocess.overlap_full_m,
    )


def test_three_layer_night_profile_is_classified_correctly():
    """Surface mixing layer + residual layer + decoupled elevated haze."""
    beta = (
        1e-7
        + 6e-6 * _slab(RANGES, None, 300.0)
        + 2.4e-6 * _slab(RANGES, 300.0, 1500.0, width=120.0)
        + 1.9e-6 * _slab(RANGES, 2400.0, 3100.0, width=100.0)
    )
    layers = _classify(beta)
    by_type = {l.type: l for l in layers}
    assert set(by_type) == {
        LayerType.MIXING_LAYER, LayerType.RESIDUAL_LAYER, LayerType.HAZE
    }
    assert by_type[LayerType.MIXING_LAYER].top_height == pytest.approx(300.0, abs=100.0)
    assert by_type[LayerType.MIXING_LAYER].base_height == 0.0
    assert by_type[LayerType.RESIDUAL_LAYER].top_height == pytest.approx(1500.0, abs=120.0)
    assert by_type[LayerType.HAZE].base_height == pytest.approx(2400.0, abs=120.0)
    assert by_type[LayerType.HAZE].top_height == pytest.approx(3100.0, abs=150.0)


def test_haze_needs_its_own_base_to_be_called_decoupled():
    """A layer contiguous with the one below is residual, not haze."""
    beta = (
        1e-7
        + 6e-6 * _slab(RANGES, None, 300.0)
        + 2.4e-6 * _slab(RANGES, 300.0, 1500.0, width=120.0)
    )
    layers = _classify(beta)
    types = {l.type for l in layers}
    assert LayerType.RESIDUAL_LAYER in types
    assert LayerType.HAZE not in types


def test_well_mixed_layer_alone_gives_only_a_mixing_layer():
    beta = 1e-7 + 3e-6 * _slab(RANGES, None, 1600.0, width=150.0)
    layers = _classify(beta)
    assert [l.type for l in layers] == [LayerType.MIXING_LAYER]
    assert layers[0].top_height == pytest.approx(1600.0, abs=120.0)


def test_entrainment_zone_is_not_reported_as_a_residual_layer():
    """A thin transition above the mixing top is the mixing layer's own edge."""
    cfg = DetectConfig()
    beta = 1e-7 + 3e-6 * _slab(RANGES, None, 1600.0, width=250.0)
    layers = _classify(beta, cfg)
    residual = [l for l in layers if l.type is LayerType.RESIDUAL_LAYER]
    assert not residual, "entrainment zone misclassified as a residual layer"


def test_nothing_is_reported_above_a_cloud_base():
    """The beam is attenuated there; any 'layer' would be an extinction artefact."""
    cfg = DetectConfig()
    beta = (
        1e-7
        + 4e-6 * _slab(RANGES, None, 1000.0)
        + 2e-6 * _slab(RANGES, 2400.0, 3100.0, width=100.0)
    )
    sigma = _sigma()
    from blview.detect.layers import CloudDetection
    cloud = CloudDetection(
        base=1500.0, top=None, peak_beta=1e-3, peak_height=1540.0,
        opaque=True, confidence=0.9,
    )
    layers = classify_profile(
        0.0, _edges(beta, cfg, ceiling=1500.0), [cloud], beta, sigma, RANGES, cfg,
        Config().preprocess.overlap_full_m,
    )
    aerosol_above = [
        l for l in layers
        if l.type is not LayerType.CLOUD and (l.top_height or 0) > 1500.0
    ]
    assert not aerosol_above


def test_confidence_is_penalised_inside_the_overlap_region():
    cfg = DetectConfig()
    shallow = 1e-7 + 8e-6 * _slab(RANGES, None, 150.0, width=40.0)
    deep = 1e-7 + 8e-6 * _slab(RANGES, None, 900.0, width=40.0)
    a = [l for l in _classify(shallow, cfg) if l.type is LayerType.MIXING_LAYER]
    b = [l for l in _classify(deep, cfg) if l.type is LayerType.MIXING_LAYER]
    assert a and b
    assert a[0].confidence < b[0].confidence
