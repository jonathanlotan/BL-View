"""End-to-end: the pipeline must recover what the generator injected."""

import numpy as np
import pytest

from blview.config import Config
from blview.model import LayerType, QualityFlag
from blview.pipeline import run_pipeline
from blview.validate import check, load_truth, score_all


@pytest.fixture(scope="module")
def result(short_dataset):
    return run_pipeline([short_dataset["raw"]], Config())


def test_pipeline_runs_end_to_end(result, short_dataset):
    assert result.adapter_name == "vaisala_cl"
    assert result.processed.n_time == result.profiles.n_time
    assert len(result.layers) == result.processed.n_time
    assert result.all_layers, "no layers detected at all"


def test_nocturnal_three_layer_structure_is_recovered(result, short_dataset):
    """01:00-03:00 UTC: mixing layer, residual layer and decoupled haze."""
    truth = load_truth(short_dataset["truth"])
    report = score_all(truth, result.layers, result.processed.quality)

    for feature in ("sml_top", "rl_top", "haze_base", "haze_top"):
        score = report["features"][feature]
        assert score["n_observable"] > 0, f"{feature}: nothing observable to score"
        assert score["detection_rate"] >= 0.9, f"{feature}: {score}"
        assert score["within_tolerance"] >= 0.9, f"{feature}: {score}"
        assert abs(score["bias"]) <= 75.0, f"{feature}: {score}"


def test_all_three_layer_types_are_reported_simultaneously(result):
    """The point of the tool: not one boundary-layer height but a structure."""
    counts = 0
    for layers in result.layers:
        types = {l.type for l in layers}
        if {LayerType.MIXING_LAYER, LayerType.RESIDUAL_LAYER, LayerType.HAZE} <= types:
            counts += 1
    assert counts > 0.5 * len(result.layers)


def test_layers_are_physically_ordered_and_well_formed(result):
    for layers in result.layers:
        for layer in layers:
            assert 0.0 <= layer.base_height <= 8000.0
            if layer.top_height is not None:
                assert layer.top_height >= layer.base_height
            assert 0.0 <= layer.confidence <= 1.0
            assert layer.track_id is not None
        ml = next((l for l in layers if l.type is LayerType.MIXING_LAYER), None)
        rl = next((l for l in layers if l.type is LayerType.RESIDUAL_LAYER), None)
        if ml and rl:
            assert rl.base_height >= ml.top_height - 1e-6


def test_layer_serialisation_round_trips(result):
    from blview.model import Layer

    for layer in result.all_layers[:200]:
        assert Layer.from_dict(layer.to_dict()) == layer


def test_screened_profiles_get_no_layers():
    """Contaminated profiles are flagged, not fed to detection."""
    from blview.synth.generator import (
        SyntheticScenario, generate, write_vaisala_file,
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        scenario = SyntheticScenario(duration_h=1.0)
        data = generate(scenario, start_time=1756090800.0)      # 03:00 -> fog
        raw = write_vaisala_file(Path(tmp) / "fog.dat", data, scenario)
        res = run_pipeline([raw], Config())
        screened = res.processed.screened_mask()
        assert screened.any()
        assert all(not res.layers[i] for i in np.flatnonzero(screened))


def test_precipitation_is_screened_not_detected_as_aerosol():
    from blview.synth.generator import (
        SyntheticScenario, generate, write_vaisala_file,
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        scenario = SyntheticScenario(duration_h=0.8)
        data = generate(scenario, start_time=1756155600.0)      # 21:00 -> precip
        raw = write_vaisala_file(Path(tmp) / "rain.dat", data, scenario)
        res = run_pipeline([raw], Config())
        flagged = (res.processed.quality & int(QualityFlag.PRECIPITATION)) != 0
        assert flagged.mean() > 0.8


def test_a_second_adapter_needs_no_downstream_change(short_dataset, tmp_path):
    """The whole point of the adapter interface."""
    from blview.adapters import get_adapter
    from blview.adapters.generic_csv import GenericCSVAdapter

    profiles = get_adapter("vaisala_cl").read(short_dataset["raw"])
    csv = GenericCSVAdapter.write(tmp_path / "same.csv", profiles)
    via_csv = run_pipeline([csv], Config())
    via_raw = run_pipeline([short_dataset["raw"]], Config())

    assert via_csv.adapter_name == "generic_csv"
    assert via_csv.processed.n_time == via_raw.processed.n_time

    def tops(res):
        return np.array([
            next((l.top_height for l in layers if l.type is LayerType.MIXING_LAYER),
                 np.nan)
            for layers in res.layers
        ])

    a, b = tops(via_csv), tops(via_raw)
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 0.8 * len(a)
    assert np.nanmax(np.abs(a[both] - b[both])) < 1e-6


def test_validation_check_catches_a_broken_report():
    """The harness must actually fail when the numbers are bad."""
    broken = {
        "n_profiles": 10,
        "features": {
            "sml_top": {
                "n_observable": 10, "n_matched": 2, "detection_rate": 0.2,
                "bias": 400.0, "within_tolerance": 0.1, "tolerance_m": 150.0,
                "false_positive_rate": 0.9,
            }
        },
        "screening": [
            {"name": "fog", "recall": 0.1, "precision": 0.1},
        ],
    }
    failures = check(broken)
    assert len(failures) >= 4
    assert any("detection rate" in f for f in failures)
    assert any("bias" in f for f in failures)
