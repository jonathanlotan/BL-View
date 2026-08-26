"""Rolling-window store: round-trip, idempotency, retention, downsampling."""

import numpy as np
import pytest

from blview.config import Config
from blview.model import LayerType, QualityFlag
from blview.pipeline import run_pipeline
from blview.store import Store
from blview.store.grids import downsample, read_grid, write_grid


@pytest.fixture(scope="module")
def populated(short_dataset, tmp_path_factory):
    config = Config()
    config.store.data_dir = tmp_path_factory.mktemp("store")
    result = run_pipeline([short_dataset["raw"]], config)
    store = Store(config)
    info = store.write(result.processed, result.layers)
    return {"store": store, "config": config, "result": result, "info": info}


def test_grid_file_round_trips_through_netcdf(tmp_path):
    time = 1.756e9 + np.arange(50) * 30.0
    range_ = (np.arange(200) + 0.5) * 10.0
    beta = np.random.default_rng(0).normal(1e-6, 1e-7, (50, 200))
    beta[5, 10] = np.nan
    quality = np.zeros(50, dtype="int64")
    quality[3] = int(QualityFlag.FOG)

    path = write_grid(tmp_path / "g.nc", time, range_, beta, quality, beta_raw=beta * 1.1)
    back = read_grid(path)
    assert np.allclose(back["time"], time)
    assert np.allclose(back["range"], range_)
    # float32 storage: ~7 significant digits, far finer than the instrument.
    assert np.nanmax(np.abs(back["beta"] - beta)) < 1e-12
    assert np.isnan(back["beta"][5, 10])
    assert back["quality"][3] == int(QualityFlag.FOG)
    assert back["attrs"]["Conventions"].startswith("CF-")
    assert "not thermodynamic" in back["attrs"]["comment"]


def test_downsample_preserves_the_mean_and_ors_quality_flags():
    time = np.arange(100.0)
    range_ = np.arange(80.0)
    beta = np.random.default_rng(1).normal(1e-6, 1e-7, (100, 80))
    quality = np.zeros(100, dtype="int64")
    quality[7] = int(QualityFlag.PRECIPITATION)

    out = downsample(time, range_, beta, quality, max_time=25, max_range=20)
    assert out["beta"].shape == (25, 20)
    assert np.nanmean(out["beta"]) == pytest.approx(np.nanmean(beta), rel=1e-9)
    # A block containing a screened profile must stay marked screened, not be
    # averaged into looking clean.
    assert out["quality"][7 // out["time_factor"]] & int(QualityFlag.PRECIPITATION)


def test_downsample_is_a_noop_when_it_would_upsample():
    time, range_ = np.arange(10.0), np.arange(5.0)
    beta = np.ones((10, 5))
    out = downsample(time, range_, beta, np.zeros(10, dtype="int64"), 100, 100)
    assert out["time_factor"] == 1 and out["range_factor"] == 1
    assert out["beta"].shape == (10, 5)


def test_write_then_read_round_trips(populated):
    store, result = populated["store"], populated["result"]
    assert populated["info"]["profiles"] == result.processed.n_time
    window = store.read_window(hours=24)
    assert window["n_time_full"] == result.processed.n_time
    assert np.allclose(window["range"], result.processed.range_)
    finite = np.isfinite(result.processed.beta) & np.isfinite(window["beta"])
    assert np.nanmax(
        np.abs(window["beta"][finite] - result.processed.beta[finite])
    ) < 1e-12


def test_reingesting_the_same_period_replaces_rather_than_duplicates(populated):
    store, result = populated["store"], populated["result"]
    before_layers = len(store.read_layers(hours=48))
    before_files = store.status()["n_grid_files"]
    store.write(result.processed, result.layers)
    assert len(store.read_layers(hours=48)) == before_layers
    assert store.status()["n_grid_files"] == before_files


def test_layers_can_be_filtered_by_type_and_confidence(populated):
    store = populated["store"]
    everything = store.read_layers(hours=48)
    haze = store.read_layers(hours=48, types=["haze"])
    assert haze and len(haze) < len(everything)
    assert {l.type for l in haze} == {LayerType.HAZE}
    confident = store.read_layers(hours=48, min_confidence=0.8)
    assert all(l.confidence >= 0.8 for l in confident)
    assert len(confident) < len(everything)


def test_layer_meta_survives_the_database(populated):
    store = populated["store"]
    layers = [l for l in store.read_layers(hours=48) if l.meta]
    assert layers, "no layer kept its diagnostics"
    assert any("scales_detected" in l.meta or "peak_beta" in l.meta for l in layers)


def test_latest_profile_is_full_resolution(populated):
    store, result = populated["store"], populated["result"]
    profile = store.latest_profile()
    assert len(profile["range"]) == result.processed.n_range
    assert profile["time"] == pytest.approx(float(result.processed.time[-1]))
    assert isinstance(profile["quality_flags"], list)


def test_retention_drops_grid_files_entirely_outside_the_window(populated, tmp_path):
    config = Config()
    config.store.data_dir = tmp_path / "retained"
    config.store.grid_file_hours = 0.25
    config.store.retention_hours = 0.5
    store = Store(config)
    result = populated["result"]
    store.write(result.processed, result.layers)

    status = store.status()
    assert status["hours_available"] <= config.store.retention_hours + 0.3
    # Every catalogued file must still exist on disk.
    for row in store.db.all_grids():
        from pathlib import Path
        assert Path(row["path"]).exists()
    # ...and nothing older than the cutoff survives in the index.
    _, last = store.time_span()
    assert not store.db.read_layers(0, last - config.store.retention_hours * 3600 - 1)


def test_empty_store_reads_cleanly(tmp_path):
    config = Config()
    config.store.data_dir = tmp_path / "empty"
    store = Store(config)
    assert store.status()["time_end"] is None
    assert store.read_layers(hours=24) == []
    assert store.latest_profile() == {}
    window = store.read_window(hours=24)
    assert window["beta"].shape == (0, 0)
