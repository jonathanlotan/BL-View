"""Shared fixtures.  Adds the repo root to sys.path so tests run without install."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blview.config import Config              # noqa: E402
from blview.synth.generator import (          # noqa: E402
    SyntheticScenario, generate, write_truth, write_vaisala_file,
)


@pytest.fixture(scope="session")
def config() -> Config:
    return Config()


@pytest.fixture(scope="session")
def short_dataset(tmp_path_factory):
    """Two hours of synthetic data spanning the nocturnal three-layer case."""
    tmp = tmp_path_factory.mktemp("synth")
    scenario = SyntheticScenario(duration_h=2.0)
    # 01:00-03:00 UTC: stable surface layer + residual layer + elevated haze,
    # no cloud, no screening -- the cleanest multi-layer case.
    data = generate(scenario, start_time=1756083600.0)
    raw = write_vaisala_file(tmp / "synth.dat", data, scenario)
    truth = write_truth(tmp / "synth_truth.json", data)
    return {"raw": raw, "truth": truth, "data": data, "scenario": scenario}
