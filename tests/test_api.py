"""HTTP API: contracts, wire format, and that the caveat is always present."""

import base64
import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from blview.api import DEFAULT_VMAX, DEFAULT_VMIN, create_app, quantise
from blview.config import Config
from blview.pipeline import run_pipeline
from blview.store import Store


@pytest.fixture(scope="module")
def client(short_dataset, tmp_path_factory):
    config = Config()
    config.store.data_dir = tmp_path_factory.mktemp("api-store")
    result = run_pipeline([short_dataset["raw"]], config)
    Store(config).write(result.processed, result.layers)
    return TestClient(create_app(config))


def _decode(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype="uint8").reshape(payload["shape"])


# ------------------------------------------------------------- quantisation
def test_quantise_round_trips_within_one_level():
    beta = np.array([[1e-7, 1e-6, 1e-5, 1e-4]])
    codes = quantise(beta, DEFAULT_VMIN, DEFAULT_VMAX)
    span = DEFAULT_VMAX - DEFAULT_VMIN
    recovered = DEFAULT_VMIN + (codes.astype(float) - 1) / 254.0 * span
    assert np.allclose(recovered, np.log10(beta), atol=span / 254.0)


def test_quantise_separates_missing_from_measured_but_negative():
    """Background subtraction makes the signal-free far field go negative."""
    beta = np.array([[np.nan, -1e-7, 0.0, 1e-6]])
    codes = quantise(beta, DEFAULT_VMIN, DEFAULT_VMAX)
    assert codes[0, 0] == 0            # NaN -> genuinely missing
    assert codes[0, 1] == 1            # negative -> bottom of the scale
    assert codes[0, 2] == 1            # zero -> bottom of the scale
    assert codes[0, 3] > 1


def test_quantise_clamps_outside_the_scale():
    beta = np.array([[1e-12, 1e-1]])
    codes = quantise(beta, DEFAULT_VMIN, DEFAULT_VMAX)
    assert codes.tolist() == [[1, 255]]


# ------------------------------------------------------------------ endpoints
def test_health_and_status(client):
    assert client.get("/api/health").json()["status"] == "ok"
    status = client.get("/api/status").json()
    assert status["hours_available"] > 0
    assert status["n_grid_files"] >= 1
    assert "thresholds" in status
    # The scientific caveat must be impossible to miss.
    assert "does not measure" in status["caveat"]
    assert "temperature" in status["caveat"]


def test_window_default_is_24h_and_self_describing(client):
    payload = client.get("/api/window").json()
    codes = _decode(payload)
    assert codes.shape == tuple(payload["shape"])
    assert len(payload["time"]) == codes.shape[0]
    assert len(payload["range"]) == codes.shape[1]
    assert len(payload["quality"]) == codes.shape[0]
    encoding = payload["encoding"]
    assert encoding["scale"] == "log10" and encoding["missing_value"] == 0
    assert encoding["units"] == "m-1 sr-1"
    assert "not thermodynamic" in payload["disclaimer"]


def test_window_respects_the_requested_display_size(client):
    payload = client.get("/api/window", params={"max_time": 60, "max_range": 40}).json()
    assert payload["shape"][0] <= 60
    assert payload["shape"][1] <= 40
    assert payload["downsample"]["time_factor"] >= 1
    assert payload["downsample"]["n_time_full"] >= payload["shape"][0]


def test_window_max_height_clips_the_profile(client):
    payload = client.get("/api/window", params={"max_height": 3000}).json()
    assert max(payload["range"]) <= 3000.0


def test_window_layers_are_columnar_and_typed(client):
    payload = client.get("/api/window").json()
    assert payload["layers"], "no layers returned"
    for name, arrays in payload["layers"].items():
        assert name in {"cloud", "mixing_layer", "residual_layer", "haze"}
        lengths = {len(v) for v in arrays.values()}
        assert len(lengths) == 1, f"{name}: ragged columns"
        assert all(t is None or t >= b for b, t in zip(arrays["base"], arrays["top"]))


def test_binary_format_matches_the_json_payload(client):
    params = {"max_time": 200, "max_range": 100}
    as_json = client.get("/api/window", params=params).json()
    raw = client.get("/api/window", params={**params, "format": "binary"}).content

    assert raw[:4] == b"BLVW"
    assert raw[4] == 1
    header_len = struct.unpack("<I", raw[5:9])[0]
    header = json.loads(raw[9: 9 + header_len])
    body = raw[9 + header_len:]

    assert header["shape"] == as_json["shape"]
    assert len(body) == header["shape"][0] * header["shape"][1]
    assert body == _decode(as_json).tobytes()
    assert len(raw) < len(json.dumps(as_json))


def test_window_rejects_an_inverted_colour_scale(client):
    assert client.get("/api/window", params={"vmin": 0, "vmax": -1}).status_code == 400


def test_layers_endpoint_filters(client):
    everything = client.get("/api/layers").json()
    assert everything["count"] > 0
    assert "not thermodynamic" in everything["disclaimer"]

    haze = client.get("/api/layers", params={"type": "haze"}).json()
    assert 0 < haze["count"] < everything["count"]
    assert {l["type"] for l in haze["layers"]} == {"haze"}

    confident = client.get("/api/layers", params={"min_confidence": 0.9}).json()
    assert all(l["confidence"] >= 0.9 for l in confident["layers"])

    measured = client.get("/api/layers", params={"include_interpolated": False}).json()
    assert all(not l["interpolated"] for l in measured["layers"])


def test_latest_profile_is_full_resolution_with_its_layers(client):
    profile = client.get("/api/profile/latest").json()
    assert len(profile["range"]) == len(profile["beta"])
    assert len(profile["range"]) > 500
    assert isinstance(profile["quality_flags"], list)
    assert isinstance(profile["screened"], bool)
    for layer in profile["layers"]:
        assert layer["type"] in {"cloud", "mixing_layer", "residual_layer", "haze"}


def test_quality_flag_decoder(client):
    assert client.get("/api/quality-flags", params={"value": 3}).json()["flags"] == [
        "precipitation", "fog"
    ]
    assert client.get("/api/quality-flags", params={"value": 0}).json()["flags"] == []


def test_empty_store_serves_an_empty_window_rather_than_erroring(tmp_path):
    config = Config()
    config.store.data_dir = tmp_path / "empty"
    empty = TestClient(create_app(config))
    payload = empty.get("/api/window").json()
    assert payload["shape"] == [0, 0]
    assert payload["time"] == []
    assert empty.get("/api/layers").json()["count"] == 0
    assert empty.get("/api/profile/latest").status_code == 404
