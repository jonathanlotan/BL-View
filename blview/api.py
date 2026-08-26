"""HTTP API serving the rolling window of processed backscatter and layers.

Endpoints
---------
``GET /api/health``           liveness
``GET /api/status``           what is in the store, and the active thresholds
``GET /api/window``           rolling window of grid + layers (default 24 h)
``GET /api/layers``           layers only
``GET /api/profile/latest``   the most recent single profile, full resolution
``GET /``                     the quicklook page

Wire format
-----------
A full 24 h CL31 grid is 2880 x 770 = 2.2 million cells; as JSON numbers that
is tens of megabytes for a picture 1200 pixels wide.  So the grid is
**downsampled to the requested display size and quantised to one byte per
cell** in log10 space, which is what the heatmap draws anyway:

* ``format=json`` (default) -- one JSON object with the payload base64-encoded.
  Self-contained and easy to consume.
* ``format=binary`` -- ``BLVW`` magic, version byte, uint32 header length,
  a JSON header, then the raw bytes.  Avoids the 33 % base64 overhead.

Byte 0 means **no data** -- a gate that was masked (below the overlap floor) or
never measured.  Values 1-255 map linearly onto ``[vmin, vmax]`` in
log10(m-1 sr-1), with 1 meaning "at or below vmin".

Backscatter that has been background-subtracted is legitimately *negative*
where there is no signal, which is most of the profile above ~4 km.  That is
measured data, not missing data, so it is clamped to 1 rather than reported as
absent -- otherwise a third of a normal quicklook would be holes.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Config
from .store import Store, describe_quality

#: Default colour-scale limits, log10(m-1 sr-1).  -7.5 is below the molecular
#: background of a clean free troposphere; -3.5 is above where a CL31 profile
#: saturates inside cloud, so the whole physical range fits.
DEFAULT_VMIN = -7.5
DEFAULT_VMAX = -3.5

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: Prepended to every quantised payload so a consumer can tell the format.
MAGIC = b"BLVW"
BINARY_VERSION = 1


def quantise(beta: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """log10-scale a grid into bytes.

    0 means *no data* (NaN: a masked gate).  1-255 span ``[vmin, vmax]``.
    Non-positive backscatter is measured data -- background subtraction makes
    the signal-free far field oscillate about zero -- so it clamps to 1 rather
    than being reported as missing.
    """
    if beta.size == 0:
        return np.zeros(beta.shape, dtype="uint8")
    measured = np.isfinite(beta)
    with np.errstate(invalid="ignore", divide="ignore"):
        logged = np.log10(np.where(beta > 0, beta, np.nan))
    span = max(vmax - vmin, 1e-9)
    scaled = np.clip((logged - vmin) / span, 0.0, 1.0)
    # Measured-but-non-positive gates land at the bottom of the scale.
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.where(measured, 1 + np.rint(scaled * 254.0), 0).astype("uint8")


def columnar_layers(layers: list) -> dict[str, dict[str, list]]:
    """Group layers by type into parallel arrays.

    A 24 h window holds several thousand layers.  As a list of JSON objects
    they outweigh the entire quantised backscatter grid by an order of
    magnitude; grouped into arrays, rounded to the precision the data actually
    has, and without the per-layer diagnostic ``meta``, they cost a fraction of
    that -- and arrive in exactly the form a canvas renderer wants for drawing
    one polyline per track.  Full records with ``meta`` remain available from
    ``/api/layers``.
    """
    out: dict[str, dict[str, list]] = {}
    for layer in layers:
        bucket = out.setdefault(
            str(layer.type),
            {"time": [], "base": [], "top": [], "confidence": [],
             "track_id": [], "interpolated": []},
        )
        bucket["time"].append(round(float(layer.time), 1))
        bucket["base"].append(round(float(layer.base_height), 1))
        bucket["top"].append(
            None if layer.top_height is None else round(float(layer.top_height), 1)
        )
        bucket["confidence"].append(round(float(layer.confidence), 3))
        bucket["track_id"].append(layer.track_id)
        bucket["interpolated"].append(int(layer.interpolated))
    return out


def _percentiles(beta: np.ndarray) -> dict[str, Optional[float]]:
    """Data-driven scale hints so the UI can offer auto-scaling."""
    with np.errstate(invalid="ignore", divide="ignore"):
        logged = np.log10(np.where(beta > 0, beta, np.nan))
    finite = logged[np.isfinite(logged)]
    if finite.size == 0:
        return {"p01": None, "p50": None, "p99": None}
    return {
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
    }


def build_window_payload(
    store: Store,
    hours: float,
    max_time: Optional[int],
    max_range: Optional[int],
    max_height: Optional[float],
    vmin: float,
    vmax: float,
    include_layers: bool,
    min_confidence: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Assemble the window response; returns ``(header, payload_bytes_array)``."""
    window = store.read_window(
        hours=hours, max_time=max_time, max_range=max_range, max_height=max_height
    )
    beta = window["beta"]
    codes = quantise(beta, vmin, vmax)

    layers: dict[str, dict[str, list]] = {}
    if include_layers and window["time_start"] is not None:
        layers = columnar_layers(
            store.read_layers(
                start=window["time_start"], end=window["time_end"],
                min_confidence=min_confidence,
            )
        )

    header = {
        "version": __version__,
        "shape": [int(codes.shape[0]), int(codes.shape[1])],
        "dtype": "uint8",
        "encoding": {
            "scale": "log10",
            "units": "m-1 sr-1",
            "vmin": vmin,
            "vmax": vmax,
            "missing_value": 0,
            "note": (
                "byte b in 1..255 maps to log10(beta) = vmin + (b-1)/254 * (vmax-vmin); "
                "0 means no data"
            ),
        },
        "time": [float(t) for t in window["time"]],
        "range": [float(r) for r in window["range"]],
        "quality": [int(q) for q in window["quality"]],
        "downsample": {
            "time_factor": window["time_factor"],
            "range_factor": window["range_factor"],
            "n_time_full": window["n_time_full"],
            "n_range_full": window["n_range_full"],
        },
        "time_start": window["time_start"],
        "time_end": window["time_end"],
        "percentiles": _percentiles(beta),
        "layers": layers,
        "layer_format": (
            "grouped by type into parallel arrays; heights in metres above the "
            "instrument, time in unix seconds. Full records with diagnostics: /api/layers"
        ),
        "disclaimer": (
            "Layer heights are aerosol backscatter gradients, not thermodynamic "
            "measurements. This is not an inversion temperature."
        ),
    }
    return header, codes


def create_app(config: Config | None = None, store: Store | None = None) -> FastAPI:
    """Build the FastAPI application."""
    config = config or Config()
    store = store or Store(config)

    app = FastAPI(
        title="BL View",
        version=__version__,
        description=(
            "Ceilometer boundary-layer, cloud and haze-layer viewer. All layer "
            "heights are aerosol backscatter gradients, NOT thermodynamic "
            "measurements -- see /api/status for the caveat in full."
        ),
    )
    app.state.store = store
    app.state.config = config

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        info = store.status()
        info["version"] = __version__
        info["caveat"] = (
            "BL View reports aerosol backscatter gradients. It does not measure, "
            "derive or estimate temperature or inversion strength. Aerosol layer "
            "boundaries often coincide with thermodynamic inversions but are not "
            "the same quantity."
        )
        info["thresholds"] = {
            "cloud_beta_threshold": config.detect.cloud_beta_threshold,
            "scales_m": list(config.detect.scales_m),
            "snr_k": config.detect.snr_k,
            "overlap_full_m": config.preprocess.overlap_full_m,
            "min_layer_height_m": config.detect.min_layer_height_m,
            "max_layer_height_m": config.detect.max_layer_height_m,
            "smooth_time_s": config.preprocess.smooth_time_s,
        }
        return info

    @app.get("/api/window")
    def window(
        hours: float = Query(24.0, gt=0, le=24 * 30,
                             description="rolling window length, hours"),
        max_time: int = Query(1200, ge=16, le=20000,
                              description="maximum time columns returned"),
        max_range: int = Query(400, ge=16, le=5000,
                               description="maximum height rows returned"),
        max_height: Optional[float] = Query(
            None, gt=0, description="clip the profile above this height, metres"
        ),
        vmin: float = Query(DEFAULT_VMIN, description="colour scale min, log10"),
        vmax: float = Query(DEFAULT_VMAX, description="colour scale max, log10"),
        layers: bool = Query(True, description="include detected layers"),
        min_confidence: float = Query(0.0, ge=0.0, le=1.0),
        format: str = Query("json", pattern="^(json|binary)$"),
    ) -> Response:
        if vmax <= vmin:
            raise HTTPException(400, "vmax must be greater than vmin")
        header, codes = build_window_payload(
            store, hours, max_time, max_range, max_height, vmin, vmax,
            layers, min_confidence,
        )
        raw = codes.tobytes()
        if format == "binary":
            head = json.dumps(header).encode("utf-8")
            body = MAGIC + bytes([BINARY_VERSION]) + struct.pack("<I", len(head)) + head + raw
            return Response(content=body, media_type="application/octet-stream")
        header["data"] = base64.b64encode(raw).decode("ascii")
        return Response(content=json.dumps(header), media_type="application/json")

    @app.get("/api/layers")
    def get_layers(
        hours: float = Query(24.0, gt=0, le=24 * 30),
        type: Optional[str] = Query(
            None, description="comma-separated: cloud,mixing_layer,residual_layer,haze"
        ),
        min_confidence: float = Query(0.0, ge=0.0, le=1.0),
        include_interpolated: bool = Query(True),
    ) -> dict[str, Any]:
        types = [t.strip() for t in type.split(",") if t.strip()] if type else None
        found = store.read_layers(hours=hours, types=types, min_confidence=min_confidence)
        if not include_interpolated:
            found = [l for l in found if not l.interpolated]
        return {
            "count": len(found),
            "layers": [l.to_dict() for l in found],
            "disclaimer": (
                "Aerosol backscatter gradients, not thermodynamic inversion heights."
            ),
        }

    @app.get("/api/profile/latest")
    def latest_profile() -> dict[str, Any]:
        profile = store.latest_profile()
        if not profile:
            raise HTTPException(404, "no profiles in the store yet")
        return profile

    @app.get("/api/quality-flags")
    def quality_flags(value: int = Query(..., ge=0)) -> dict[str, Any]:
        return {"value": value, "flags": describe_quality(value)}

    if WEB_DIR.is_dir():
        @app.get("/")
        def index() -> FileResponse:
            page = WEB_DIR / "index.html"
            if not page.is_file():
                raise HTTPException(
                    404, "quicklook page not found; the API is still available at /docs"
                )
            return FileResponse(page)

        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    return app


app = create_app()
