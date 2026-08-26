"""netCDF4 storage for backscatter grids, plus time/height downsampling.

One file per ``StoreConfig.grid_file_hours`` block, CF-style coordinate
variables and units, zlib-compressed float32.  float32 carries ~7 significant
digits, which is four orders of magnitude more precision than the instrument's
20-bit profile actually has, so nothing measurable is lost and the files halve.
"""

from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
from netCDF4 import Dataset

from .. import UNITS_BETA, UNITS_RANGE, UNITS_TIME, __version__

FILL = np.float32(np.nan)


def write_grid(
    path: str | Path,
    time: np.ndarray,
    range_: np.ndarray,
    beta: np.ndarray,
    quality: np.ndarray,
    beta_raw: Optional[np.ndarray] = None,
    attrs: Optional[dict[str, Any]] = None,
) -> Path:
    """Write one time-height block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.8"
        ds.title = "BL View processed ceilometer attenuated backscatter"
        ds.source = f"blview {__version__}"
        ds.history = f"{dt.datetime.now(dt.timezone.utc).isoformat()} created by blview"
        ds.comment = (
            "Layer heights derived from this field are aerosol backscatter "
            "gradients, not thermodynamic measurements."
        )
        for key, value in (attrs or {}).items():
            if value is None:
                continue
            try:
                setattr(ds, str(key), value if isinstance(value, (str, int, float)) else str(value))
            except (TypeError, ValueError):
                continue

        ds.createDimension("time", time.size)
        ds.createDimension("range", range_.size)

        v_time = ds.createVariable("time", "f8", ("time",))
        v_time.units = UNITS_TIME
        v_time.standard_name = "time"
        v_time.calendar = "standard"
        v_time[:] = time

        v_range = ds.createVariable("range", "f4", ("range",))
        v_range.units = UNITS_RANGE
        v_range.long_name = "height above instrument"
        v_range.positive = "up"
        v_range[:] = range_

        def add(name: str, data: np.ndarray, long_name: str) -> None:
            var = ds.createVariable(
                name, "f4", ("time", "range"), zlib=True, complevel=4,
                fill_value=FILL,
            )
            var.units = UNITS_BETA
            var.long_name = long_name
            var[:] = np.asarray(data, dtype="float32")

        add("beta_att", beta,
            "attenuated backscatter, range/background/overlap corrected")
        if beta_raw is not None:
            add("beta_att_raw", beta_raw,
                "attenuated backscatter as delivered by the instrument")

        v_q = ds.createVariable("quality_flag", "i4", ("time",))
        v_q.long_name = "blview.model.QualityFlag bitmask"
        v_q.flag_masks = "1 2 4 8 16 32 64 128"
        v_q.flag_meanings = (
            "precipitation fog low_snr saturated instrument_warning "
            "instrument_alarm window_contaminated no_detection"
        )
        v_q[:] = np.asarray(quality, dtype="int32")
    return path


def read_grid(path: str | Path, variable: str = "beta_att") -> dict[str, Any]:
    """Read one grid file back."""
    with Dataset(path, "r") as ds:
        return {
            "time": np.array(ds.variables["time"][:], dtype="float64"),
            "range": np.array(ds.variables["range"][:], dtype="float64"),
            "beta": np.ma.filled(
                np.asarray(ds.variables[variable][:], dtype="float64"), np.nan
            ),
            "quality": np.array(ds.variables["quality_flag"][:], dtype="int64"),
            "attrs": {k: ds.getncattr(k) for k in ds.ncattrs()},
        }


# ---------------------------------------------------------------- resampling
def _block_reduce(values: np.ndarray, factor: int, axis: int) -> np.ndarray:
    """NaN-aware block mean along one axis, padding the final partial block."""
    if factor <= 1:
        return values
    n = values.shape[axis]
    pad = (-n) % factor
    if pad:
        shape = list(values.shape)
        shape[axis] = pad
        values = np.concatenate([values, np.full(shape, np.nan)], axis=axis)
    shape = list(values.shape)
    shape[axis: axis + 1] = [values.shape[axis] // factor, factor]
    reshaped = values.reshape(shape)
    with warnings.catch_warnings():
        # A block can be entirely NaN (padding at the end, or a masked region);
        # NaN is the correct answer there, not a warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(reshaped, axis=axis + 1)


def downsample(
    time: np.ndarray,
    range_: np.ndarray,
    beta: np.ndarray,
    quality: np.ndarray,
    max_time: Optional[int] = None,
    max_range: Optional[int] = None,
) -> dict[str, Any]:
    """Reduce a grid to at most ``max_time`` x ``max_range`` cells.

    Block means, not decimation: dropping samples would make an intermittent
    cumulus appear and disappear depending on the zoom level.  Quality flags
    are combined with bitwise OR, so a block containing any screened profile
    stays marked as screened rather than being averaged into looking clean.
    """
    t_factor = max(1, int(np.ceil(time.size / max_time))) if max_time else 1
    r_factor = max(1, int(np.ceil(range_.size / max_range))) if max_range else 1
    if t_factor == 1 and r_factor == 1:
        return {
            "time": time, "range": range_, "beta": beta, "quality": quality,
            "time_factor": 1, "range_factor": 1,
        }

    out = _block_reduce(_block_reduce(beta, t_factor, axis=0), r_factor, axis=1)
    new_time = _block_reduce(time.astype("float64"), t_factor, axis=0)
    new_range = _block_reduce(range_.astype("float64"), r_factor, axis=0)

    if t_factor > 1:
        pad = (-quality.size) % t_factor
        q = np.concatenate([quality, np.zeros(pad, dtype=quality.dtype)]) if pad else quality
        new_quality = np.bitwise_or.reduce(
            q.reshape(-1, t_factor).astype("int64"), axis=1
        )
    else:
        new_quality = quality

    return {
        "time": new_time, "range": new_range, "beta": out, "quality": new_quality,
        "time_factor": t_factor, "range_factor": r_factor,
    }
