"""Rolling-window store: SQLite index + netCDF grid files.

See :mod:`blview.store.db` for why the split.  The whole store is a directory::

    data/
      blview.sqlite            layers, per-profile quality, grid catalogue
      grids/
        grid_20260825T000000Z.nc
        grid_20260825T060000Z.nc
        ...

Retention is a rolling window (``StoreConfig.retention_hours``): on every
ingest, layers and profiles older than the cutoff are deleted and grid files
that lie *entirely* before it are unlinked.  A file straddling the cutoff is
kept, because part of it is still inside the window.
"""

from __future__ import annotations

import datetime as dt
import logging
import time as _time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import Config
from ..model import Layer, ProcessedProfiles, QualityFlag
from .db import LayerDB
from .grids import downsample, read_grid, write_grid

log = logging.getLogger("blview.store")

__all__ = ["Store", "LayerDB", "write_grid", "read_grid", "downsample"]


class Store:
    """Facade over the SQLite index and the netCDF grid directory."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.root = Path(self.config.store.data_dir)
        self.grid_dir = self.root / "grids"
        self.grid_dir.mkdir(parents=True, exist_ok=True)
        self.db = LayerDB(self.root / "blview.sqlite")

    # ----------------------------------------------------------- ingesting
    def write(
        self,
        processed: ProcessedProfiles,
        layers: list[list[Layer]],
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Store one processed block, replacing anything already in its span."""
        if processed.n_time == 0:
            return {"profiles": 0, "layers": 0, "grid_files": []}

        t_start = float(processed.time[0])
        t_end = float(processed.time[-1])

        # Drop any existing grid file overlapping this span: reprocessing a
        # period must replace it, never leave two versions to be merged.
        stale = [row["path"] for row in self.db.grids_overlapping(t_start, t_end)]
        for path in stale:
            Path(path).unlink(missing_ok=True)
        self.db.unregister_grids(stale)

        written: list[str] = []
        block_seconds = max(self.config.store.grid_file_hours * 3600.0, 60.0)
        edges = np.arange(
            np.floor(t_start / block_seconds) * block_seconds,
            t_end + block_seconds,
            block_seconds,
        )
        for lo in edges:
            hi = lo + block_seconds
            sel = (processed.time >= lo) & (processed.time < hi)
            if not sel.any():
                continue
            stamp = dt.datetime.fromtimestamp(float(lo), dt.timezone.utc)
            path = self.grid_dir / f"grid_{stamp:%Y%m%dT%H%M%SZ}.nc"
            write_grid(
                path,
                time=processed.time[sel],
                range_=processed.range_,
                beta=processed.beta[sel],
                quality=processed.quality[sel],
                beta_raw=(
                    processed.beta_raw[sel] if processed.beta_raw is not None else None
                ),
                attrs={
                    "site_name": self.config.site_name,
                    "instrument": self.config.instrument,
                    **{
                        k: v for k, v in processed.attrs.items()
                        if isinstance(v, (str, int, float))
                    },
                },
            )
            self.db.register_grid(
                str(path), float(processed.time[sel][0]), float(processed.time[sel][-1]),
                int(sel.sum()), processed.n_range, _time.time(),
            )
            written.append(str(path))

        flat = [layer for profile in layers for layer in profile]
        n_layers = self.db.replace_layers(t_start, t_end, flat)
        self.db.replace_profiles(
            processed.time,
            processed.quality,
            np.array([len(p) for p in layers], dtype="int64"),
        )
        self.db.set_meta(
            {
                "site_name": self.config.site_name,
                "instrument": self.config.instrument,
                "range_resolution_m": processed.range_resolution,
                "n_range": processed.n_range,
                "max_range_m": float(processed.range_[-1]),
                "lowest_usable_height_m": processed.attrs.get("lowest_usable_height_m"),
                "config": self.config.to_dict(),
                "last_ingest": _time.time(),
                **(extra_meta or {}),
            }
        )
        purged = self.purge()
        return {
            "profiles": processed.n_time,
            "layers": n_layers,
            "grid_files": written,
            "purged_files": purged,
        }

    def purge(self) -> int:
        """Apply the rolling-window retention.  Returns files removed."""
        _, latest = self.db.time_span()
        if latest is None:
            return 0
        cutoff = latest - self.config.store.retention_hours * 3600.0
        removed = self.db.purge_before(cutoff)
        for path in removed:
            Path(path).unlink(missing_ok=True)
        if removed:
            log.info("retention: removed %d grid file(s) before %.0f", len(removed), cutoff)
        return len(removed)

    # ------------------------------------------------------------- reading
    def time_span(self) -> tuple[Optional[float], Optional[float]]:
        return self.db.time_span()

    def read_window(
        self,
        hours: float = 24.0,
        end: Optional[float] = None,
        start: Optional[float] = None,
        max_time: Optional[int] = None,
        max_range: Optional[int] = None,
        max_height: Optional[float] = None,
        variable: str = "beta_att",
    ) -> dict[str, Any]:
        """Read a rolling window of the grid, optionally downsampled."""
        first, last = self.db.time_span()
        if last is None:
            return _empty_window()
        t_end = float(end) if end is not None else float(last)
        t_start = float(start) if start is not None else t_end - hours * 3600.0

        rows = self.db.grids_overlapping(t_start, t_end)
        blocks = []
        range_ref: Optional[np.ndarray] = None
        for row in rows:
            path = Path(row["path"])
            if not path.exists():
                continue
            try:
                grid = read_grid(path, variable=variable)
            except (OSError, KeyError) as exc:
                log.warning("skipping unreadable grid %s: %s", path, exc)
                continue
            if range_ref is None:
                range_ref = grid["range"]
            elif grid["range"].shape != range_ref.shape or not np.allclose(
                grid["range"], range_ref
            ):
                # A change of instrument configuration mid-window; the older
                # grid cannot be stacked onto the newer one.
                log.warning("skipping %s: different range grid", path)
                continue
            sel = (grid["time"] >= t_start) & (grid["time"] <= t_end)
            if sel.any():
                blocks.append(
                    (grid["time"][sel], grid["beta"][sel], grid["quality"][sel])
                )
        if not blocks or range_ref is None:
            return _empty_window()

        blocks.sort(key=lambda b: b[0][0])
        time = np.concatenate([b[0] for b in blocks])
        beta = np.vstack([b[1] for b in blocks])
        quality = np.concatenate([b[2] for b in blocks])
        order = np.argsort(time, kind="stable")
        time, beta, quality = time[order], beta[order], quality[order]

        range_ = range_ref
        if max_height is not None:
            keep = range_ <= max_height
            range_, beta = range_[keep], beta[:, keep]

        reduced = downsample(time, range_, beta, quality, max_time, max_range)
        reduced["time_start"] = float(time[0])
        reduced["time_end"] = float(time[-1])
        reduced["n_time_full"] = int(time.size)
        reduced["n_range_full"] = int(range_.size)
        reduced["variable"] = variable
        return reduced

    def read_layers(
        self,
        hours: float = 24.0,
        end: Optional[float] = None,
        start: Optional[float] = None,
        types: Optional[list[str]] = None,
        min_confidence: float = 0.0,
    ) -> list[Layer]:
        _, last = self.db.time_span()
        if last is None:
            return []
        t_end = float(end) if end is not None else float(last)
        t_start = float(start) if start is not None else t_end - hours * 3600.0
        return self.db.read_layers(t_start, t_end, types, min_confidence)

    def latest_profile(self, variable: str = "beta_att") -> dict[str, Any]:
        """The most recent single profile at full vertical resolution."""
        latest = self.db.latest_profile_time()
        if latest is None:
            return {}
        rows = self.db.grids_overlapping(latest, latest)
        if not rows:
            return {}
        grid = read_grid(Path(rows[-1]["path"]), variable=variable)
        j = int(np.argmin(np.abs(grid["time"] - latest)))
        layers = self.db.read_layers(latest - 0.5, latest + 0.5)
        quality = int(grid["quality"][j])
        return {
            "time": float(grid["time"][j]),
            "range": grid["range"].tolist(),
            "beta": [None if not np.isfinite(v) else float(v) for v in grid["beta"][j]],
            "quality": quality,
            "quality_flags": describe_quality(quality),
            "screened": bool(QualityFlag(quality).screened),
            "layers": [l.to_dict() for l in layers],
        }

    def status(self) -> dict[str, Any]:
        first, last = self.db.time_span()
        grids = self.db.all_grids()
        total = sum(
            Path(row["path"]).stat().st_size
            for row in grids if Path(row["path"]).exists()
        )
        db_bytes = self.db.path.stat().st_size if self.db.path.exists() else 0
        meta = self.db.get_meta()
        return {
            "time_start": first,
            "time_end": last,
            "hours_available": (
                (last - first) / 3600.0 if first is not None and last is not None else 0.0
            ),
            "n_grid_files": len(grids),
            "grid_bytes": int(total),
            "database_bytes": int(db_bytes),
            "retention_hours": self.config.store.retention_hours,
            "site_name": meta.get("site_name", self.config.site_name),
            "instrument": meta.get("instrument", self.config.instrument),
            "range_resolution_m": meta.get("range_resolution_m"),
            "max_range_m": meta.get("max_range_m"),
            "lowest_usable_height_m": meta.get("lowest_usable_height_m"),
            "last_ingest": meta.get("last_ingest"),
        }


def describe_quality(value: int) -> list[str]:
    """Human-readable names for a quality bitmask."""
    flags = QualityFlag(int(value))
    return [f.name.lower() for f in QualityFlag if f.value and f & flags]


def _empty_window() -> dict[str, Any]:
    return {
        "time": np.array([]), "range": np.array([]),
        "beta": np.zeros((0, 0)), "quality": np.array([], dtype="int64"),
        "time_factor": 1, "range_factor": 1,
        "time_start": None, "time_end": None,
        "n_time_full": 0, "n_range_full": 0, "variable": "beta_att",
    }
