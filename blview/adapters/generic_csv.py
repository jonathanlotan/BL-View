"""Generic gridded-CSV adapter.

This adapter exists for two reasons:

1. It is the worked example of how to plug a *different* raw format into BL
   View without touching preprocessing, detection, storage or the API.
2. Unlike the Vaisala CL-series (whose firmware range-corrects the profile
   before it leaves the instrument), this format defaults to
   ``range_corrected = false``, so the R^2 correction path in
   :mod:`blview.preprocess` is exercised by real data rather than only by unit
   tests.

Layout::

    # blview-csv v1
    # range_corrected = false
    # background_subtracted = false
    # beta_units = m-1 sr-1
    # site = Example
    time,10.0,20.0,30.0, ...        <- header row: "time" then the range grid (m)
    2026-08-25T00:00:00Z,1.2e-6,9.8e-7, ...
    ...

``NaN`` (or an empty field) marks a missing gate.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from ..model import ProfileSet
from .base import AdapterError, CeilometerAdapter, register_adapter

_TRUE = {"1", "true", "yes", "y", "on"}


def _parse_time(token: str) -> float:
    token = token.strip()
    try:                                   # plain unix seconds
        return float(token)
    except ValueError:
        pass
    t = token.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(t)
    if parsed.tzinfo is None:              # naive timestamps are assumed UTC
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


@register_adapter
class GenericCSVAdapter(CeilometerAdapter):
    """Reader for the simple ``time x range`` CSV described above."""

    name = "generic_csv"
    description = "Generic gridded CSV (time rows x range columns) with '#' metadata"
    patterns = ("*.csv", "*.csv.gz")

    def __init__(self, beta_scale: float = 1.0, **options: Any) -> None:
        super().__init__(**options)
        self.beta_scale = float(beta_scale)

    @classmethod
    def sniff(cls, path: str | Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for _ in range(20):
                    line = fh.readline()
                    if not line:
                        return False
                    if line.startswith("# blview-csv"):
                        return True
                    if line.startswith("#"):
                        continue
                    return line.lower().lstrip().startswith("time,")
        except OSError:
            return False
        return False

    def read(self, path: str | Path) -> ProfileSet:
        path = Path(path)
        meta: dict[str, str] = {}
        rows: list[list[str]] = []
        header: list[str] | None = None

        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            for line in fh:
                if line.startswith("#"):
                    body = line[1:].strip()
                    for sep in ("=", ":"):
                        if sep in body:
                            k, v = body.split(sep, 1)
                            meta[k.strip().lower()] = v.strip()
                            break
                    continue
                if not line.strip():
                    continue
                fields = next(csv.reader([line]))
                if header is None:
                    header = fields
                else:
                    rows.append(fields)

        if header is None or not rows:
            raise AdapterError(f"{path}: no data rows found")

        try:
            range_ = np.array([float(c) for c in header[1:]], dtype="float64")
        except ValueError as exc:
            raise AdapterError(
                f"{path}: header row must be 'time' followed by numeric range gates"
            ) from exc

        n_time, n_range = len(rows), range_.size
        time = np.empty(n_time, dtype="float64")
        beta = np.full((n_time, n_range), np.nan)
        for i, row in enumerate(rows):
            time[i] = _parse_time(row[0])
            vals = row[1: n_range + 1]
            for j, v in enumerate(vals):
                v = v.strip()
                if v and v.lower() not in {"nan", "na", "null"}:
                    try:
                        beta[i, j] = float(v)
                    except ValueError:
                        beta[i, j] = np.nan
        beta *= self.beta_scale

        order = np.argsort(time, kind="stable")
        return ProfileSet(
            time=time[order],
            range_=range_,
            beta=beta[order],
            range_corrected=meta.get("range_corrected", "false").lower() in _TRUE,
            background_subtracted=meta.get("background_subtracted", "false").lower() in _TRUE,
            attrs={
                "source_file": str(path),
                "adapter": self.name,
                "instrument_resolution_m": float(np.median(np.diff(range_)))
                if range_.size > 1 else float("nan"),
                "instrument_n_gates": int(n_range),
                **{f"csv_{k}": v for k, v in meta.items()},
            },
        )

    # ------------------------------------------------------------------ write
    @staticmethod
    def write(path: str | Path, profiles: ProfileSet) -> Path:
        """Write a :class:`ProfileSet` back out in this format (round-trip aid)."""
        path = Path(path)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("# blview-csv v1\n")
            fh.write(f"# range_corrected = {str(profiles.range_corrected).lower()}\n")
            fh.write(
                f"# background_subtracted = {str(profiles.background_subtracted).lower()}\n"
            )
            fh.write("# beta_units = m-1 sr-1\n")
            fh.write("time," + ",".join(f"{r:g}" for r in profiles.range_) + "\n")
            for i in range(profiles.n_time):
                stamp = dt.datetime.fromtimestamp(
                    profiles.time[i], dt.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                fh.write(stamp + "," + ",".join(f"{v:.6g}" for v in profiles.beta[i]) + "\n")
        return path
