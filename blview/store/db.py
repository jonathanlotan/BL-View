"""SQLite index: detected layers, per-profile quality, and the grid catalogue.

Why SQLite plus files rather than one or the other:

* Detected layers are small, are queried by time range and type, and want
  transactional replacement when a period is reprocessed -- that is a database.
* Backscatter grids are large, dense, homogeneous float arrays with coordinate
  axes and units -- that is netCDF, the format every atmospheric-science tool
  already reads. Putting megabyte BLOBs in SQLite would make the file
  unbrowsable by the tools the user already has.

So SQLite holds the layers and a *catalogue* of the grid files, and the grids
live next to it in ``grids/``. Both halves are plain files in the data
directory: nothing to install, nothing to administer, and a rolling window is
a ``DELETE`` plus an ``unlink``.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np

from ..model import Layer, LayerType

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS grid_files (
    path        TEXT PRIMARY KEY,
    time_start  REAL NOT NULL,
    time_end    REAL NOT NULL,
    n_time      INTEGER NOT NULL,
    n_range     INTEGER NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_grid_span ON grid_files (time_start, time_end);

CREATE TABLE IF NOT EXISTS layers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    time         REAL NOT NULL,
    type         TEXT NOT NULL,
    base_height  REAL NOT NULL,
    top_height   REAL,
    confidence   REAL NOT NULL,
    track_id     INTEGER,
    interpolated INTEGER NOT NULL DEFAULT 0,
    meta         TEXT
);
CREATE INDEX IF NOT EXISTS idx_layers_time ON layers (time);
CREATE INDEX IF NOT EXISTS idx_layers_type_time ON layers (type, time);

CREATE TABLE IF NOT EXISTS profiles (
    time     REAL PRIMARY KEY,
    quality  INTEGER NOT NULL,
    n_layers INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class LayerDB:
    """Thin wrapper over the SQLite index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------- writing
    def replace_layers(
        self, time_start: float, time_end: float, layers: Iterable[Layer]
    ) -> int:
        """Replace every layer in ``[time_start, time_end]``.

        Replacement rather than insertion makes reprocessing a period
        idempotent: re-ingesting the same raw files can never duplicate.
        """
        rows = [
            (
                l.time, str(l.type), l.base_height, l.top_height, l.confidence,
                l.track_id, int(l.interpolated),
                json.dumps(l.meta, default=_json_safe) if l.meta else None,
            )
            for l in layers
        ]
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM layers WHERE time >= ? AND time <= ?",
                (time_start, time_end),
            )
            conn.executemany(
                "INSERT INTO layers (time, type, base_height, top_height, "
                "confidence, track_id, interpolated, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def replace_profiles(
        self, times: np.ndarray, quality: np.ndarray, n_layers: np.ndarray
    ) -> int:
        rows = [
            (float(t), int(q), int(n))
            for t, q, n in zip(times, quality, n_layers)
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM profiles WHERE time >= ? AND time <= ?",
                (rows[0][0], rows[-1][0]),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO profiles (time, quality, n_layers) "
                "VALUES (?, ?, ?)",
                rows,
            )
        return len(rows)

    def register_grid(
        self, path: str, time_start: float, time_end: float,
        n_time: int, n_range: int, created_at: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO grid_files "
                "(path, time_start, time_end, n_time, n_range, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (path, time_start, time_end, n_time, n_range, created_at),
            )

    def set_meta(self, values: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [(k, json.dumps(v, default=_json_safe)) for k, v in values.items()],
            )

    # ------------------------------------------------------------- reading
    def get_meta(self) -> dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[row["key"]] = json.loads(row["value"])
            except (TypeError, json.JSONDecodeError):
                out[row["key"]] = row["value"]
        return out

    def read_layers(
        self,
        time_start: float,
        time_end: float,
        types: Optional[Iterable[str]] = None,
        min_confidence: float = 0.0,
    ) -> list[Layer]:
        sql = (
            "SELECT * FROM layers WHERE time >= ? AND time <= ? AND confidence >= ?"
        )
        params: list[Any] = [time_start, time_end, min_confidence]
        types = list(types) if types else None
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            params.extend(types)
        sql += " ORDER BY time, base_height"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            Layer(
                time=row["time"],
                type=LayerType(row["type"]),
                base_height=row["base_height"],
                top_height=row["top_height"],
                confidence=row["confidence"],
                track_id=row["track_id"],
                interpolated=bool(row["interpolated"]),
                meta=json.loads(row["meta"]) if row["meta"] else {},
            )
            for row in rows
        ]

    def read_profiles(self, time_start: float, time_end: float) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT time, quality, n_layers FROM profiles "
                "WHERE time >= ? AND time <= ? ORDER BY time",
                (time_start, time_end),
            ).fetchall()

    def grids_overlapping(self, time_start: float, time_end: float) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM grid_files WHERE time_end >= ? AND time_start <= ? "
                "ORDER BY time_start",
                (time_start, time_end),
            ).fetchall()

    def all_grids(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM grid_files ORDER BY time_start"
            ).fetchall()

    def time_span(self) -> tuple[Optional[float], Optional[float]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MIN(time_start) AS a, MAX(time_end) AS b FROM grid_files"
            ).fetchone()
        return (row["a"], row["b"]) if row else (None, None)

    def latest_profile_time(self) -> Optional[float]:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(time) AS t FROM profiles").fetchone()
        return row["t"] if row and row["t"] is not None else None

    # ------------------------------------------------------------ deleting
    def purge_before(self, cutoff: float) -> list[str]:
        """Delete layers/profiles before ``cutoff`` and return dropped grid paths.

        A grid file is only dropped when it is *entirely* older than the
        cutoff -- a file straddling it still holds data inside the window.
        """
        with self.connect() as conn:
            paths = [
                row["path"] for row in conn.execute(
                    "SELECT path FROM grid_files WHERE time_end < ?", (cutoff,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM grid_files WHERE time_end < ?", (cutoff,))
            conn.execute("DELETE FROM layers WHERE time < ?", (cutoff,))
            conn.execute("DELETE FROM profiles WHERE time < ?", (cutoff,))
        return paths

    def unregister_grids(self, paths: Iterable[str]) -> None:
        paths = list(paths)
        if not paths:
            return
        with self.connect() as conn:
            conn.executemany("DELETE FROM grid_files WHERE path = ?",
                             [(p,) for p in paths])


def _json_safe(obj: Any) -> Any:
    """Make numpy scalars JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
