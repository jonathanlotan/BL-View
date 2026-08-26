"""Pluggable raw-format adapter interface.

Downstream code (preprocess / detect / store / api) only ever sees a
:class:`~blview.model.ProfileSet`.  Supporting a new instrument or file format
therefore means writing one class here and registering it -- nothing else in
the pipeline changes.

To add an adapter::

    from blview.adapters.base import CeilometerAdapter, register_adapter

    @register_adapter
    class MyAdapter(CeilometerAdapter):
        name = "my_format"
        description = "..."

        @classmethod
        def sniff(cls, path): ...     # cheap "is this my format?" test
        def read(self, path): ...     # -> ProfileSet
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterable, Iterator, Type

from ..model import ProfileSet


class AdapterError(Exception):
    """Raised when a file cannot be parsed by the selected adapter."""


class CeilometerAdapter(abc.ABC):
    """Base class for raw-format readers.

    Subclasses must set :attr:`name` and implement :meth:`read` and
    :meth:`sniff`.
    """

    #: Short identifier used on the command line (``--adapter vaisala_cl``).
    name: str = "base"
    #: One-line human description shown by ``blview adapters``.
    description: str = ""
    #: Filename globs this adapter typically handles; used to order sniffing.
    patterns: tuple[str, ...] = ()

    def __init__(self, **options: object) -> None:
        self.options = options

    # ------------------------------------------------------------------ api
    @abc.abstractmethod
    def read(self, path: str | Path) -> ProfileSet:
        """Parse one raw file into a :class:`ProfileSet`."""

    @classmethod
    @abc.abstractmethod
    def sniff(cls, path: str | Path) -> bool:
        """Cheap test: does ``path`` look like this adapter's format?

        Must not raise on binary or truncated files -- return False instead.
        """

    # ------------------------------------------------------- convenience --
    def read_many(self, paths: Iterable[str | Path]) -> ProfileSet:
        """Read and concatenate several files in chronological order."""
        sets = [self.read(p) for p in sorted(paths, key=str)]
        sets = [s for s in sets if s.n_time]
        if not sets:
            raise AdapterError("no profiles found in the supplied files")
        sets.sort(key=lambda s: float(s.time[0]))
        out = sets[0]
        for s in sets[1:]:
            out = out.concat(s)
        order = out.time.argsort(kind="stable")
        out.time = out.time[order]
        out.beta = out.beta[order]
        out.quality = out.quality[order]
        out.cloud_base_reported = out.cloud_base_reported[order]
        return out


# --------------------------------------------------------------- registry --
_REGISTRY: dict[str, Type[CeilometerAdapter]] = {}


def register_adapter(cls: Type[CeilometerAdapter]) -> Type[CeilometerAdapter]:
    """Class decorator adding an adapter to the global registry."""
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"adapter name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def available_adapters() -> dict[str, Type[CeilometerAdapter]]:
    """All registered adapters, keyed by :attr:`CeilometerAdapter.name`."""
    return dict(_REGISTRY)


def get_adapter(name: str, **options: object) -> CeilometerAdapter:
    """Instantiate an adapter by name."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise AdapterError(
            f"unknown adapter {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None
    return cls(**options)


def detect_adapter(path: str | Path, **options: object) -> CeilometerAdapter:
    """Pick an adapter for ``path`` by asking each registered one to sniff it."""
    path = Path(path)
    candidates = sorted(
        _REGISTRY.values(),
        key=lambda c: 0 if any(path.match(p) for p in c.patterns) else 1,
    )
    for cls in candidates:
        try:
            if cls.sniff(path):
                return cls(**options)
        except Exception:  # a broken sniff must never break format detection
            continue
    raise AdapterError(
        f"no registered adapter recognises {path}; "
        f"pass --adapter explicitly (available: {', '.join(sorted(_REGISTRY))})"
    )


def iter_input_files(inputs: Iterable[str | Path]) -> Iterator[Path]:
    """Expand a list of files/directories/globs into concrete file paths."""
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.is_file())
        elif any(ch in str(item) for ch in "*?["):
            yield from sorted(Path().glob(str(item)))
        else:
            yield p
