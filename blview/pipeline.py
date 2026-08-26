"""End-to-end orchestration: ingest -> preprocess -> detect -> track."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .adapters import CeilometerAdapter, detect_adapter, get_adapter, iter_input_files
from .config import Config
from .detect import detect_layers, track_layers
from .model import Layer, ProcessedProfiles, ProfileSet
from .preprocess import preprocess

log = logging.getLogger("blview.pipeline")


@dataclass
class PipelineResult:
    """Everything one pipeline run produced."""

    profiles: ProfileSet
    processed: ProcessedProfiles
    layers: list[list[Layer]]
    config: Config
    adapter_name: str

    @property
    def all_layers(self) -> list[Layer]:
        return [layer for profile in self.layers for layer in profile]


def read_profiles(
    inputs: Iterable[str | Path],
    adapter: str | CeilometerAdapter | None = None,
    adapter_options: Optional[dict] = None,
) -> tuple[ProfileSet, str]:
    """Read raw files with an explicit, named, or auto-detected adapter."""
    paths = list(iter_input_files(inputs))
    if not paths:
        raise FileNotFoundError(f"no input files matched {list(inputs)!r}")

    options = adapter_options or {}
    if isinstance(adapter, CeilometerAdapter):
        reader = adapter
    elif isinstance(adapter, str):
        reader = get_adapter(adapter, **options)
    else:
        reader = detect_adapter(paths[0], **options)
        log.info("auto-detected adapter %r for %s", reader.name, paths[0])

    return reader.read_many(paths), reader.name


def run_pipeline(
    inputs: Iterable[str | Path],
    config: Config | None = None,
    adapter: str | CeilometerAdapter | None = None,
    adapter_options: Optional[dict] = None,
    track: bool = True,
) -> PipelineResult:
    """Run every stage over ``inputs`` and return the result in memory."""
    config = config or Config()
    profiles, adapter_name = read_profiles(inputs, adapter, adapter_options)
    log.info(
        "ingested %d profiles x %d gates (%.0f m resolution) via %s",
        profiles.n_time, profiles.n_range, profiles.range_resolution, adapter_name,
    )

    processed = preprocess(profiles, config)
    log.info(
        "preprocessed: %s (%d profiles screened)",
        processed.attrs.get("preprocess_notes", ""),
        processed.attrs.get("n_screened", 0),
    )

    layers = detect_layers(processed, config)
    if track:
        layers = track_layers(layers, processed.time, config)
    log.info("detected %d layers over %d profiles",
             sum(len(x) for x in layers), processed.n_time)

    return PipelineResult(
        profiles=profiles, processed=processed, layers=layers,
        config=config, adapter_name=adapter_name,
    )
