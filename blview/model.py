"""Core data model shared by every stage of the pipeline.

All heights are **metres above the instrument** (see ASSUMPTIONS.md: the
instrument is assumed to be ground-level and vertically pointing, so height
above instrument == height above ground level).  All times are UTC unix
seconds.  All backscatter is attenuated backscatter in m-1 sr-1.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np


class LayerType(str, enum.Enum):
    """Type assigned to a detected layer.

    The three aerosol types are *not* physically distinct measurements -- they
    are the same backscatter-gradient detection, labelled by where the layer
    sits relative to the surface and to the other layers in the same profile.
    """

    CLOUD = "cloud"
    MIXING_LAYER = "mixing_layer"
    RESIDUAL_LAYER = "residual_layer"
    HAZE = "haze"

    def __str__(self) -> str:  # keeps f-strings / JSON tidy
        return self.value


class QualityFlag(enum.IntFlag):
    """Per-profile quality bitmask.

    Stored as an integer alongside every profile so the UI can grey out
    screened periods instead of silently dropping them.
    """

    OK = 0
    PRECIPITATION = 1 << 0      #: deep surface-connected enhanced backscatter
    FOG = 1 << 1                #: full obscuration / signal extinguished near-surface
    LOW_SNR = 1 << 2            #: profile too noisy for layer detection
    SATURATED = 1 << 3          #: profile hit the raw-format dynamic-range clip
    INSTRUMENT_WARNING = 1 << 4  #: adapter reported a hardware warning
    INSTRUMENT_ALARM = 1 << 5   #: adapter reported a hardware alarm
    WINDOW_CONTAMINATED = 1 << 6  #: dirty window reported by instrument
    NO_DETECTION = 1 << 7       #: detection deliberately skipped for this profile

    @property
    def screened(self) -> bool:
        """True when the profile must be excluded from layer detection."""
        blocking = (
            QualityFlag.PRECIPITATION
            | QualityFlag.FOG
            | QualityFlag.LOW_SNR
            | QualityFlag.INSTRUMENT_ALARM
        )
        return bool(self & blocking)


@dataclass
class Layer:
    """One detected layer boundary at one timestamp.

    Attributes
    ----------
    type:
        See :class:`LayerType`.
    base_height:
        Metres above instrument.  For ``mixing_layer`` / ``residual_layer``
        this is the base of the aerosol layer (0 m for the surface-connected
        mixing layer); for ``cloud`` and ``haze`` it is the detected sharp
        bottom edge.
    top_height:
        Metres above instrument, or ``None`` when the top is not determinable
        (opaque cloud that extinguishes the beam, or a layer whose top is above
        the usable range of the instrument).
    confidence:
        0..1.  Combines edge strength relative to noise, persistence across
        wavelet dilation scales, contrast across the edge, and a penalty for
        detections inside the incomplete-overlap region.
    track_id:
        Identifier of the temporal track this detection was assigned to;
        ``None`` before tracking has run.
    interpolated:
        True when this detection was filled in across a short gap by the
        tracker rather than detected in this profile directly.
    """

    time: float
    type: LayerType
    base_height: float
    top_height: Optional[float] = None
    confidence: float = 0.0
    track_id: Optional[int] = None
    interpolated: bool = False
    #: free-form provenance (edge strength, detection scale, ...) -- kept for
    #: debugging and surfaced in the API but not part of the stable contract.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = str(self.type)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Layer":
        d = dict(d)
        d["type"] = LayerType(d["type"])
        return cls(**d)


@dataclass
class ProfileSet:
    """A time x range block of ceilometer profiles as returned by an adapter.

    This is the *only* structure downstream code sees, which is what makes the
    adapters pluggable: a new raw format needs to produce this and nothing
    else.

    Attributes
    ----------
    time:
        (n_time,) float64, UTC unix seconds, strictly increasing.
    range_:
        (n_range,) float64, metres above the instrument, strictly increasing,
        assumed (but not required) to be evenly spaced.
    beta:
        (n_time, n_range) float64 attenuated backscatter in m-1 sr-1, exactly
        as reported by the instrument.  May contain NaN for missing gates.
    range_corrected:
        True when ``beta`` already has the R^2 range correction applied by the
        instrument (Vaisala CL-series does).  Drives whether the preprocessor
        applies it.
    background_subtracted:
        True when the instrument already removed the detector background.
    quality:
        (n_time,) int64 bitmask of :class:`QualityFlag` values that the
        *adapter* could determine (instrument alarms, obscuration status).
        Preprocessing adds its own flags on top.
    cloud_base_reported:
        (n_time, 3) float64 -- cloud bases reported by the instrument's own
        firmware, NaN where absent.  Used only as a cross-check; BL View does
        its own cloud detection.
    """

    time: np.ndarray
    range_: np.ndarray
    beta: np.ndarray
    range_corrected: bool = True
    background_subtracted: bool = True
    quality: Optional[np.ndarray] = None
    cloud_base_reported: Optional[np.ndarray] = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype="float64")
        self.range_ = np.asarray(self.range_, dtype="float64")
        self.beta = np.asarray(self.beta, dtype="float64")
        if self.beta.shape != (self.time.size, self.range_.size):
            raise ValueError(
                f"beta shape {self.beta.shape} does not match "
                f"(n_time={self.time.size}, n_range={self.range_.size})"
            )
        if self.quality is None:
            self.quality = np.zeros(self.time.size, dtype="int64")
        else:
            self.quality = np.asarray(self.quality, dtype="int64")
        if self.cloud_base_reported is None:
            self.cloud_base_reported = np.full((self.time.size, 3), np.nan)

    @property
    def n_time(self) -> int:
        return int(self.time.size)

    @property
    def n_range(self) -> int:
        return int(self.range_.size)

    @property
    def range_resolution(self) -> float:
        """Median gate spacing in metres."""
        if self.range_.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.range_)))

    def concat(self, other: "ProfileSet") -> "ProfileSet":
        """Append ``other`` (must share the same range grid) after this set."""
        if not np.allclose(self.range_, other.range_):
            raise ValueError("cannot concatenate ProfileSets with different range grids")
        return ProfileSet(
            time=np.concatenate([self.time, other.time]),
            range_=self.range_,
            beta=np.vstack([self.beta, other.beta]),
            range_corrected=self.range_corrected,
            background_subtracted=self.background_subtracted,
            quality=np.concatenate([self.quality, other.quality]),
            cloud_base_reported=np.vstack(
                [self.cloud_base_reported, other.cloud_base_reported]
            ),
            attrs={**other.attrs, **self.attrs},
        )


@dataclass
class ProcessedProfiles:
    """Output of :func:`blview.preprocess.preprocess`.

    Carries three versions of the backscatter field, deliberately:

    ``beta_raw``
        Exactly what the adapter delivered.  Archived so a reprocessing run
        can start from the instrument's own numbers.
    ``beta``
        Range-corrected, background-offset removed, overlap-corrected, at full
        time resolution.  This is what the quicklook heatmap displays -- it is
        the crispest honest version of the data.
    ``beta_smooth``
        ``beta`` additionally averaged in time and height, with screened
        profiles excluded from the average.  This is what layer detection
        runs on; a single ceilometer profile is far too noisy to find a weak
        elevated gradient in.

    ``sigma`` is the propagated noise standard deviation **of beta_smooth**,
    which is what the detection significance thresholds are compared against.
    """

    time: np.ndarray
    range_: np.ndarray
    beta: np.ndarray
    beta_smooth: np.ndarray
    sigma: np.ndarray
    quality: np.ndarray
    beta_raw: Optional[np.ndarray] = None
    cloud_base_reported: Optional[np.ndarray] = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        return int(self.time.size)

    @property
    def n_range(self) -> int:
        return int(self.range_.size)

    @property
    def range_resolution(self) -> float:
        if self.range_.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.range_)))

    def screened_mask(self) -> np.ndarray:
        """(n_time,) bool -- True where the profile must not be used for detection."""
        blocking = int(
            QualityFlag.PRECIPITATION
            | QualityFlag.FOG
            | QualityFlag.LOW_SNR
            | QualityFlag.INSTRUMENT_ALARM
            | QualityFlag.NO_DETECTION
        )
        return (np.asarray(self.quality) & blocking) != 0
