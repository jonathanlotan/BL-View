"""Tunable configuration for the whole pipeline.

Every threshold BL View uses lives here, with the reasoning for its default in
the docstring/comment next to it, and is mirrored in ASSUMPTIONS.md.  Nothing
downstream may hard-code a magic number: if you find one, it is a bug.

A config can be overridden from a JSON file (``--config my.json``) so the tool
can be retuned for a different instrument or site without editing code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class PreprocessConfig:
    """Preprocessing thresholds."""

    #: Range below which the receiver field-of-view / laser beam overlap is
    #: assumed incomplete.  The true overlap function is instrument- and
    #: alignment-specific and is NOT published by Vaisala, so BL View uses a
    #: generic parametric model up to this height (ASSUMPTIONS.md #P1).
    overlap_full_m: float = 200.0

    #: Range at which the modelled overlap first becomes non-zero.  Below this
    #: the correction is not attempted and gates are masked.
    overlap_start_m: float = 30.0

    #: Shape parameter of the modelled overlap ramp (see preprocess.overlap_function).
    #: 2.0 gives a smooth S-curve reaching ~1 at overlap_full_m.
    overlap_shape: float = 2.0

    #: Maximum factor by which the overlap correction is allowed to amplify the
    #: signal.  Caps noise blow-up in the lowest gates.
    overlap_max_gain: float = 8.0

    #: Fractional uncertainty in the modelled overlap function itself.  Since
    #: the true overlap is not published, the correction is wrong by some
    #: amount that grows as the correction grows.  This is a *correlated*
    #: error across neighbouring gates, so it does not average away and it is
    #: not noise -- it is propagated into the edge-detection significance as a
    #: bias term (see detect.detector).  Without it, the residual of an
    #: imperfect overlap correction is itself detected as a shallow layer.
    overlap_uncertainty: float = 0.5

    #: Bottom of the range window used to estimate the residual detector
    #: background / noise floor.  Must be above any realistic aerosol.
    noise_ref_bottom_m: float = 6000.0

    #: Top of that window; defaults to the top of the profile when None.
    noise_ref_top_m: float | None = None

    #: Number of profiles either side used when smoothing the noise estimate in
    #: time (a single profile gives a very uncertain sigma).
    noise_time_smooth: int = 15

    #: Multiplicative ("speckle") noise as a fraction of the signal itself.
    #: Without this the pure R^2 detector-noise model implies near-zero noise
    #: close to the ground, which would make every trivial low-level wiggle
    #: statistically significant.  Added in quadrature with the R^2 term.
    speckle_fraction: float = 0.02

    #: --- Precipitation screening ----------------------------------------
    #: Precipitation is *surface-connected enhanced backscatter filling a deep
    #: layer*.  The discriminator is the "fill depth": the height up to which
    #: backscatter stays continuously above precip_beta_threshold, starting
    #: from precip_bottom_m and tolerating short dropouts.  Cloud does not
    #: trigger it (not surface-connected) and mixing-layer aerosol does not
    #: reach the threshold.  Evaluated BEFORE fog, because a precipitating
    #: profile also looks extinguished aloft.
    precip_bottom_m: float = 100.0
    precip_beta_threshold: float = 1.0e-5   # m-1 sr-1; well above ML aerosol
    precip_min_depth_m: float = 800.0
    precip_gap_tolerance_m: float = 150.0   # dropout tolerated within the fill

    #: --- Fog / full-obscuration screening -------------------------------
    #: Fog is a *strong but shallow* near-surface return that extinguishes the
    #: beam: peak backscatter below fog_probe_m exceeds fog_beta_threshold AND
    #: the profile above fog_extinction_height_m has collapsed to the noise
    #: floor.  The shallowness requirement (fill depth below the precipitation
    #: depth) is what separates fog from precipitation.
    fog_probe_m: float = 300.0
    fog_beta_threshold: float = 3.0e-5      # m-1 sr-1, on the peak not the mean
    fog_extinction_height_m: float = 600.0
    fog_extinction_span_m: float = 1400.0   # band above that height to test
    fog_extinction_snr: float = 2.0         # median SNR in that band

    #: --- Low-SNR screening ----------------------------------------------
    #: Minimum median SNR in the 200-1500 m band for a profile to be usable.
    min_profile_snr: float = 1.5

    #: Vertical smoothing (in metres) applied before detection.  Small enough
    #: not to blur real layer edges at CL31's 10 m gate spacing.
    smooth_vertical_m: float = 40.0

    #: Temporal pre-averaging window in seconds.  Ceilometer profiles are very
    #: noisy individually; BL-View-style products routinely average a few
    #: minutes before layer detection.
    smooth_time_s: float = 300.0


@dataclass
class DetectConfig:
    """Layer-detection thresholds."""

    #: Haar wavelet dilations (metres).  Small scales localise sharp edges
    #: (nocturnal surface layer, cloud base); large scales find the broad,
    #: weak gradients at the top of a deep afternoon mixing layer.
    scales_m: tuple[float, ...] = (60.0, 120.0, 240.0, 480.0, 960.0)

    #: Significance threshold: |W| must exceed this multiple of the wavelet
    #: transform's own propagated noise standard deviation.
    snr_k: float = 3.0

    #: An edge must also carry at least this fraction of the strongest edge in
    #: the profile (at the same scale) to be kept -- suppresses a forest of
    #: marginal extrema in clean profiles.
    min_relative_strength: float = 0.10

    #: Absolute floor on |W| in log10(backscatter) units.  Inside a mixing
    #: layer the SNR reaches several hundred, so a 1 % ripple is *statistically*
    #: significant while being physically meaningless; the significance test
    #: alone then picks a trivial wiggle as the mixing-layer top in preference
    #: to the real edge above it.  W is roughly half the log10 change across an
    #: edge, so 0.03 corresponds to a ~15 % change in backscatter -- the
    #: smallest step worth calling a layer boundary.
    min_edge_log_amplitude: float = 0.03

    #: Lowest height at which any layer boundary may be reported.  Between
    #: here and overlap_full_m detections are allowed but confidence-penalised.
    #: The default matches the lowest gate that survives the overlap-correction
    #: gain cap (see preprocess.overlap_function): correcting below that would
    #: mean dividing by an overlap value small enough to manufacture gradients
    #: out of noise.
    min_layer_height_m: float = 90.0

    #: Highest height at which an *aerosol* layer may be reported.  Above this
    #: CL31 SNR is too poor for a trustworthy gradient.
    max_layer_height_m: float = 5000.0

    #: Edges found within this distance at different scales are treated as the
    #: same physical edge (the finest scale supplies the height).
    scale_merge_tolerance_m: float = 150.0

    #: --- Cloud classification -------------------------------------------
    #: Attenuated backscatter above which a return is cloud rather than
    #: aerosol.  Aerosol in a polluted mixing layer peaks around 1e-5;
    #: liquid cloud base is 1e-4..1e-3.  1e-4 sits in the gap.
    cloud_beta_threshold: float = 1.0e-4    # m-1 sr-1
    #: Number of consecutive gates that must exceed the threshold (rejects
    #: single-gate spikes / cosmic-ray-like artefacts).
    cloud_min_gates: int = 2
    #: Cloud peak must exceed this ratio times the median backscatter in the
    #: 300 m below its base -- enforces the "sharp return" requirement.
    cloud_sharpness_ratio: float = 8.0
    #: Gates this far below a detected cloud base, and everything above them,
    #: are masked out before the aerosol transforms are computed.  Excluding
    #: cloud only from the *candidate* list is not enough: a cloud sitting in
    #: the upper half of a Haar window swamps the difference of the half-window
    #: means and erases the mixing-layer edge below it.  Masking makes the
    #: contaminated windows return NaN instead, so the same edge is still found
    #: at the smaller dilations whose windows stay clear of the cloud.
    cloud_mask_margin_m: float = 50.0
    #: Above a cloud, if the signal stays within this multiple of the noise
    #: floor for cloud_opaque_probe_m the beam is considered extinguished and
    #: the cloud top is reported as None.
    cloud_opaque_snr: float = 2.0
    cloud_opaque_probe_m: float = 300.0

    #: --- Aerosol layer classification -----------------------------------
    #: A surface-connected layer's mean backscatter must exceed this multiple
    #: of the air just above it for its top to be a mixing-layer top.
    #: Deliberately low: when a growing mixing layer is eating into a residual
    #: layer, the contrast across the SML/RL interface is only ~1.5, and a
    #: stricter gate makes the detector skip it and report the *residual*
    #: layer top as the mixing height -- the single worst failure mode of a
    #: single-height retrieval, and the one this tool exists to avoid.  Noise
    #: is guarded against by the wavelet significance test and
    #: min_edge_log_amplitude, not by this gate.
    surface_layer_min_contrast: float = 1.15
    #: An elevated (residual/haze) layer's mean backscatter must exceed this
    #: multiple of the air just above its top to be reported at all.
    elevated_layer_min_contrast: float = 2.0
    #: ...and must itself be this many standard deviations above the noise.
    #: Contrast alone is not enough: two noise excursions can have a large
    #: ratio while neither is real.
    elevated_layer_min_snr: float = 5.0
    #: Minimum base height for a layer to be called *haze* rather than a
    #: residual layer: haze is decoupled and sits well above the mixed layer.
    haze_min_base_m: float = 800.0
    #: A candidate elevated layer is "decoupled" (haze) rather than a residual
    #: layer when the backscatter minimum between it and the layer below drops
    #: to this fraction of the elevated layer's own mean backscatter.
    decoupling_ratio: float = 0.6
    #: Minimum geometric depth of an elevated layer to be reported.
    min_elevated_depth_m: float = 200.0
    #: Minimum depth for a *residual* layer specifically.  The entrainment
    #: zone immediately above a mixing-layer top is genuine elevated aerosol,
    #: passes every contrast and SNR test, and is not a residual layer -- it is
    #: the mixing layer's own upper transition.  Depth is what separates them:
    #: an entrainment zone scales with ~10% of the mixing depth, a residual
    #: layer is hundreds of metres to kilometres deep.  Measured across seven
    #: synthetic days at different diurnal phases, raising this from 300 m to
    #: 500 m takes residual-layer false positives from 3-20% to 0% for about
    #: 2 points of detection rate: below ~500 m the two are not separable at
    #: this vertical resolution, so claiming a residual layer there is a guess.
    min_residual_depth_m: float = 500.0

    #: --- Confidence weighting -------------------------------------------
    #: Confidence multiplier applied to detections below overlap_full_m.
    overlap_confidence_penalty: float = 0.6


@dataclass
class TrackConfig:
    """Temporal continuity / tracking parameters."""

    #: Base tolerance for associating a detection with an existing track.
    max_jump_m: float = 250.0
    #: Additional tolerance per minute of gap, covering genuine layer motion
    #: (morning mixing-layer growth reaches ~500 m/h = ~8 m/min; the allowance
    #: is deliberately generous because detection noise dominates).
    max_jump_m_per_min: float = 120.0
    #: A track with no detection for longer than this is closed.
    max_gap_s: float = 900.0
    #: Tracks shorter than this many profiles are discarded as flicker.
    min_track_profiles: int = 5
    #: Width (in profiles) of the running median applied to a track's heights.
    #: Must be odd.
    smooth_profiles: int = 5
    #: Fill short gaps inside a track by linear interpolation.
    fill_gaps: bool = True


@dataclass
class StoreConfig:
    """Where processed data lives and how long it is kept."""

    data_dir: Path = Path("data")
    #: Rolling-window retention.  Older grids/layers are purged on ingest.
    retention_hours: float = 72.0
    #: Hours of data per netCDF grid file.
    grid_file_hours: float = 6.0


@dataclass
class Config:
    """Top-level configuration object passed through the pipeline."""

    site_name: str = "Synthetic Test Site"
    instrument: str = "Vaisala CL31 (synthetic)"
    #: Height of the instrument's first gate above ground level, metres.
    #: Assumed 0 -- see ASSUMPTIONS.md #G2.
    instrument_altitude_agl_m: float = 0.0

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    store: StoreConfig = field(default_factory=StoreConfig)

    # ---------------------------------------------------------------- io --
    def to_dict(self) -> dict[str, Any]:
        def _enc(o: Any) -> Any:
            if is_dataclass(o):
                return {f.name: _enc(getattr(o, f.name)) for f in fields(o)}
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return o

        return _enc(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d)
        sub = {
            "preprocess": PreprocessConfig,
            "detect": DetectConfig,
            "track": TrackConfig,
            "store": StoreConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in d.items():
            if key in sub:
                klass = sub[key]
                valid = {f.name for f in fields(klass)}
                inner = {k: v for k, v in value.items() if k in valid}
                if key == "store" and "data_dir" in inner:
                    inner["data_dir"] = Path(inner["data_dir"])
                if key == "detect" and "scales_m" in inner:
                    inner["scales_m"] = tuple(float(s) for s in inner["scales_m"])
                kwargs[key] = klass(**inner)
            elif key in {f.name for f in fields(cls)}:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


DEFAULT_CONFIG = Config()
