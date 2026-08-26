"""Generate a realistic synthetic Vaisala CL31-style raw file with known truth.

No real ceilometer hardware or file is required to build and validate the whole
pipeline: this module synthesises attenuated backscatter from an explicit
atmospheric model, writes it out in the raw message format that
:mod:`blview.adapters.vaisala_cl` parses, and writes a sidecar JSON file
recording *exactly* which layers were injected at every timestamp.  The
validation harness (``scripts/validate.py``) then checks that the pipeline
recovers those injected layers.

Atmospheric model
-----------------
The profile is a sum of slabs, each with a smooth ``tanh`` edge:

* molecular (Rayleigh) background, scale height 8.5 km;
* a surface-connected **mixing layer** with a diurnal cycle;
* a nocturnal **residual layer** capping the previous day's mixed layer;
* a decoupled **elevated haze layer** at 2-3 km;
* **cloud** (a morning stratocumulus deck and afternoon cumulus at the mixing
  layer top);
* **fog** and **precipitation** episodes, which exist purely so the screening
  stage has something to screen.

Two-way attenuation ``exp(-2 * integral(lidar_ratio * beta) dr)`` is applied,
so cloud genuinely extinguishes the beam above it -- which is why cloud tops
are frequently not determinable, in the synthetic data exactly as in reality.

The instrument model then applies an incomplete near-range overlap function,
adds range-dependent detector noise (constant in raw photon units, therefore
growing as R^2 once the instrument range-corrects the profile), quantises to
the 20-bit two's-complement raw format and clips -- so dense cloud saturates,
again exactly as real CL31 data does.

All times are UTC and the diurnal cycle is keyed to the UTC hour of day
(ASSUMPTIONS.md #S1).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.vaisala_cl import INT20_MAX, PROFILE_UNIT_M1_SR1, crc16_vaisala

_HEXCHARS = np.frombuffer(b"0123456789ABCDEF", dtype=np.uint8)

#: Lidar ratios (extinction-to-backscatter, sr) used to build the attenuation.
LIDAR_RATIO_AEROSOL = 30.0
LIDAR_RATIO_CLOUD = 18.0
#: Rain/drizzle: large drops backscatter strongly for their extinction, so the
#: lidar ratio is far lower than for fine aerosol.  Without this, synthetic
#: precipitation would attenuate itself away within a few hundred metres.
LIDAR_RATIO_PRECIP = 12.0
LIDAR_RATIO_MOLECULAR = 8.0 * math.pi / 3.0


@dataclass
class SyntheticScenario:
    """Every knob of the synthetic day, with realistic defaults."""

    # ------------------------------------------------------------ sampling
    n_gates: int = 770                 #: CL31 in 10 m mode -> 7.7 km
    resolution_m: float = 10.0
    interval_s: float = 30.0           #: one profile every 30 s
    duration_h: float = 24.0

    # ------------------------------------------------------- mixing layer
    sml_night_m: float = 260.0         #: stable nocturnal layer depth
    sml_max_m: float = 1700.0          #: afternoon maximum
    growth_start_h: float = 6.0        #: ~1 h after sunrise
    growth_peak_h: float = 15.0
    collapse_start_h: float = 18.0
    collapse_end_h: float = 20.0

    # ------------------------------------------------------ residual layer
    rl_top_evening_m: float = 1620.0   #: inherited from the previous afternoon
    rl_top_dawn_m: float = 1360.0      #: subsides overnight
    rl_erosion_margin_m: float = 120.0  #: RL disappears once SML gets this close

    # --------------------------------------------------------- haze layer
    haze_base_start_m: float = 2450.0
    haze_base_end_m: float = 2120.0
    haze_depth_m: float = 680.0
    haze_start_h: float = 0.0
    haze_end_h: float = 17.0
    haze_ramp_h: float = 1.0           #: fade in/out time

    # -------------------------------------------------------------- cloud
    stratus_start_h: float = 8.5
    stratus_end_h: float = 11.0
    stratus_base_m: float = 1900.0
    stratus_thickness_m: float = 320.0
    cumulus_start_h: float = 13.5
    cumulus_end_h: float = 16.5
    cumulus_fraction: float = 0.45     #: fraction of profiles containing cumulus
    cumulus_thickness_m: float = 220.0

    # ------------------------------------------------- fog / precipitation
    fog_start_h: float = 3.0
    fog_end_h: float = 4.5
    precip_start_h: float = 21.0
    precip_end_h: float = 22.0

    # -------------------------------------------- backscatter magnitudes
    beta_molecular_surface: float = 1.7e-7   #: Rayleigh at 910 nm, m-1 sr-1
    beta_free_trop: float = 8.0e-8           #: residual aerosol aloft
    #: Mixing-layer aerosol backscatter at the *reference* depth
    #: ``sml_night_m``.  A fixed aerosol burden spread through a deeper layer
    #: is more dilute, so the value scales as (reference / depth)^exponent
    #: rather than switching between a day and a night value: a step there
    #: would make the mixing-layer/residual-layer contrast collapse
    #: discontinuously at one instant, which no real atmosphere does.
    beta_ml_reference: float = 6.5e-6
    beta_ml_floor: float = 2.6e-6            #: deep, well-diluted afternoon layer
    beta_ml_dilution_exponent: float = 0.5
    beta_residual: float = 2.4e-6
    beta_haze: float = 1.9e-6
    beta_cloud_peak: float = 3.0e-3
    beta_fog: float = 4.0e-4
    beta_precip: float = 3.0e-5

    # ------------------------------------------------------ instrument model
    #: Detector-noise standard deviation of the *range-corrected* profile at
    #: 3 km, m-1 sr-1.  Grows as R^2 either side of that reference.
    noise_beta_at_3km: float = 8.0e-7
    noise_daylight_factor: float = 1.6       #: extra solar background noise at noon
    speckle_fraction: float = 0.02           #: multiplicative signal noise
    #: True overlap function: full overlap at this height, with this exponent.
    #: Deliberately *different* from the model the preprocessor assumes, since
    #: the real overlap function is never known exactly (ASSUMPTIONS.md #S3).
    overlap_full_m: float = 180.0
    overlap_start_m: float = 15.0
    overlap_exponent: float = 1.6

    seed: int = 20260825
    site_name: str = "Synthetic Test Site"
    unit_id: str = "01"

    def ranges(self) -> np.ndarray:
        """Gate-centre heights, metres above the instrument."""
        return (np.arange(self.n_gates, dtype="float64") + 0.5) * self.resolution_m


# --------------------------------------------------------------------- model
def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    """Clamped cubic smoothstep on [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _slab(
    r: np.ndarray, z_bot: float | None, z_top: float | None,
    d_bot: float = 60.0, d_top: float = 100.0,
) -> np.ndarray:
    """Unit-amplitude slab between ``z_bot`` and ``z_top`` with tanh edges."""
    out = np.ones_like(r)
    if z_bot is not None:
        out = out * 0.5 * (1.0 + np.tanh((r - z_bot) / max(d_bot * 0.5, 1.0)))
    if z_top is not None:
        out = out * 0.5 * (1.0 - np.tanh((r - z_top) / max(d_top * 0.5, 1.0)))
    return out


def mixing_layer_height(hour: float, sc: SyntheticScenario) -> float:
    """Diurnal surface mixing-layer depth (m) at UTC hour-of-day ``hour``.

    Defined so that it is *continuous across midnight*: the nocturnal branch is
    parameterised on the wrapped interval [collapse_end_h, growth_start_h + 24]
    and starts and ends at ``sml_night_m``.  A discontinuity here would inject
    a step into the truth that no detector could reproduce.
    """
    if sc.growth_start_h <= hour < sc.growth_peak_h:
        frac = (hour - sc.growth_start_h) / (sc.growth_peak_h - sc.growth_start_h)
        # Convective growth: fast in late morning, flattening in the afternoon.
        return sc.sml_night_m + (sc.sml_max_m - sc.sml_night_m) * float(
            _smoothstep(frac ** 0.75)
        )
    if sc.growth_peak_h <= hour < sc.collapse_start_h:
        return sc.sml_max_m
    if sc.collapse_start_h <= hour < sc.collapse_end_h:
        frac = (hour - sc.collapse_start_h) / (sc.collapse_end_h - sc.collapse_start_h)
        return sc.sml_max_m + (sc.sml_night_m - sc.sml_max_m) * float(_smoothstep(frac))
    # Nocturnal: gentle deepening through the night, back to the minimum by
    # sunrise.  phase runs 0 -> 1 from collapse_end_h round to growth_start_h.
    night_span = (sc.growth_start_h + 24.0) - sc.collapse_end_h
    phase = ((hour - sc.collapse_end_h) % 24.0) / night_span
    return sc.sml_night_m * (1.0 + 0.22 * math.sin(math.pi * min(phase, 1.0)))


def residual_layer_top(hour: float, sc: SyntheticScenario, sml: float) -> float | None:
    """Nocturnal residual-layer top (m), or ``None`` when there is none.

    Like :func:`mixing_layer_height` this is continuous across midnight: the
    residual layer forms shortly after the evening collapse begins and subsides
    monotonically on the wrapped interval until it is eroded by the growing
    surface mixing layer the next morning.
    """
    form_h = sc.collapse_start_h + 0.3        # forms just after the collapse starts
    decay_span = (12.0 + 24.0) - form_h       # subsides until midday next day
    phase = ((hour - form_h) % 24.0) / decay_span
    if phase > 1.0:
        return None                            # fully mixed away by the afternoon
    top = sc.rl_top_evening_m + (sc.rl_top_dawn_m - sc.rl_top_evening_m) * float(phase)
    if sml >= top - sc.rl_erosion_margin_m:
        return None                            # entrained by the growing SML
    return float(top)


def haze_layer(hour: float, sc: SyntheticScenario) -> tuple[float, float, float] | None:
    """Elevated haze layer as ``(base_m, top_m, amplitude_fraction)``."""
    if hour < sc.haze_start_h or hour > sc.haze_end_h:
        return None
    span = max(sc.haze_end_h - sc.haze_start_h, 1e-6)
    frac = (hour - sc.haze_start_h) / span
    base = sc.haze_base_start_m + (sc.haze_base_end_m - sc.haze_base_start_m) * frac
    amp = float(
        _smoothstep((hour - sc.haze_start_h) / sc.haze_ramp_h)
        * _smoothstep((sc.haze_end_h - hour) / sc.haze_ramp_h)
    )
    return float(base), float(base + sc.haze_depth_m), amp


def atmosphere_state(
    time_utc: float, sc: SyntheticScenario, rng_cloud: float = 0.0
) -> dict[str, Any]:
    """Ground truth for one timestamp.

    ``rng_cloud`` is a pre-computed temporally-correlated random number in
    [0, 1] used to decide cumulus presence, so cumulus comes in clumps rather
    than flickering profile to profile.
    """
    stamp = dt.datetime.fromtimestamp(time_utc, dt.timezone.utc)
    hour = stamp.hour + stamp.minute / 60.0 + stamp.second / 3600.0

    fog = sc.fog_start_h <= hour < sc.fog_end_h
    precip = sc.precip_start_h <= hour < sc.precip_end_h

    sml = mixing_layer_height(hour, sc)
    rl = residual_layer_top(hour, sc, sml)
    haze = haze_layer(hour, sc)

    cloud_base: float | None = None
    cloud_top: float | None = None
    cloud_kind: str | None = None
    if sc.stratus_start_h <= hour < sc.stratus_end_h:
        cloud_base = sc.stratus_base_m
        cloud_top = sc.stratus_base_m + sc.stratus_thickness_m
        cloud_kind = "stratocumulus"
    elif sc.cumulus_start_h <= hour < sc.cumulus_end_h and rng_cloud < sc.cumulus_fraction:
        # Cumulus form at the lifting condensation level, just above the
        # mixing-layer top.
        cloud_base = sml + 90.0
        cloud_top = cloud_base + sc.cumulus_thickness_m
        cloud_kind = "cumulus"
    elif precip:
        cloud_base = 1150.0
        cloud_top = None                    # precipitating deck, beam extinguished
        cloud_kind = "precipitating"

    sml_top = None if fog else round(float(sml), 1)
    rl_top = None if (fog or rl is None) else round(float(rl), 1)
    haze_base = None if (fog or haze is None or haze[2] < 0.5) else round(haze[0], 1)
    haze_top = None if (fog or haze is None or haze[2] < 0.5) else round(haze[1], 1)

    # --- observability -------------------------------------------------
    # A layer that was *injected* is not necessarily *observable*: a ceilometer
    # beam is extinguished by cloud, so nothing above a cloud base can be
    # retrieved, and fog / precipitation profiles are screened out entirely
    # before detection ever runs.  The validation harness asserts recovery only
    # where these flags are true (ASSUMPTIONS.md #S4).
    screened = bool(fog or precip)

    def _visible(base: float | None) -> bool:
        if screened or base is None:
            return False
        if cloud_base is not None and base >= cloud_base - 50.0:
            return False
        return True

    return {
        "time": float(time_utc),
        "hour_utc": round(hour, 4),
        "fog": bool(fog),
        "precip": bool(precip),
        "screened": screened,
        "sml_top": sml_top,
        "rl_top": rl_top,
        "haze_base": haze_base,
        "haze_top": haze_top,
        "haze_amplitude": 0.0 if haze is None else round(haze[2], 4),
        "cloud_base": None if cloud_base is None else round(float(cloud_base), 1),
        "cloud_top": None if cloud_top is None else round(float(cloud_top), 1),
        "cloud_kind": cloud_kind,
        "sml_visible": _visible(sml_top),
        "rl_visible": _visible(rl_top),
        "haze_visible": _visible(haze_base),
        "cloud_visible": bool(cloud_base is not None and not screened),
    }


def build_profile(
    r: np.ndarray, state: dict[str, Any], sc: SyntheticScenario
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(beta_total, alpha_total)`` for one profile, before attenuation."""
    hour = state["hour_utc"]
    beta_mol = sc.beta_molecular_surface * np.exp(-r / 8500.0)
    beta = beta_mol + sc.beta_free_trop * np.exp(-r / 4000.0)
    alpha = LIDAR_RATIO_MOLECULAR * beta_mol

    aerosol = np.zeros_like(r)

    # --- surface mixing layer -------------------------------------------
    sml = state["sml_top"] or mixing_layer_height(hour, sc)
    beta_ml = float(
        np.clip(
            sc.beta_ml_reference
            * (sc.sml_night_m / max(sml, 1.0)) ** sc.beta_ml_dilution_exponent,
            sc.beta_ml_floor,
            sc.beta_ml_reference,
        )
    )
    entrain = max(0.10 * sml, 60.0)          # entrainment-zone depth
    ml_shape = _slab(r, None, sml, d_top=entrain)
    # Aerosol thins slightly with height inside the mixed layer.
    ml_shape = ml_shape * (1.0 - 0.15 * np.clip(r / max(sml, 1.0), 0.0, 1.0))
    aerosol += beta_ml * ml_shape

    # --- residual layer --------------------------------------------------
    if state["rl_top"] is not None:
        rl = state["rl_top"]
        aerosol += sc.beta_residual * _slab(
            r, sml, rl, d_bot=entrain, d_top=max(0.08 * rl, 80.0)
        )

    # --- elevated haze ---------------------------------------------------
    haze = haze_layer(hour, sc)
    if haze is not None and haze[2] > 0.0:
        base, top, amp = haze
        aerosol += sc.beta_haze * amp * _slab(r, base, top, d_bot=90.0, d_top=140.0)

    beta = beta + aerosol
    alpha = alpha + LIDAR_RATIO_AEROSOL * aerosol

    # --- fog -------------------------------------------------------------
    # Optically thick enough to be genuine full obscuration: extinction
    # ~1.2e-2 m-1 gives ~250 m visibility, so the beam is dead above ~300 m.
    if state["fog"]:
        fog = sc.beta_fog * _slab(r, None, 300.0, d_top=120.0)
        beta = beta + fog
        alpha = alpha + LIDAR_RATIO_AEROSOL * fog

    # --- precipitation ---------------------------------------------------
    if state["precip"]:
        precip = sc.beta_precip * _slab(r, None, 1200.0, d_bot=40.0, d_top=260.0)
        beta = beta + precip
        alpha = alpha + LIDAR_RATIO_PRECIP * precip

    # --- cloud -----------------------------------------------------------
    if state["cloud_base"] is not None:
        base = state["cloud_base"]
        top = state["cloud_top"]
        # Sharp base, peak ~40 m above it, then the attenuation integral does
        # the rest of the work.
        peak_h = base + 45.0
        width = 55.0
        cloud = np.exp(-(((r - peak_h) / width) ** 2))
        if top is not None:
            cloud = cloud + 0.35 * _slab(r, base, top, d_bot=25.0, d_top=60.0)
        cloud = cloud * (r >= base - 25.0)
        beta = beta + sc.beta_cloud_peak * cloud
        alpha = alpha + LIDAR_RATIO_CLOUD * sc.beta_cloud_peak * cloud

    return beta, alpha


def _instrument(
    beta: np.ndarray, alpha: np.ndarray, r: np.ndarray, hour: float,
    sc: SyntheticScenario, rng: np.random.Generator,
) -> np.ndarray:
    """Apply attenuation, overlap and detector noise -> measured beta_att."""
    dr = float(np.median(np.diff(r))) if r.size > 1 else sc.resolution_m
    optical_depth = np.cumsum(alpha) * dr
    transmission_sq = np.exp(-2.0 * optical_depth)
    beta_att = beta * transmission_sq

    # Incomplete near-range overlap.
    frac = np.clip(
        (r - sc.overlap_start_m) / max(sc.overlap_full_m - sc.overlap_start_m, 1e-6),
        0.0, 1.0,
    )
    beta_att = beta_att * frac ** sc.overlap_exponent

    # Detector noise: constant in raw photon units -> R^2 after the
    # instrument's own range correction.  Solar background raises it by day.
    daylight = 1.0 + (sc.noise_daylight_factor - 1.0) * max(
        0.0, math.cos((hour - 12.0) / 12.0 * math.pi / 2.0 * 2.0)
    )
    sigma = sc.noise_beta_at_3km * daylight * (r / 3000.0) ** 2
    noise = rng.normal(0.0, 1.0, size=r.shape) * sigma
    speckle = rng.normal(0.0, 1.0, size=r.shape) * sc.speckle_fraction * np.abs(beta_att)
    return beta_att + noise + speckle


# ----------------------------------------------------------------- generation
def generate(
    scenario: SyntheticScenario | None = None,
    start_time: float | None = None,
) -> dict[str, Any]:
    """Generate the synthetic dataset.

    Returns a dict with ``time``, ``range``, ``beta`` (the measured attenuated
    backscatter the instrument would report) and ``truth`` (a list of
    per-timestamp ground-truth dicts).
    """
    sc = scenario or SyntheticScenario()
    rng = np.random.default_rng(sc.seed)
    r = sc.ranges()
    n_time = int(round(sc.duration_h * 3600.0 / sc.interval_s))
    if start_time is None:
        # Default: a whole number of intervals ending at the most recent
        # interval boundary, so the API's "latest 24 h" window is full.
        now = dt.datetime.now(dt.timezone.utc).timestamp()
        end = math.floor(now / sc.interval_s) * sc.interval_s
        start_time = end - (n_time - 1) * sc.interval_s
    time = start_time + np.arange(n_time) * sc.interval_s

    # Temporally-correlated AR(1) process driving cumulus presence.
    cloud_noise = np.empty(n_time)
    x = 0.0
    rho = math.exp(-sc.interval_s / 420.0)      # ~7 min correlation time
    for i in range(n_time):
        x = rho * x + math.sqrt(1 - rho * rho) * rng.normal()
        cloud_noise[i] = x
    cloud_u = 0.5 * (1.0 + np.vectorize(math.erf)(cloud_noise / math.sqrt(2.0)))

    beta = np.empty((n_time, r.size))
    truth: list[dict[str, Any]] = []
    for i in range(n_time):
        state = atmosphere_state(float(time[i]), sc, float(cloud_u[i]))
        b, a = build_profile(r, state, sc)
        beta[i] = _instrument(b, a, r, state["hour_utc"], sc, rng)
        truth.append(state)

    return {
        "time": time,
        "range": r,
        "beta": beta,
        "truth": truth,
        "scenario": asdict(sc),
    }


# --------------------------------------------------------------------- output
def _encode_profile(counts: np.ndarray) -> str:
    """Vectorised 20-bit two's-complement 5-hex-character encoding."""
    clipped = np.clip(np.nan_to_num(counts, nan=0.0), -INT20_MAX - 1, INT20_MAX)
    u = (np.rint(clipped).astype("int64") & 0xFFFFF).astype("uint32")
    nib = np.stack([(u >> (4 * (4 - k))) & 0xF for k in range(5)], axis=-1)
    return _HEXCHARS[nib].tobytes().decode("ascii")


def _status_line(state: dict[str, Any], sc: SyntheticScenario) -> str:
    """Build the CL31 detection-status / cloud-base line."""
    if state["fog"]:
        # Status 4: full obscuration.  Field 1 = vertical visibility,
        # field 2 = highest signal detected.
        return "40 00090 00140 " + "0" * 8
    bases: list[float] = []
    if state["cloud_base"] is not None:
        bases.append(state["cloud_base"])
    digit = str(len(bases)) if bases else "0"
    fields = " ".join(f"{int(round(b)):05d}" for b in bases)
    return f"{digit}0 " + (fields + " " if fields else "") + "0" * 8


def write_vaisala_file(
    path: str | Path, data: dict[str, Any], scenario: SyntheticScenario | None = None
) -> Path:
    """Write the generated dataset as a logged CL31 raw message file."""
    sc = scenario or SyntheticScenario(**data["scenario"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    time = np.asarray(data["time"])
    beta = np.asarray(data["beta"])
    counts = beta / PROFILE_UNIT_M1_SR1        # -> raw profile units
    n_gates = beta.shape[1]

    with open(path, "w", encoding="ascii", newline="") as fh:
        fh.write(
            f"-- BL View synthetic Vaisala CL31 raw file, site '{sc.site_name}'\r\n"
            f"-- Generated by blview.synth.generate; SYNTHETIC DATA, NOT AN OBSERVATION\r\n"
        )
        for i in range(time.size):
            state = data["truth"][i]
            stamp = dt.datetime.fromtimestamp(float(time[i]), dt.timezone.utc)
            header = f"CL{sc.unit_id}0112"
            status = _status_line(state, sc)
            # scale, resolution, gates, laser energy, laser T, receiver
            # sensitivity, window contamination, instrument params,
            # background light, backscatter sum.
            # "Sum of detected and normalised backscatter" (sr-1, scaled by 1e4).
            bsum = int(np.clip(np.nansum(beta[i]) * sc.resolution_m * 1e4, 0, 999))
            params = (
                f"00100 {int(sc.resolution_m):02d} {n_gates:04d} 099 "
                f"{25 + int(6 * math.sin(state['hour_utc'] / 24 * 2 * math.pi)):+03d} "
                f"099 0000 L0112HN15 {int(30 + 100 * max(0.0, math.cos((state['hour_utc'] - 12) / 12 * math.pi))):03d} "
                f"{bsum:03d}"
            )
            profile = _encode_profile(counts[i])
            body = "\r\n".join([header, status, params, profile]) + "\r\n\x03"
            crc = crc16_vaisala(body.encode("ascii"))
            fh.write(f"-{stamp:%Y-%m-%d %H:%M:%S}\r\n")
            fh.write("\x01" + body + f"{crc:04x}\r\n\r\n")
    return path


def write_truth(path: str | Path, data: dict[str, Any]) -> Path:
    """Write the injected ground truth as JSON alongside the raw file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Ground truth injected by blview.synth.generate. Heights are metres "
            "above the instrument. Null means the feature was absent at that time."
        ),
        "scenario": data["scenario"],
        "states": data["truth"],
    }
    path.write_text(json.dumps(payload, indent=1))
    return path
