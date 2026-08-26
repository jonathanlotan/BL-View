# Assumptions

Every place the specification was ambiguous, or where a fact about real
hardware was unavailable, BL View makes an explicit assumption rather than
guessing silently. They are all listed here, grouped by pipeline stage, and
each is referenced from a comment at the point in the code where it bites.

Anything marked **[tunable]** is a named constant in `blview/config.py` (or an
adapter option) and can be changed without editing logic.

---

## G — General / scientific framing

**#G1 — "Inversion" means *aerosol gradient*, never temperature.**
The specification explicitly forbids claiming a thermodynamic inversion
temperature, and nothing in this tool computes one. Every height BL View
reports is the height of a **gradient in attenuated aerosol backscatter**.
Aerosol layering is frequently co-located with a thermodynamic inversion —
which is why it is a useful proxy — but the two can decouple badly (see
README "Known limitations"). All UI and API copy says "aerosol gradient".

**#G2 — Instrument geometry: ground-level and vertically pointing.**
Heights are reported as **metres above the instrument**, and are assumed equal
to metres above ground level. Real CL31/CL51 units are often tilted a few
degrees off zenith (to shed rain) and mounted on a roof. Neither is corrected
for: `Config.instrument_altitude_agl_m` exists to record the offset but
defaults to `0.0`, and no cosine tilt correction is applied. For a tilted
instrument, true vertical height is `reported x cos(tilt)` — a 4° tilt is a
0.24 % error, far below the detection tolerances here.

**#G3 — All times are UTC unix seconds; no local time or DST anywhere.**
Timestamps in raw files without an explicit zone are read as UTC.

**#G4 — Internal units are fixed and unconverted.**
Attenuated backscatter `m^-1 sr^-1`, range/height `m`, time `s`. Adapters
convert into these; nothing downstream reconverts. Vaisala can be configured
to report heights in feet — this is **not** auto-detected (see #V3).

**#G5 — A single vertical range grid per ingest.**
All profiles in one file must share one gate spacing and gate count; a file
mixing instrument configurations is rejected rather than silently regridded.

---

## V — Vaisala CL31/CL51 raw message format

The CL-series data message is proprietary and only semi-documented. The
structure below is what the public manuals describe consistently, and is what
`blview/synth/generator.py` writes and `blview/adapters/vaisala_cl.py` reads.

**#V1 — Message layout ("message 2", no sky-condition block).**

```
-2026-08-25 00:00:00                                      logger timestamp line
<SOH>CL010112                                             header
30 01230 12340 23450 FEDCBA98                             detection status
00100 10 0770 099 +34 099 0000 L0112HN15 139 026          parameters
<n_gates x 5 hex characters>                              profile
<ETX>1a2b                                                 CRC-16
```

The leading `-YYYY-MM-DD hh:mm:ss` timestamp line is the convention used by
Vaisala's own logging software, not part of the instrument message itself.

**#V2 — Parameter-line field meanings.** Assumed, in order:
scale (%), range resolution (m), number of gates, laser pulse energy (%),
laser temperature (°C), receiver sensitivity (%), window contamination (mV),
*instrument measurement-parameter string*, background light (mV), backscatter
sum (sr^-1 x 1e4). The 8th token (`L0112HN15`) has no published layout, so it
is **stored verbatim and never interpreted**.

**#V3 — Reported heights are metres, not feet.**
CL31/CL51 can be configured to report cloud bases in either. There is no
reliable flag for this in the message, so metres is assumed. A unit configured
for feet will report cloud bases ~3.3x too high. Only affects the
instrument's *own* cloud bases, which BL View uses solely as a cross-check —
its own cloud detection works from the profile and is unaffected.

**#V4 — Profile counts are `1e-9 m^-1 sr^-1` per unit. [tunable]**
This scaling is the least well documented part of the format. It is a single
named constant `PROFILE_UNIT_M1_SR1` and an adapter option `profile_unit=`, so
a deployment that calibrates differently changes one number. Choosing this
wrong rescales every backscatter value by a constant — which shifts the cloud
threshold (#D5) but leaves all *gradient*-based layer heights unchanged.

**#V5 — CRC-16 is reflected CCITT (poly `0x8408`, init `0xFFFF`, final
complement), computed over everything after `<SOH>` through `<ETX>` inclusive.**
The exact seeding is not published. A CRC mismatch is therefore treated as a
**quality flag, never a parse failure** (`strict_crc=True` opts into flagging
such profiles as low-SNR). Real loggers also rewrite line endings, which would
otherwise break the checksum on data that is perfectly good.

**#V6 — Gate `i` is reported at its centre, `(i + 0.5) x resolution`.**
Gate `i` integrates the interval `[i*res, (i+1)*res)`. Using the centre rather
than the top edge is a half-gate (5 m at CL31 10 m resolution) systematic
choice, an order of magnitude below the detection tolerances.

**#V7 — CL-series profiles arrive already range- and background-corrected.**
The adapter sets `range_corrected=True` / `background_subtracted=True`, so the
preprocessor skips the R² multiplication (see #P1) and applies only a residual
offset removal. The `generic_csv` adapter defaults both to `False`, which is
what keeps the R² code path exercised by real data rather than only tests.

**#V8 — Detection-status digit is trusted for obscuration only.**
Digit `4` (full obscuration) sets the `FOG` flag and digit `/` (missing/suspect
raw data) sets `LOW_SNR`. Digits `1`-`3` (instrument cloud bases) are recorded
but *not* used for detection — BL View re-detects cloud from the profile so
that behaviour is identical across instruments.

**#V9 — Gate count comes from the parameter line, with the profile line as
fallback.** Header digits are never trusted for anything numeric; a truncated
or corrupt message is skipped and parsing continues with the next one.

**#V10 — Files with no timestamp lines get synthesised timestamps.**
Cadence is interpolated from whatever timestamps exist; if there are none at
all, the series is assumed regular at `default_interval_s` (30 s) ending at the
file's mtime, unless `start_time=` is supplied. This is a fallback for
hand-extracted files, not a supported ingest path.

---

## S — Synthetic data

No sample file was present in the repository, so the pipeline is built and
validated against generated data (`blview/synth/generator.py`). **The synthetic
file is not an observation and is labelled as such in its own header.**

**#S1 — The diurnal cycle is keyed to UTC hour-of-day.**
Sunrise ~05:00, sunset ~19:00 UTC, i.e. the site is treated as being at
longitude 0. Choosing a real site's local solar time would change nothing
structurally.

**#S2 — The injected atmosphere is a mid-latitude summer fair-weather day.**
Nocturnal stable layer ~260-315 m; convective growth from 06:00 to a 1700 m
maximum at 15:00; evening collapse 18:00-20:00; residual layer 1620 m
subsiding to 1360 m overnight and eroded by the growing mixed layer around
10:30; a decoupled haze layer at 2.4-2.1 km, 680 m deep, present 00:00-17:00;
a stratocumulus deck at 1900 m from 08:30-11:00; intermittent cumulus at the
mixing-layer top 13:30-16:30; radiation fog 03:00-04:30; precipitation
21:00-22:00. Both height functions are **continuous across midnight** — a
discontinuity there would be an artefact no detector could reproduce.

**#S3 — The synthetic overlap function deliberately differs from the one the
preprocessor assumes.** The generator uses full overlap at 180 m with exponent
1.6; the preprocessor assumes 200 m with exponent 2.0 (#P2). The real overlap
function is never known exactly, so validating against a perfectly matched one
would be self-fulfilling.

**#S4 — "Injected" is not the same as "observable", and the truth file records
both.** A ceilometer beam is extinguished by cloud, so nothing above a cloud
base can be retrieved; fog and precipitation profiles are screened out before
detection runs. The truth file therefore carries `sml_visible`, `rl_visible`,
`haze_visible`, `cloud_visible` and `screened` flags, and the validation
harness asserts recovery **only where the feature was observable**. Asserting
otherwise would be demanding that the tool invent data.

**#S5 — Instrument noise is constant in raw photon units.**
It therefore grows as R² in the already-range-corrected profile that the
instrument reports. Normalised to `noise_beta_at_3km = 8e-7 m^-1 sr^-1`, with
a x1.6 solar-background increase at midday and 2 % multiplicative speckle.
This makes the elevated haze layer roughly SNR 2 in a *single* profile — it is
only detectable after the temporal averaging in #P6, which is exactly the
regime a real CL31 operates in.

**#S6 — Lidar ratios for the attenuation integral: 30 sr aerosol, 18 sr cloud
droplets, 8π/3 molecular.** Standard literature values; they only affect how
fast the synthetic beam is extinguished.

**#S7 — Cloud saturates the 20-bit raw format.**
The generator clips to ±524287 counts exactly as the real format does, which
reproduces the real-world consequence that cloud tops are frequently not
determinable.

---

## P — Preprocessing

**#P1 — R² range correction is applied only when the adapter says it is
needed.** `ProfileSet.range_corrected` drives this. Applying R² twice is a
silent, catastrophic error that looks superficially plausible (it just tilts
the profile), so it is controlled by the format adapter rather than by a flag
the operator has to remember.

**#P2 — The near-range overlap function is unknown, so it is modelled and the
un-modellable part is masked. [tunable]**
The specification says to assume incomplete overlap below ~200 m. BL View
models overlap as `((R - 30) / 170)^2`, rising from zero at 30 m to full
overlap at 200 m. **Gates where the correction would need a gain above
`overlap_max_gain` (8x) are masked out entirely, not corrected** — the lowest
usable gate is therefore 90 m. Dividing a noisy signal by a small *guessed*
overlap value manufactures gradients that look exactly like a shallow
nocturnal layer, and there is no way to tell them from the real thing after
the fact. `DetectConfig.min_layer_height_m` (90 m) is set to match.

Consequence: **BL View cannot report a layer below 90 m.** A very shallow
nocturnal stable layer (which can be 50-100 m in strong radiative cooling)
will be missed, or its top reported at the 90 m floor. Supplying a real
measured overlap function for the instrument is the only fix.

**#P3 — Detector background is removed as a residual offset in
un-range-corrected space.** A constant detector offset `eps` appears in an
already-range-corrected profile as `eps * R^2` — a growing ramp, not a
constant. It is estimated as the median of `beta / R^2` above 6 km and
subtracted as `eps * R^2`. On CL-series data (already background-corrected by
the firmware) this is a near-no-op, which is the intended behaviour.

**#P4 — Nothing above 6 km is signal. [tunable]**
`noise_ref_bottom_m = 6000` defines the "noise only" reference window used for
both the background offset and the noise scale. At a site with routine
cirrus or elevated smoke above 6 km this window must be raised, or the noise
scale will be overestimated and real layers suppressed.

**#P5 — Noise model: `sigma(R)^2 = (sigma0 * R^2)^2 + (0.02 * beta)^2`. [tunable]**
The first term is detector noise, which is constant in raw photon units and
therefore grows as R² once range-corrected; `sigma0` is measured per profile
from the far-field scatter (robust MAD estimator, smoothed over 15 profiles).
The second is signal-proportional speckle. Without the speckle term the model
implies essentially zero noise near the ground, which would make every trivial
low-level wiggle statistically significant.

**#P6 — Detection runs on a 5-minute / 40 m average, not on single profiles.
[tunable]**
A single ceilometer profile has SNR ~2 at the height of a weak elevated haze
layer — far too noisy for gradient detection. `beta_smooth` is a boxcar
average in time and height, and `sigma` is propagated through it using the
Kish effective sample size, so partially-masked or edge windows get the right
noise, not an assumed one. The full-resolution `beta` field is kept separately
and is what the quicklook displays.

Consequence: **reported layer heights are 5-minute averages.** Genuinely
fast transitions (a convective plume, a passing front) are smeared over that
window.

**#P7 — Screened profiles are excluded from their neighbours' averages, and a
profile whose average has too few contributors is flagged `NO_DETECTION`.**
Otherwise a fog bank would corrupt the ten profiles either side of it.
Screening is dilated by one profile in each direction, deliberately erring
towards discarding a good profile rather than trusting a contaminated one.

**#P8 — Precipitation is tested before fog, and separated from it by depth.**
Precipitation is surface-connected backscatter above `1e-5 m^-1 sr^-1`
continuously filling 800 m or more (tolerating 150 m dropouts). Fog is a
strong *shallow* return (peak above `3e-5` below 300 m) that leaves the
profile above 600 m at the noise floor. A precipitating profile also looks
extinguished aloft, so testing fog first would misclassify it. Neither test
uses the instrument's own status digit, which not all instruments provide
reliably (#V8).

Consequence: **drizzle too light to reach `1e-5`, or virga that does not
reach the ground, will not be screened** and may be reported as an aerosol
layer. This is the single most likely source of a wrong layer in real data.
