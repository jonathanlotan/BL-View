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

---

## D — Layer detection

**#D1 — Detection runs on a Haar wavelet covariance transform at five
dilations: 60, 120, 240, 480, 960 m. [tunable]**
Brooks (2003) uses a single dilation chosen for the layer of interest. BL View
must find both the sharp cap of a 250 m nocturnal stable layer and the broad,
weak entrainment gradient at the top of a 1700 m afternoon mixed layer, and no
single dilation does both. An edge seen at several dilations within a merge
tolerance is one physical edge: its height comes from the **finest** dilation
that saw it (best localisation) and the number of dilations that saw it becomes
a persistence score feeding the confidence.

**#D2 — Aerosol edges are found in log space; heights are localised in linear
space.** This is a deliberate departure from Brooks 2003, which works on the
range-corrected signal directly. In linear space the near-surface gradient is
so much larger than an elevated haze edge that no relative-strength cut can
keep both. Log space makes them comparable. But log space biases layer **tops
upward** — for a layer decaying onto a background, the steepest *fractional*
change is above the steepest *absolute* change — so once an edge is found, its
height is snapped to the nearest significant extremum of the linear transform
at the same dilation (`refine_edge_heights`). Measured against the synthetic
truth this removed a systematic bias of tens of metres.

**#D3 — Statistical significance is not physical significance. [tunable]**
Inside a mixing layer the SNR reaches several hundred, so a 1 % ripple passes
any `|W| > k·sigma` test while meaning nothing — and the detector then picks
that ripple as the mixing-layer top in preference to the real edge above it.
Three cuts are therefore applied together: `|W| > 3 sigma_W`, `|W|` at least
10 % of the strongest edge at that dilation, and an **absolute** floor
`|W| >= 0.03` in log10 units (about a 15 % change in backscatter). Without the
absolute floor, validation p95 error on the mixing-layer top was 925 m; with
it, 331 m.

**#D4 — Two known systematic errors are propagated as *bias*, not noise.**
Summing per-gate variances lets a correlated error average away, which is
wrong and dangerous — it makes an artefact look highly significant.

* *Overlap-model error* (#P2): the error profile
  `overlap_uncertainty * (1 - O(R))` is pushed through the same Haar transform
  as the data and added in quadrature to the transform's noise. Without it the
  leftover ramp from an imperfect overlap correction is reliably detected as a
  shallow layer at ~155 m.
* *Log-clamp ramp*: where backscatter falls below the noise floor it is clamped
  to `sigma(R)`, and since `sigma` grows as R² a fully-clamped stretch becomes
  a smooth upward ramp — the signature of a layer *base*. Clamped gates are
  therefore given a 0.5 dex uncertainty (a factor of ~3), which is all that is
  actually known about them.

**#D5 — Cloud is a magnitude decision, made first, at full time resolution.
[tunable]**
Threshold `1e-4 m^-1 sr^-1` over at least 2 consecutive gates, with the peak at
least 8x the median of the 300 m below it. Aerosol in a polluted mixing layer
peaks around `1e-5` and liquid cloud base runs `1e-4`-`1e-3`, so the threshold
sits in the gap. It is a *fixed backscatter threshold*, so it depends on the
profile-unit assumption (#V4) in a way the gradient-based detections do not.

Cloud detection runs on the **unsmoothed** field while aerosol detection runs
on the 5-minute average: a cloud return is 100x above noise and needs no
averaging, whereas averaging would dilute an intermittent cumulus below the
threshold and lose it entirely.

**#D6 — Cloud masks the aerosol analysis above it, and the mask is a running
minimum over the averaging window.** Excluding cloud from the *candidate list*
is not enough — a cloud sitting in the upper half of a Haar window swamps the
difference of the half-window means and erases the mixing-layer edge below it.
Everything from 50 m below the cloud base upward is set to NaN before the
transforms run, so contaminated windows return NaN and the same edge is still
found at the smaller dilations whose windows stay clear. Because a cloud
present in a few profiles is smeared across the whole temporal averaging
window, the ceiling used is the running minimum over that window, not the
profile's own cloud.

Consequence: **nothing is reported above a cloud base.** This is physics, not
policy — the beam is attenuated, and any "layer" found up there is an artefact
of the extinction.

**#D7 — Cloud tops are usually not determinable, and are reported as `null`.**
A top is only claimed when the signal above the cloud recovers to at least 2x
the noise floor over 300 m. For CL31-class instruments this almost never
happens with liquid cloud, which is the honest answer. The code path is
exercised by a unit test with a thin, transparent cloud.

**#D8 — Layer type is decided by structure, not by height. [tunable]**
Working upward: the lowest **surface-connected** top (no detected base beneath
it) is the mixing-layer top; an elevated layer with its own detected base and a
backscatter minimum below it is **decoupled** and therefore haze; an elevated
layer sitting directly on the one below with no such base is contiguous and
therefore the **residual layer**. Only one residual layer is reported per
profile — a second contiguous layer with no base of its own is the decaying
tail of the layer below, not a structure.

**#D9 — A residual layer must be at least 300 m deep; other elevated layers
200 m. [tunable]**
The entrainment zone immediately above a mixing-layer top is genuine elevated
aerosol and passes every contrast and SNR test, but it is the mixing layer's
own upper transition, not a residual layer. Depth is what separates them: an
entrainment zone scales with ~10 % of the mixing depth, a residual layer is
hundreds of metres to kilometres deep. This one cut removed roughly 400 false
residual layers per synthetic day.

**#D10 — The mixing layer's reported base is 0 m.**
It reaches the ground by definition. The lowest gate actually *measured* is
90 m (#P2) and is recorded in the layer's `meta`.

**#D11 — Confidence is a blend, not a probability.**
`0.5 x edge significance + 0.3 x scale persistence + 0.2 x contrast`, multiplied
by 0.6 for detections inside the overlap region. It is a **relative** quality
ranking for sorting and display. It is not calibrated and must not be read as
"70 % likely to be correct".

---

## T — Temporal tracking

**#T1 — Each layer type is tracked on the height that identifies it.**
Cloud and haze on their **base** (the sharp, well-determined edge, and for haze
the quantity of interest); mixing and residual layers on their **top** (their
base is the surface, or the layer below).

**#T2 — Association is optimal (Hungarian), not greedy, within a jump
tolerance of `250 m + 120 m per minute of gap`. [tunable]**
The allowance is deliberately generous relative to real layer motion (morning
mixing-layer growth reaches ~8 m/min) because detection scatter, not physical
motion, dominates the profile-to-profile difference.

**#T3 — A track shorter than 5 profiles is discarded as flicker; gaps shorter
than 15 minutes are interpolated and marked. [tunable]**
Interpolated points carry `interpolated=True` and half the confidence of the
measurements bracketing them, so a consumer can always distinguish a filled
point from a measured one. Gaps longer than that are left empty rather than
bridged.

**#T4 — At most one mixing layer and one residual layer per profile.**
Two tracks of the same type can be alive at once — during the morning
transition one follows the rising mixing-layer top while another still sits on
the residual-layer top — and each fills its own gaps, so a profile could end up
with a measured mixing layer *and* an interpolated one at a completely
different height. A measured detection wins over an interpolated one, then
confidence breaks ties. Cloud and haze are left alone: several of each in one
profile is real.

**#T5 — Heights are smoothed with a running median over 5 profiles.**
A median removes single-profile spikes without rounding off genuine
transitions the way a mean would.

---

## X — Validation

**#X1 — Scoring is restricted to *observable* timestamps (#S4).**
The counts of observable and skipped timestamps are printed, so the restriction
is visible rather than hidden.

**#X2 — Tolerances: cloud base 50 m, mixing/residual top and haze base 150 m,
haze top 200 m.** Cloud is essentially gate-resolution limited; the aerosol
tolerances are of order one entrainment-zone depth, which is the physical width
of the feature being located. The haze top is the weakest edge in the scene and
gets the loosest tolerance.

**#X3 — Required detection rates are set below what a perfect algorithm could
achieve, because parts of the scene are genuinely ambiguous.**
The residual-layer requirement is the lowest (75 %) because during the two
transitions the mixing layer and residual layer merge into a single
indistinguishable feature. These thresholds are a **regression gate**, not a
performance claim; the measured numbers are in the README.

**#X4 — A 15 % false-positive rate is tolerated per feature.**
Reporting a layer that truth says was absent is scored only on unscreened
profiles where truth is unambiguous.
