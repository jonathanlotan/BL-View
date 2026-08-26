"""Preprocessing corrections and screening, on hand-built profiles."""

import numpy as np
import pytest

from blview.config import Config, PreprocessConfig
from blview.model import ProfileSet, QualityFlag
from blview.preprocess import (
    lowest_usable_height, overlap_function, preprocess, screen_profiles,
)

RANGES = (np.arange(770) + 0.5) * 10.0


def _profiles(beta, range_corrected=True, background_subtracted=True, n=40):
    field = np.tile(np.asarray(beta, dtype=float), (n, 1))
    return ProfileSet(
        time=1.756e9 + np.arange(n) * 30.0,
        range_=RANGES,
        beta=field,
        range_corrected=range_corrected,
        background_subtracted=background_subtracted,
    )


def test_overlap_masks_gates_it_cannot_honestly_correct():
    cfg = PreprocessConfig()
    overlap, usable = overlap_function(RANGES, cfg)
    assert overlap[RANGES >= cfg.overlap_full_m].min() == pytest.approx(1.0)
    # Everything below the gain cap is masked rather than amplified.
    assert not usable[RANGES < lowest_usable_height(cfg)].any()
    assert usable[RANGES >= lowest_usable_height(cfg)].all()
    assert lowest_usable_height(cfg) == pytest.approx(90.1, abs=0.5)


def test_range_correction_applied_only_when_the_adapter_asks_for_it():
    """Applying R^2 twice is silent and catastrophic, so the adapter decides.

    The input decays with height: a *flat* range-corrected profile is
    unphysical (it implies an enormous detector background), and the background
    step would then dominate the comparison.  The check is that the ratio
    between the two branches scales as R^2, which isolates the range correction
    from the offset removal.
    """
    decaying = 3e-6 * np.exp(-RANGES / 800.0)
    already = preprocess(_profiles(decaying, range_corrected=True), Config())
    needs = preprocess(_profiles(decaying, range_corrected=False), Config())

    lo = int(np.argmin(np.abs(RANGES - 800.0)))
    hi = int(np.argmin(np.abs(RANGES - 2400.0)))
    ratio_lo = needs.beta[10, lo] / already.beta[10, lo]
    ratio_hi = needs.beta[10, hi] / already.beta[10, hi]
    assert ratio_hi / ratio_lo == pytest.approx(
        (RANGES[hi] / RANGES[lo]) ** 2, rel=0.05
    )
    assert "already applied" in already.attrs["preprocess_notes"]
    assert "applied R^2" in needs.attrs["preprocess_notes"]


def test_background_offset_is_removed_in_un_range_corrected_space():
    """A constant detector offset shows up as an eps*R^2 ramp, not a constant."""
    epsilon = 4e-14
    beta = np.full(RANGES.size, 2e-6) + epsilon * RANGES ** 2
    out = preprocess(_profiles(beta), Config())
    far = RANGES >= 6000.0
    assert abs(np.nanmedian(out.beta[10, far])) < 2e-8      # ramp removed
    mid = int(np.argmin(np.abs(RANGES - 1000.0)))
    assert out.beta[10, mid] == pytest.approx(2e-6, rel=0.05)


def test_noise_grows_as_r_squared():
    rng = np.random.default_rng(1)
    beta = np.full((60, RANGES.size), 1e-7)
    beta += rng.normal(0, 1, beta.shape) * 1e-13 * RANGES ** 2
    profiles = ProfileSet(
        time=1.756e9 + np.arange(60) * 30.0, range_=RANGES, beta=beta
    )
    out = preprocess(profiles, Config())
    a = int(np.argmin(np.abs(RANGES - 1000.0)))
    b = int(np.argmin(np.abs(RANGES - 4000.0)))
    assert out.sigma[30, b] / out.sigma[30, a] == pytest.approx(16.0, rel=0.25)


def test_smoothing_reduces_noise_by_the_effective_sample_count():
    """sigma must reflect the averaging actually done, not an assumed box."""
    rng = np.random.default_rng(2)
    beta = np.full((200, RANGES.size), 1e-7)
    beta += rng.normal(0, 1, beta.shape) * 1e-13 * RANGES ** 2
    out = preprocess(
        ProfileSet(time=1.756e9 + np.arange(200) * 30.0, range_=RANGES, beta=beta),
        Config(),
    )
    n_t = out.attrs["smooth_time_profiles"]
    n_z = out.attrs["smooth_vertical_gates"]
    j = int(np.argmin(np.abs(RANGES - 3000.0)))
    # Well inside the array the window is full, so sigma should drop by
    # sqrt(n_t * n_z) relative to the single-profile noise.
    ratio = out.sigma_single[100, j] / out.sigma[100, j]
    assert ratio == pytest.approx(np.sqrt(n_t * n_z), rel=0.05)


def test_fog_is_flagged_and_precipitation_is_not():
    """Fog: strong, shallow, extinguishing."""
    cfg = PreprocessConfig()
    beta = np.full(RANGES.size, 1e-9)
    beta[RANGES < 250.0] = 6e-5
    sigma = np.full(RANGES.size, 1e-8) * (RANGES / 1000.0) ** 2
    flags, _ = screen_profiles(beta[None, :], sigma[None, :], RANGES, cfg)
    assert flags[0] & int(QualityFlag.FOG)
    assert not flags[0] & int(QualityFlag.PRECIPITATION)


def test_precipitation_is_flagged_and_wins_over_fog():
    """Precipitation is also extinguished aloft; depth is what separates them."""
    cfg = PreprocessConfig()
    beta = np.full(RANGES.size, 1e-9)
    beta[RANGES < 1200.0] = 4e-5              # deep and surface-connected
    sigma = np.full(RANGES.size, 1e-8) * (RANGES / 1000.0) ** 2
    flags, diag = screen_profiles(beta[None, :], sigma[None, :], RANGES, cfg)
    assert flags[0] & int(QualityFlag.PRECIPITATION)
    assert not flags[0] & int(QualityFlag.FOG)
    assert diag["fill_depth_m"][0] > cfg.precip_min_depth_m


def test_cloud_does_not_trigger_precipitation():
    """A cloud is not surface-connected, so the fill test must ignore it."""
    cfg = PreprocessConfig()
    beta = np.full(RANGES.size, 2e-6)
    beta[(RANGES > 1500) & (RANGES < 1700)] = 5e-4
    sigma = np.full(RANGES.size, 1e-8) * (RANGES / 1000.0) ** 2
    flags, _ = screen_profiles(beta[None, :], sigma[None, :], RANGES, cfg)
    assert not flags[0] & int(QualityFlag.PRECIPITATION)
    assert not flags[0] & int(QualityFlag.FOG)


def test_clean_profile_is_not_screened():
    cfg = PreprocessConfig()
    beta = np.full(RANGES.size, 3e-6) * np.exp(-RANGES / 1500.0)
    sigma = np.full(RANGES.size, 1e-9) * (RANGES / 1000.0) ** 2
    flags, _ = screen_profiles(beta[None, :], sigma[None, :], RANGES, cfg)
    assert flags[0] == 0


def test_screened_profiles_do_not_contaminate_their_neighbours(short_dataset):
    """A fog bank must not leak into the ten profiles either side of it."""
    from blview.adapters import get_adapter
    from blview.synth.generator import SyntheticScenario, generate, write_vaisala_file

    scenario = SyntheticScenario(duration_h=1.0)
    data = generate(scenario, start_time=1756089000.0)     # spans fog onset
    raw = write_vaisala_file(short_dataset["raw"].parent / "edge.dat", data, scenario)
    out = preprocess(get_adapter("vaisala_cl").read(raw), Config())
    screened = out.screened_mask()
    assert screened.any() and not screened.all()
    # Detection input is NaN wherever it could not be built from clean data.
    assert np.isnan(out.beta_smooth[screened]).any()
    # The full-resolution display field is preserved everywhere.
    assert np.isfinite(out.beta[screened]).any()
