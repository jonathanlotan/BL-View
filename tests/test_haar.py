"""The Haar covariance transform must match its analytic behaviour exactly."""

import numpy as np
import pytest

from blview.detect.haar import (
    haar_covariance, haar_noise, half_width_gates, local_extrema, log_field,
)

DZ = 10.0
RANGES = (np.arange(400) + 0.5) * DZ


@pytest.mark.parametrize("scale", [60.0, 240.0, 960.0])
def test_step_response_locates_edge_and_has_expected_amplitude(scale):
    """A unit downward step gives W = +0.5 at the step height."""
    field = np.where(RANGES < 2000.0, 1.0, 0.0)[None, :]
    w = haar_covariance(field, DZ, scale)[0]
    peak = int(np.nanargmax(w))
    assert w[peak] == pytest.approx(0.5, abs=1e-9)
    # Gate centres sit at (i+0.5)*dz, so the best any transform can do is a
    # half-gate away from the true step.
    assert abs(RANGES[peak] - 2000.0) <= DZ


@pytest.mark.parametrize("scale", [60.0, 240.0, 960.0])
def test_sign_convention_top_positive_base_negative(scale):
    """Brooks' convention: W > 0 at a layer top, W < 0 at a layer base."""
    top = np.where(RANGES < 2000.0, 1.0, 0.0)[None, :]
    base = np.where(RANGES > 2000.0, 1.0, 0.0)[None, :]
    assert np.nanmax(haar_covariance(top, DZ, scale)) > 0
    assert np.nanmin(haar_covariance(base, DZ, scale)) < 0


@pytest.mark.parametrize("scale", [60.0, 240.0, 960.0])
def test_noise_prediction_matches_measured_noise(scale):
    """haar_noise must predict the actual scatter of the transform."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, size=(3000, RANGES.size))
    measured = float(np.nanstd(haar_covariance(noise, DZ, scale)))
    predicted = float(np.nanmedian(haar_noise(np.ones_like(noise), DZ, scale)))
    assert predicted == pytest.approx(measured, rel=0.03)
    # ...and match the closed form sigma / sqrt(2n).
    n = half_width_gates(scale, DZ)
    assert predicted == pytest.approx(1.0 / np.sqrt(2 * n), rel=1e-6)


def test_larger_dilation_suppresses_noise():
    """Noise falls as 1/sqrt(scale) -- why weak edges need the wide dilations."""
    sigma = np.ones((1, RANGES.size))
    small = float(np.nanmedian(haar_noise(sigma, DZ, 60.0)))
    large = float(np.nanmedian(haar_noise(sigma, DZ, 960.0)))
    assert large == pytest.approx(small / 4.0, rel=1e-6)


def test_masked_gates_produce_nan_not_a_biased_difference():
    """A partial window must be NaN: a half-filled mean would fake an edge."""
    field = np.ones((1, RANGES.size))
    field[0, 100:105] = np.nan
    w = haar_covariance(field, DZ, 240.0)[0]
    n = half_width_gates(240.0, DZ)
    # W[i] spans gates i-n .. i+n-1, so every centre whose window touches the
    # masked block is NaN, and the ones either side of that are not.
    assert np.all(np.isnan(w[101 - n: 105 + n]))
    assert np.isfinite(w[100 - n])
    assert np.isfinite(w[105 + n])
    assert np.isfinite(w[300])


def test_window_running_off_the_profile_is_nan():
    """Valid centres are n <= i <= n_range - n inclusive (see index convention)."""
    field = np.ones((1, RANGES.size))
    w = haar_covariance(field, DZ, 240.0)[0]
    n = half_width_gates(240.0, DZ)
    assert np.all(np.isnan(w[:n]))
    assert np.isfinite(w[n])
    assert np.isfinite(w[RANGES.size - n])
    assert np.all(np.isnan(w[RANGES.size - n + 1:]))


def test_log_field_clamps_at_the_noise_floor_and_propagates_error():
    """Background-subtracted backscatter is legitimately negative in clean air."""
    beta = np.array([[1e-6, -3e-8, 5e-7]])
    sigma = np.array([[1e-8, 1e-7, 1e-8]])
    value, error = log_field(beta, sigma, floor_snr=1.0)
    assert np.all(np.isfinite(value))
    assert value[0, 1] == pytest.approx(np.log10(1e-7))       # clamped
    assert value[0, 0] == pytest.approx(np.log10(1e-6))       # untouched
    # d(log10 x)/dx = 1/(x ln10)
    assert error[0, 0] == pytest.approx(1e-8 / (1e-6 * np.log(10)))


def test_local_extrema_ignores_nan_neighbours():
    values = np.array([[0.0, 1.0, 0.0, np.nan, 5.0, np.nan, 0.0]])
    assert local_extrema(values, +1)[0].tolist() == [False, True, False, False, False, False, False]
