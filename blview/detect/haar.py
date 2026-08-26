"""Haar wavelet covariance transform (Brooks, 2003).

Brooks, I. M. (2003), *Finding boundary layer top: application of a wavelet
covariance transform to lidar backscatter profiles*, J. Atmos. Oceanic
Technol., 20, 1092-1105.

The transform is

.. math::
    W_f(a, b) = \\frac{1}{a} \\int f(z)\\, h\\!\\left(\\frac{z - b}{a}\\right) dz

with the Haar step

.. math::
    h(x) = \\begin{cases} +1 & -1/2 \\le x < 0 \\\\
                          -1 & 0 < x \\le 1/2 \\\\
                           0 & \\text{otherwise} \\end{cases}

so :math:`W > 0` where the signal *decreases* with height (the **top** of an
aerosol layer) and :math:`W < 0` where it *increases* (the **base** of an
elevated layer).  The dilation ``a`` sets the vertical scale of the gradient
the transform is most sensitive to, which is why BL View runs it at several
dilations: a 60 m dilation localises the sharp cap of a nocturnal stable
layer, while a 960 m dilation is what finds the broad, weak entrainment-zone
gradient at the top of a deep afternoon mixed layer.

Discretely, with gate spacing ``dz`` and ``n = a / (2 dz)`` gates per half
window, the transform reduces to half the difference of the two half-window
means::

    W(a, z_i) = (mean of gates i-n .. i-1  -  mean of gates i .. i+n-1) / 2

which makes both the transform and its noise a cumulative-sum operation over
the whole time-height array at once.

Extension to log space
----------------------
Brooks applies the transform to the range-corrected signal directly.  BL View
also runs it on ``log10(beta)`` (:func:`log_field`), because in linear space
the strong near-surface gradient dominates a profile so completely that a weak
elevated haze edge -- an order of magnitude smaller in absolute terms but just
as sharp in relative terms -- fails any sensible relative-strength cut.  Both
transforms are computed; aerosol edges come from the log transform and
magnitude/shape decisions from the linear one.  This is a deliberate departure
from the paper and is recorded in ASSUMPTIONS.md #D2.
"""

from __future__ import annotations

import numpy as np


def half_width_gates(scale_m: float, dz: float) -> int:
    """Gates per half window for a dilation of ``scale_m`` metres."""
    return max(int(round(scale_m / (2.0 * dz))), 1)


def _masked_cumsum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative sums of a NaN-masked field along the range axis.

    Returns ``(cumulative_value, cumulative_count)``, each with a leading zero
    column so that ``cs[:, j] - cs[:, i]`` is the sum over gates ``i..j-1``.
    """
    valid = np.isfinite(field)
    filled = np.where(valid, field, 0.0)
    zeros = np.zeros((field.shape[0], 1), dtype="float64")
    cs = np.concatenate([zeros, np.cumsum(filled, axis=1)], axis=1)
    cv = np.concatenate([zeros, np.cumsum(valid, axis=1, dtype="float64")], axis=1)
    return cs, cv


def haar_covariance(field: np.ndarray, dz: float, scale_m: float) -> np.ndarray:
    """Haar covariance transform of a ``(n_time, n_range)`` field.

    Returns an array of the same shape.  Gates whose window would run off
    either end of the profile, or would include a masked gate, are NaN --
    a partially-filled window would bias the difference of the half-window
    means and there is no honest way to correct for that.

    **Index convention.** The transform is about a *boundary between gates*,
    not a gate.  ``W[i]`` describes the boundary immediately below gate ``i``:
    its lower half-window is gates ``i-n .. i-1`` and its upper half-window is
    gates ``i .. i+n-1``.  Valid centres are therefore ``n <= i <= n_range-n``
    inclusive, which leaves ``n`` NaNs at the bottom of the profile and
    ``n-1`` at the top.
    """
    n_time, n_range = field.shape
    n = half_width_gates(scale_m, dz)
    out = np.full((n_time, n_range), np.nan)
    if 2 * n > n_range:
        return out

    cs, cv = _masked_cumsum(field)
    i = np.arange(n, n_range - n + 1)
    lower = cs[:, i] - cs[:, i - n]
    upper = cs[:, i + n] - cs[:, i]
    n_lower = cv[:, i] - cv[:, i - n]
    n_upper = cv[:, i + n] - cv[:, i]

    complete = (n_lower == n) & (n_upper == n)
    with np.errstate(invalid="ignore"):
        w = (lower - upper) / (2.0 * n)
    out[:, i] = np.where(complete, w, np.nan)
    return out


def haar_noise(sigma: np.ndarray, dz: float, scale_m: float) -> np.ndarray:
    """Standard deviation of :func:`haar_covariance` given per-gate ``sigma``.

    For independent per-gate noise,
    ``Var(W) = sum(sigma_k^2 over the 2n window) / (2n)^2``, so a wider
    dilation suppresses noise as ``1/sqrt(a)`` -- which is exactly why weak
    elevated edges only become significant at the larger scales.
    """
    n_time, n_range = sigma.shape
    n = half_width_gates(scale_m, dz)
    out = np.full((n_time, n_range), np.nan)
    if 2 * n > n_range:
        return out

    var = np.where(np.isfinite(sigma), sigma, np.nan) ** 2
    cs, cv = _masked_cumsum(var)
    i = np.arange(n, n_range - n + 1)
    total = cs[:, i + n] - cs[:, i - n]
    count = cv[:, i + n] - cv[:, i - n]
    complete = count == 2 * n
    out[:, i] = np.where(complete, np.sqrt(total) / (2.0 * n), np.nan)
    return out


def log_field(
    beta: np.ndarray,
    sigma: np.ndarray,
    floor_snr: float = 1.0,
    clamped_error: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert backscatter and its noise into log10 space.

    Backscatter that has been background-subtracted is legitimately negative
    where there is no signal, and ``log`` of that is undefined.  Values are
    therefore clamped at ``floor_snr * sigma``: below the noise floor the log
    value carries no information anyway, and clamping there (rather than at an
    arbitrary constant) keeps the floor range-dependent, which is what the
    R^2-growing noise requires.

    **The clamp itself has to be declared uncertain.**  Because the noise floor
    grows as R^2, a stretch of profile that is entirely below it becomes a
    smooth upward ramp in log space -- and an upward ramp is exactly the
    signature of a layer *base*.  Left alone, that manufactures a decoupled
    aerosol layer out of clean air whenever the free troposphere is quieter
    than the detector.  A clamped gate is only known to satisfy
    ``beta <= floor_snr * sigma``, so it is given a large uncertainty
    (``clamped_error`` dex; 0.5 is a factor of ~3) rather than the tiny
    linearised error the formula would otherwise produce.  Any Haar window
    lying inside the clamped region then has a noise level far above its own
    amplitude and cannot yield a significant edge, while a window straddling a
    real layer edge still does.

    Returns ``(log10_beta, sigma_of_log10_beta)``.
    """
    floor = np.maximum(floor_snr * np.abs(sigma), 1e-30)
    filled = np.nan_to_num(beta, nan=-np.inf)
    is_clamped = filled < floor
    clamped = np.maximum(filled, floor)
    clamped = np.where(np.isfinite(beta), clamped, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.log10(clamped)
        error = sigma / (clamped * np.log(10.0))
    error = np.where(is_clamped, clamped_error, error)
    error = np.where(np.isfinite(beta), error, np.nan)
    return value, error


def multiscale(
    field: np.ndarray, sigma: np.ndarray, dz: float, scales_m: tuple[float, ...]
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Run the transform at every dilation.

    Returns ``{scale_m: (W, sigma_W)}``.
    """
    return {
        float(scale): (
            haar_covariance(field, dz, scale),
            haar_noise(sigma, dz, scale),
        )
        for scale in scales_m
    }


def local_extrema(values: np.ndarray, sign: int) -> np.ndarray:
    """Boolean mask of strict local maxima (``sign=+1``) or minima (``-1``).

    Operates along the last axis.  NaN neighbours never form an extremum, so
    the edges of masked regions cannot masquerade as detections.
    """
    v = values * sign
    out = np.zeros(v.shape, dtype=bool)
    left = v[..., 1:-1] > v[..., :-2]
    right = v[..., 1:-1] >= v[..., 2:]
    finite = (
        np.isfinite(v[..., 1:-1]) & np.isfinite(v[..., :-2]) & np.isfinite(v[..., 2:])
    )
    out[..., 1:-1] = left & right & finite
    return out
