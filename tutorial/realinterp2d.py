# Written by Kaya Unalmis following discussion with Barnett on 08/15/25.
# [Note added by Barnett: some of the notation used here differs from
# the standard notation of FINUFFT docs. However it is a nice demo.]

import numpy as np
import finufft as fi
import pytest


def nufft1d2r(x, f, **kwargs):
    """Non-uniform real fast Fourier transform of second type.

    Parameters
    ----------
    x : np.ndarray
        Real query points of coordinate where interpolation is desired.
    f : np.ndarray
        Fourier coefficients fₙ of the map x ↦ c(x)
        such that c(x) = ∑ₙ fₙ exp(i n x) where n >= 0.

    Returns
    -------
    cq : np.ndarray
        Real function value at query points.

    """
    s = f.shape[-1] // 2
    s = np.exp(1j * s * x)
    return np.real(fi.nufft1d2(x, f, isign=1, modeord=0, **kwargs) * s)


def nufft2d2r(x, y, f, rfft_axis=-1, **kwargs):
    """Non-uniform real fast Fourier transform of second type.

    Parameters
    ----------
    x : np.ndarray
        Real query points of coordinate where interpolation is desired.
        The coordinates stored here must be the same coordinate enumerated across axis
        ``-2`` of ``f``.
    y : np.ndarray
        Real query points of coordinate where interpolation is desired.
        The coordinates stored here must be the same coordinate enumerated across axis
        ``-1`` of ``f``.
    f : np.ndarray
        Fourier coefficients fₘₙ of the map x,y ↦ c(x,y)
        such that c(x,y) = ∑ₘₙ fₘₙ exp(i m x) exp(i n y).
    rfft_axis : int
        Axis along which real FFT was performed.
        If -1 (-2), assumes c(x,y) = ∑ₘₙ fₘₙ exp(i m x) exp(i n y) where
            n ( m) >= 0, respectively.

    Returns
    -------
    cq : np.ndarray
        Real function value at query points.

    """
    if rfft_axis != -1 and rfft_axis != -2:
        raise NotImplementedError(f"rfft_axis must be -1 or -2, but got {rfft_axis}.")

    s = f.shape[rfft_axis] // 2
    s = np.exp(1j * s * (y if rfft_axis == -1 else x))
    f = np.fft.ifftshift(f, axes=rfft_axis)
    return np.real(fi.nufft2d2(x, y, f, isign=1, modeord=1, **kwargs) * s)


def _f_1d(x):
    """Test function for 1D FFT."""
    return np.cos(7 * x) + np.sin(x) - 33.2


def _f_1d_nyquist_freq():
    return 7


def _f_2d(x, y):
    """Test function for 2D FFT."""
    x_freq, y_freq = 3, 5
    return (
        # something that's not separable
        np.cos(x_freq * x) * np.sin(2 * x + y)
        + np.sin(y_freq * y) * np.cos(x + 3 * y)
        - 33.2
        + np.cos(x)
        + np.cos(y)
    )


def _f_2d_nyquist_freq():
    x_freq, y_freq = 3, 5
    x_freq_nyquist = x_freq + 2
    y_freq_nyquist = y_freq + 3
    return x_freq_nyquist, y_freq_nyquist


@pytest.mark.parametrize(
    "func, n",
    [
        (_f_1d, 2 * _f_1d_nyquist_freq() + 1),
        (_f_1d, 2 * _f_1d_nyquist_freq()),
    ],
)
def test_non_uniform_real_FFT(func, n):
    """Test non-uniform real FFT interpolation."""
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = func(x)
    xq = np.array([7.34, 1.10134, 2.28])

    f = 2 * np.fft.rfft(c, norm="forward")
    f[..., (0, -1) if (n % 2 == 0) else 0] /= 2
    np.testing.assert_allclose(nufft1d2r(xq, f), func(xq))


@pytest.mark.parametrize(
    "func, m, n, rfft_axis",
    [
        (_f_2d, 2 * _f_2d_nyquist_freq()[0] + 1, 2 * _f_2d_nyquist_freq()[1] + 1, -1),
        (_f_2d, 2 * _f_2d_nyquist_freq()[0] + 1, 2 * _f_2d_nyquist_freq()[1] + 1, -2),
    ],
)
def test_non_uniform_real_FFT_2D(func, m, n, rfft_axis):
    """Test non-uniform real FFT 2D interpolation."""
    x = np.linspace(0, 2 * np.pi, m, endpoint=False)
    y = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = map(np.ravel, tuple(np.meshgrid(x, y, indexing="ij")))
    c = func(x, y).reshape(m, n)

    xq = np.array([7.34, 1.10134, 2.28, 1e3 * np.e])
    yq = np.array([1.1, 3.78432, 8.542, 0])

    index = [slice(None)] * c.ndim
    index[rfft_axis] = (0, -1) if (c.shape[rfft_axis] % 2 == 0) else 0
    index = tuple(index)
    axes = (-2, -1)
    if rfft_axis == -2:
        axes = axes[::-1]

    f = 2 * np.fft.rfft2(c, axes=axes, norm="forward")
    f[index] /= 2
    np.testing.assert_allclose(nufft2d2r(xq, yq, f, rfft_axis), func(xq, yq))
