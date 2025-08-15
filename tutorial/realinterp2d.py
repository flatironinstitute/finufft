import numpy as np
import finufft as fi
import pytest


def nufft1d2(x, f, domain=(0, 2 * np.pi), rfft_axis=None, eps=1e-6):
    """Non-uniform fast transform of second type.

    Parameters
    ----------
    x : np.ndarray
        Real query points of coordinate in ``domain`` where interpolation is desired.
    f : np.ndarray
        Fourier coefficients e.g. ``f=fft(c,norm="forward")``.
    domain : tuple[float]
        Domain of coordinate specified by ``x`` over which samples were taken.
    rfft_axis : int
        Axis along which real FFT was performed.
        Default is to assume no real FFT was performed.
        If given assumes ``f`` has coefficients along that axis
        such that the real part of the function can be recovered
        from ∑ₙ fₙ exp(i n θ) for n > 0.
    eps : float
        Precision requested. Default is ``1e-6``.

    Returns
    -------
    cq : np.ndarray
        Complex function value at query points.

    """
    scale = 2 * np.pi / (domain[1] - domain[0])
    x = (x - domain[0]) * scale

    if rfft_axis is None:
        s = 1
    else:
        if rfft_axis != -1:
            raise NotImplementedError("rfft_axis must be -1.")
        s = f.shape[rfft_axis] // 2
        s = np.exp(1j * s * x)
        f = np.fft.ifftshift(f, axes=rfft_axis)

    return fi.nufft1d2(x, f, isign=1, eps=eps, modeord=1) * s


def nufft2d2(
    x, y, f, domain_x=(0, 2 * np.pi), domain_y=(0, 2 * np.pi), rfft_axis=None, eps=1e-6
):
    """Non-uniform fast transform of second type.

    Parameters
    ----------
    x : np.ndarray
        Real query points of coordinate in ``domain_x`` where interpolation is desired.
        The coordinates stored here must be the same coordinate enumerated across axis
        ``-2`` of ``f``.
    y : np.ndarray
        Real query points of coordinate in ``domain_y`` where interpolation is desired.
        The coordinates stored here must be the same coordinate enumerated across axis
        ``-2`` of ``f``.
    f : np.ndarray
        Fourier coefficients e.g. ``f=fft2(c,norm="forward")``.
    domain_x : tuple[float]
        Domain of coordinate specified by ``x`` over which samples were taken.
    domain_y : tuple[float]
        Domain of coordinate specified by ``y`` over which samples were taken.
    rfft_axis : int
        Axis along which real FFT was performed.
        Default is to assume no real FFT was performed.
        If given assumes ``f`` has coefficients along that axis
        such that the real part of the function can be recovered
        from ∑ₙ fₙ exp(i n θ) for n > 0.
    eps : float
        Precision requested. Default is ``1e-6``.

    Returns
    -------
    cq : np.ndarray
        Complex function value at query points.

    """
    scale_x = 2 * np.pi / (domain_x[1] - domain_x[0])
    scale_y = 2 * np.pi / (domain_y[1] - domain_y[0])
    x = (x - domain_x[0]) * scale_x
    y = (y - domain_y[0]) * scale_y

    if rfft_axis is None:
        s = 1
    else:
        if rfft_axis != -1 and rfft_axis != -2:
            raise NotImplementedError("rfft_axis must be -1 or -2.")
        s = f.shape[rfft_axis] // 2
        s = np.exp(1j * s * (y if rfft_axis == -1 else x))
        f = np.fft.ifftshift(f, axes=rfft_axis)

    return fi.nufft2d2(x, y, f, isign=1, eps=eps, modeord=1) * s


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


_test_inputs_1D = [
    (_f_1d, 2 * _f_1d_nyquist_freq() + 1, (0, 2 * np.pi)),
    (_f_1d, 2 * _f_1d_nyquist_freq(), (0, 2 * np.pi)),
    (_f_1d, 2 * _f_1d_nyquist_freq() + 1, (-np.pi, np.pi)),
    (_f_1d, 2 * _f_1d_nyquist_freq(), (-np.pi, np.pi)),
    (lambda x: np.cos(7 * x), 2, (-np.pi / 7, np.pi / 7)),
    (lambda x: np.sin(7 * x), 3, (-np.pi / 7, np.pi / 7)),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "func, n, domain, imag_undersampled",
    [
        (*_test_inputs_1D[0], False),
        (*_test_inputs_1D[1], True),
        (*_test_inputs_1D[2], False),
        (*_test_inputs_1D[3], True),
        (*_test_inputs_1D[4], True),
        (*_test_inputs_1D[5], False),
    ],
)
def test_non_uniform_FFT(func, n, domain, imag_undersampled):
    """Test non-uniform FFT interpolation."""
    x = np.linspace(domain[0], domain[1], n, endpoint=False)
    c = func(x)
    xq = np.array([7.34, 1.10134, 2.28])

    f = np.fft.fft(c, norm="forward")
    cq = nufft1d2(xq, f, domain)
    np.testing.assert_allclose(cq.real if imag_undersampled else cq, func(xq))


@pytest.mark.unit
@pytest.mark.parametrize("func, n, domain", _test_inputs_1D)
def test_non_uniform_real_FFT(func, n, domain):
    """Test non-uniform real FFT interpolation."""
    x = np.linspace(domain[0], domain[1], n, endpoint=False)
    c = func(x)
    xq = np.array([7.34, 1.10134, 2.28])

    f = 2 * np.fft.rfft(c, norm="forward")
    f[..., (0, -1) if (n % 2 == 0) else 0] /= 2
    np.testing.assert_allclose(
        nufft1d2(xq, f, domain, rfft_axis=-1).real,
        func(xq),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "func, m, n, domain_x, domain_y",
    [
        (
            _f_2d,
            2 * _f_2d_nyquist_freq()[0] + 1,
            2 * _f_2d_nyquist_freq()[1] + 1,
            (0, 2 * np.pi),
            (0, 2 * np.pi),
        ),
        (
            _f_2d,
            2 * _f_2d_nyquist_freq()[0] + 1,
            2 * _f_2d_nyquist_freq()[1] + 1,
            (-np.pi / 3, 5 * np.pi / 3),
            (np.pi, 3 * np.pi),
        ),
        (
            lambda x, y: np.cos(30 * x) + np.sin(y) ** 2 + 1,
            2 * 30 // 30 + 1,
            2 * 2 + 1,
            (0, 2 * np.pi / 30),
            (np.pi, 3 * np.pi),
        ),
    ],
)
def test_non_uniform_real_FFT_2D(func, m, n, domain_x, domain_y):
    """Test non-uniform real FFT 2D interpolation."""
    x = np.linspace(domain_x[0], domain_x[1], m, endpoint=False)
    y = np.linspace(domain_y[0], domain_y[1], n, endpoint=False)
    x, y = map(np.ravel, list(np.meshgrid(x, y, indexing="ij")))
    c = func(x, y).reshape(m, n)

    xq = np.array([7.34, 1.10134, 2.28, 1e3 * np.e])
    yq = np.array([1.1, 3.78432, 8.542, 0])

    f = 2 * np.fft.rfft2(c, norm="forward")
    f[..., (0, -1) if (n % 2 == 0) else 0] /= 2
    np.testing.assert_allclose(
        nufft2d2(xq, yq, f, domain_x, domain_y, rfft_axis=-1).real,
        func(xq, yq),
    )
