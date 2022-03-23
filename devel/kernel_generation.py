"""Utility code to generate polynomial approximation of the given kernel.
"""

import dataclasses
import functools
from math import degrees
from typing import Iterable

import numpy as np

def kernel(z, beta):
    return np.exp(beta * np.sqrt(1 - z**2))

def _make_grid(n):
    return 2 * (np.arange(n) + 0.5) / n - 1

def make_colocation_points(num_points: int) -> np.ndarray:
    grid = _make_grid(num_points)
    return np.concatenate((grid - 1j, 1 + grid * 1j, grid + 1j, -1 + grid * 1j))

def evaluate_function(fn, x: np.ndarray, width: int):
    grid = _make_grid(width)
    return fn(np.add.outer(x / width, grid))

def fit_polynomial(fn, degree: int, width: int, num_colocation_points: int=None) -> np.ndarray:
    """Fits a polynomial to the given function supported on [-1, 1] at equispaced points
    on [-1, 1] where the number of points is given by width.

    Note that the function must be able to be evaluated on the complex plane.

    Parameters
    ----------
    fn : callable
        Function to fit.
    degree : int
        Degree of the polynomial to fit.
    width : int
        Number of grid samples at which to evaluate the fit
    num_colocation_points : int, optional
        If not `None`, the number of points on each side of the box to evaluate
        for the fit. Otherwise, uses 16 times the degree.

    Returns
    -------
    np.ndarray
        An array of shape [degree + 1, width] containing the coefficients of the polynomial
        at each location.
    """
    if num_colocation_points is None:
        num_colocation_points = 16 * degree

    x = make_colocation_points(num_colocation_points)
    V = np.vander(x, degree + 1, increasing=True)
    R = evaluate_function(fn, x, width)
    return np.real(np.linalg.lstsq(V, R, rcond=None)[0])


@dataclasses.dataclass
class CodeGenConfig:
    floating_type: str = 'FT'
    width_name: str = 'w'
    input_name: str = 'z'
    output_name: str = 'ker'


def _generate_poly_eval(v, elements):
    if len(elements) == 1:
        return f'{elements[0]}'

    return f'{elements[0]} + {v} * ({_generate_poly_eval(v, elements[1:])})'

def generate_c_source_poly_eval_scalar(coefficients: np.ndarray, config: CodeGenConfig=CodeGenConfig(), indent: int=0):
    """Generates a C function body corresponding to the evaluation of the given polynomial
    """
    degree, width = coefficients.shape

    result = []
    for i in range(degree):
        elements = ', '.join(f'{coefficients[i, j]:+.16e}' for j in range(width))
        result.append(f'{config.floating_type} c{i}[] = {{ {elements} }};')

    result.append(f'for (int i = 0; i < sizeof(c0) / sizeof(c0[0]); i++) {{')
    result.append(f'  {config.output_name}[i] = ' + _generate_poly_eval(config.input_name, [f'c{i}[i]' for i in range(degree)]) + ';')
    result.append('}')

    result = [' ' * indent + line for line in result]
    return '\n'.join(result)


def write_scalar_source(path, degrees: Iterable[int], betas: Iterable[float], widths: Iterable[int]):
    """Writes scalar sources compatible with current kernel code.
    """
    with open(path, 'w') as f:
        codegen_config = CodeGenConfig(floating_type='FLT', width_name='w')

        for i, d, b, w in zip(range(len(degrees)), 20, betas, widths)[:d, :]:
            coeffs = fit_polynomial(functools.partial(kernel, beta=b), d, w)
            w_padded = ((w + 3) // 4) * 4

            coeffs = np.concatenate((coeffs, np.zeros((coeffs.shape[0], w_padded - w))), axis=-1)

            if i == 0:
                f.write('if (w == %d) {\n' % w)
            else:
                f.write('} else if (w == %d) {\n' % w)
            f.write(generate_c_source_poly_eval_scalar(coeffs, codegen_config))
            f.write('\n')
        f.write('} else {\nprintf("width not implemented!\\n");\n}\n')


def standard_configuration():
    widths = np.arange(2, 17)
    beta_over_width = np.array([2.20, 2.26, 2.38, 2.30])
    beta = beta_over_width[np.clip(widths - 2, a_min=None, a_max=len(beta_over_width) - 1)] * widths
    degrees = widths + 2 + (widths <= 8);  # between 2-3 more degree than w
    return degrees, beta, widths

def generic_configuration(upsample_factor=2):
    widths = np.arange(2, 17)
    gamma=0.97
    betaoverws = gamma*np.pi*(1-1/(2*upsample_factor))
    beta = betaoverws * widths
    degrees = widths + 1 + (widths<=8)
    return degrees, beta, widths

def write_matlab_code():
    """Function which mimcs the current functionality of the matlab script "gen_all_horner_C_code.m"
    """
    write_scalar_source('../src/ker_horner_allw_loop.c', *standard_configuration)
    write_scalar_source('../src/ker_lowupsampfrac_horner_allw_loop.c', *generic_configuration(1.25))


def write_scalar_kernels_structs(path, degrees: Iterable[int], betas: Iterable[float], widths: Iterable[int]):
    with open(path, 'w') as f:
        codegen_config = CodeGenConfig(width_name='width')

        all_kernel_names = []

        f.write('#pragma once\n\n')
        f.write('#include <tuple>\n\n')
        f.write('namespace finufft {\n')
        f.write('namespace detail {\n')

        for i, d, b, w in zip(range(len(degrees)), degrees, betas, widths):
            coeffs = fit_polynomial(functools.partial(kernel, beta=b), 20, w, 7)[:d, :]
            w_padded = ((w + 3) // 4) * 4

            coeffs = np.concatenate((coeffs, np.zeros((coeffs.shape[0], w_padded - w))), axis=-1)

            kernel_name = f'ker_horner_scalar_{i}'
            all_kernel_names.append(kernel_name)

            f.write(f'struct {kernel_name} {{\n')
            f.write(f'  constexpr static const int width = {w};\n')
            f.write(f'  constexpr static const int degree = {d};\n')
            f.write(f'  constexpr static const double beta = {b};\n')
            f.write(f'  constexpr static const int out_width = {coeffs.shape[-1]};\n')
            f.write(f'  template<typename FT> void operator()({codegen_config.floating_type} x, {codegen_config.floating_type}* {codegen_config.output_name}) const {{\n')
            f.write(f'    FT {codegen_config.input_name} = 2 * x + width - 1;\n')
            f.write(generate_c_source_poly_eval_scalar(coeffs, codegen_config, indent=4))
            f.write(f'  }};\n')
            f.write(f'}};\n')

        f.write('typedef std::tuple<')
        f.write(', '.join(all_kernel_names))
        f.write('> all_scalar_kernels_tuple;\n')

        f.write('} // namespace detail\n')
        f.write('} // namespace finufft\n')

def main():
    write_scalar_kernels_structs('../src/kernels/spread/spread_poly_scalar_impl.h', *standard_configuration())

if __name__ == '__main__':
    main()
