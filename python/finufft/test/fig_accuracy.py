# python/finufft/test/fig_accuracy.py
# Plots err = ||f_nufft - f_exact||_2 / ||f_exact||_2 vs tol.

import numpy as np
import matplotlib.pyplot as plt
import finufft

def finufft1d_accuracy_plot(M=10_000, N=100, isign=+1, upsampfac=2.00, seed=42, hold_inputs=False):
    rng = np.random.default_rng(seed)

    # Tolerances: 10.^(-1:-0.02:-15)
    exps = np.arange(-1.0, -15.0 - 1e-12, -0.02, dtype=np.float64)
    tols = np.power(10.0, exps, dtype=np.float64)
    errs = np.empty_like(tols)

    # Mode indices: k = ceil(-N/2) : floor((N-1)/2)
    kmin = int(np.ceil(-N / 2))
    kmax = int(np.floor((N - 1) / 2))
    ns = np.arange(kmin, kmax + 1, dtype=np.int64)  # length N

    # Optionally hold inputs fixed across tolerances
    if hold_inputs:
        x = rng.uniform(-np.pi, np.pi, size=M)                  # no dtype kw
        c = rng.normal(size=M) + 1j * rng.normal(size=M)
        x = np.asarray(x, dtype=np.float64)
        c = np.asarray(c, dtype=np.complex128)

    for t, tol in enumerate(tols):
        if not hold_inputs:
            x = rng.uniform(-np.pi, np.pi, size=M)
            c = rng.normal(size=M) + 1j * rng.normal(size=M)
            x = np.asarray(x, dtype=np.float64)
            c = np.asarray(c, dtype=np.complex128)

        # FINUFFT type-1. Pass (N,) not N. Pass opts via 'opts=...'.
        f_nufft = finufft.nufft1d1(x, c, N, eps=tol, upsampfac=upsampfac)

        # Exact result: fe[k] = sum_j c_j * exp(i*isign*k*x_j)
        fe = np.exp(1j * isign * (ns[:, None] * x[None, :])) @ c

        errs[t] = np.linalg.norm(f_nufft - fe) / np.linalg.norm(fe)

    # Plot
    plt.figure()
    plt.loglog(tols, errs, '+', label='measured')
    plt.plot(tols, tols, '-', label='y=x')
    plt.xlabel('tol')
    plt.ylabel('err')
    plt.title(rf'1d1: $\|\tilde f - f\|_2 / \|f\|_2$, M={M}, N={N}')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return tols, errs

if __name__ == "__main__":
    # Default: new x,c per tol; set hold_inputs=True to reuse inputs.
    finufft1d_accuracy_plot()
    #finufft1d_accuracy_plot(upsampfac=1.25)
