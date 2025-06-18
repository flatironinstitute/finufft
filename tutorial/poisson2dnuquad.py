# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NUFFT Poisson equation
#
# Demo solving Poisson eqn on nonuniform grid in [0,2pi)^2 periodic box.
# The NU grid has a quadrature rule associated with it, meaning that no
# inverse-NUFFT is needed. We use simple tensor-prod grid to illustrate.
#
# Herb 18/6/20 based on the Matlab demo by Barnett 2/14/2

# %%
import finufft
import numpy as np
np.seterr(divide='ignore')  # Disable division by zero warning
from matplotlib.colors import CenteredNorm
import matplotlib.pyplot as plt

# %% [markdown]
# source function in $[0,2\pi)^2$ (eps-periodic, zero-mean to be consistent)

# %%
w0 = 0.1 # width of bumps
src = lambda x, y: np.exp(-0.5*((x-1)**2+(y-2)**2)/w0**2)-np.exp(-0.5*((x-3)**2+(y-5)**2)/w0**2)

# %% [markdown]
# A) Solve $-\Delta u = f$, on regular grid via FFT, to warm up

# %%
for n in range(40, 120 + 20, 20):             # convergence study of grid points per side
  x = 2*np.pi*np.arange(n) / n                # grid
  xx, yy = np.meshgrid(x, x)                  # ordering: x fast, y slow
  f = src(xx,yy)                              # eval source on grid
  fhat = np.fft.ifft2(f)                      # step 1: Fourier coeffs by Euler-F projection
  k = np.fft.fftfreq(n) * n                   # Fourier mode grid
  kx, ky = np.meshgrid(k, k)
  kfilter = 1. / (kx**2 + ky**2)              # -(Laplacian)^{-1} in Fourier space
  kfilter[0, 0] = 0                           # kill the zero mode (even if inconsistent)
  kfilter[n//2, :] = 0
  kfilter[:, n//2] = 0                        # kill n/2 modes since non-symm
  u = np.fft.fft2(kfilter * fhat).real        # steps 2 and 3
  print(f"n={n}:\t\tu(0,0) = {u[0,0]:.15e}")  # check conv at a point

# %%
fig, ax = plt.subplots(1, 2, figsize=[10, 4], dpi=150)
imshow_args = {"origin": "lower", "cmap": "jet", "extent": [0.0, 2.*np.pi, 0.0, 2.*np.pi]}
cax = ax[0].imshow(f, **imshow_args)
fig.colorbar(cax, ax=ax[0])
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$y$")
ax[0].set_title("source term $f$")
cax = ax[1].imshow(u, **imshow_args)
fig.colorbar(cax, ax=ax[1])
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$y$")
ax[1].set_title("FFT solution $u$")
plt.savefig("pois_fft_python.png")
fig.tight_layout()
plt.show()

# %% [markdown]
# B) solve on general nonuniform (still quadrature) grid

# %%
deform = lambda t, s: np.stack([t + 0.5*np.sin(t) + 0.2*np.sin(2*s), s + 0.3*np.sin(2*s) + 0.3*np.sin(s-t)])
deformJ = lambda t, s: np.stack([
    np.stack([1 + 0.5*np.cos(t), 0.4*np.cos(2*s)], axis=-1),
    np.stack([-0.3*np.cos(s-t), 1+0.6*np.cos(2*s)+0.3*np.cos(s-t)], axis=-1)
], axis=-1) # its 2x2 Jacobian

# %%
t = 2 * np.pi * np.arange(n) / n           # 1d unif grid
tt, ss = np.meshgrid(t, t)
xxx = deform(tt, ss)
xx, yy = xxx[0], xxx[1]
f = src(xx, yy)
fig, ax = plt.subplots(1, 1, figsize=[4, 4], dpi=150)
cax = ax.pcolormesh(xx, yy, f, shading='gouraud', cmap="jet", norm=CenteredNorm())
ax.set_title("$f$ on mesh")
ax.axis("equal")
plt.savefig("pois_nugrid_python.png")
plt.show()

# %%
tol = 1e-12                               # NUFFT precision
for n in range(80, 240 + 40, 40):         # convergence study of grid points per side
    t = 2 * np.pi * np.arange(n) / n      # 1d unif grid
    tt, ss = np.meshgrid(t, t)
    xxx = deform(tt, ss)
    xx, yy = xxx[0], xxx[1]               # 2d NU pts
    J = deformJ(tt.T, ss.T)
    detJ = np.linalg.det(J).T
    ww = detJ / n**2                      # 2d quadr weights, including 1/(2pi)^2 in E-F integr
    f = src(xx, yy)
    Nk = 0.5 * n
    Nk = int(2 * np.ceil(Nk / 2))         # modes to trust due to quadr err
    fhat = finufft.nufft2d1(xx.ravel(), yy.ravel(), (f * ww).ravel().astype(np.complex128),
                            n_modes=(Nk, Nk), isign=1, eps=tol, modeord=1);  # do E-F
    k = np.fft.fftfreq(Nk) * Nk           # Fourier mode grid
    kx, ky = np.meshgrid(k, k)
    kfilter = 1. / (kx**2 + ky**2)        # -(Laplacian)^{-1} in Fourier space
    kfilter[0,0] = 0                      # kill the zero mode (even if inconsistent)
    kfilter[Nk//2,:] = 0
    kfilter[:,Nk//2] = 0                  # kill Nk/2 modes since non-symm
    u = finufft.nufft2d2(xx.ravel(), yy.ravel(), (kfilter * fhat),
                         isign=-1, eps=tol, modeord=1).real.reshape((n,n))  # eval filt F series @ NU
    print(f"n={n}:\t\tNk={Nk}\tu(0,0) = {u[0,0]:.15e}")   # check conv at a point

# %%
fig, ax = plt.subplots(1, 2, figsize=[10, 4], dpi=150)
pcolormesh_args = {"shading": "gouraud", "cmap": "jet"}
cax = ax[0].pcolormesh(xx, yy, f, norm=CenteredNorm(), **pcolormesh_args)
fig.colorbar(cax, ax=ax[0])
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$y$")
ax[0].set_title("source term $f$")
ax[0].axis("equal")
cax = ax[1].pcolormesh(xx, yy, u, norm=CenteredNorm(), **pcolormesh_args)
fig.colorbar(cax, ax=ax[1])
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$y$")
ax[1].set_title("NUFFT solution $u$")
ax[1].axis("equal")
fig.tight_layout()
plt.savefig("pois_nufft_python.png")
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=[10, 4], dpi=150)
cax = ax[0].imshow(np.log10(np.abs(fhat)), origin="lower", cmap="jet")
fig.colorbar(cax, ax=ax[0])
ax[0].set_title(r"FFT: $\mathrm{log}_{10}(|\hat{f}|)$")
ax[0].axis("equal")
cax = ax[1].imshow(np.log10(np.abs(fhat)), origin="lower", cmap="jet")
fig.colorbar(cax, ax=ax[1])
ax[1].set_title(r"NUFFT: $\mathrm{log}_{10}(|\hat{f}|)$")
ax[1].axis("equal")
fig.tight_layout()
plt.savefig("pois_fhat_python.png")
plt.show()

# %% [markdown]
# Note: if you really wanted to have an adaptive grid, using Fourier modes
# is a waste, since you need as many modes as nodes in a uniform FFT solver;
# you may as well use an FFT solver. For a fully adaptive fast Poisson solver
# use a `box-code`, i.e., periodic FMM applied to a quad-tree quadrature scheme.

# %%
