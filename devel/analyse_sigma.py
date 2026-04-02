#!/usr/bin/env python3
"""Plot sigma training data and fit polynomial.
Usage: python3 devel/analyse_sigma.py /tmp/sigma_d1d1.csv [output.png]
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.genfromtxt(
    sys.argv[1], delimiter=",", names=True, dtype=None, encoding="utf-8"
)
mask = (data["prec"] == "d") & (data["type"] == 1) & (data["dim"] == 1)
d = data[mask]
tol = d["tol"].astype(float)
sig_emp = d["sigma_empirical"].astype(float)
sig_an = d["sigma_analytical"].astype(float)
log_tol = np.log(tol)

# Transition region: empirical sigma strictly between MINSIGMA and MAXSIGMA
trans = (sig_emp > 1.26) & (sig_emp < 1.99)
tol_t = tol[trans]
sig_t = sig_emp[trans]
logt_t = log_tol[trans]

# Fit degree-1 and degree-3 in log(tol)
c1 = np.polyfit(logt_t, sig_t, 1)
c3 = np.polyfit(logt_t, sig_t, 3)
fit1 = np.polyval(c1, logt_t)
fit3 = np.polyval(c3, logt_t)
res1 = np.max(np.abs(fit1 - sig_t))
res3 = np.max(np.abs(fit3 - sig_t))

print(f"Transition tol range: [{tol_t.min():.3e}, {tol_t.max():.3e}]")
print(f"Degree-1 max residual: {res1:.5f}  coeffs (high->low): {c1}")
print(f"Degree-3 max residual: {res3:.5f}  coeffs (high->low): {c3}")
print(
    f"{'=> Use degree-1' if res1 < 0.02 else '=> Use degree-3 (degree-1 residual too large)'}"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.semilogx(tol, sig_emp, "b.-", ms=3, label="empirical (binary search)")
ax1.semilogx(tol, sig_an, "r--", label="analytical (lowest_sigma)")
ax1.set_xlabel("tol")
ax1.set_ylabel("sigma_min")
ax1.set_title("Full range — double, type=1, dim=1")
ax1.legend()

ax2.semilogx(tol_t, sig_t, "b.-", ms=4, label="empirical")
ax2.semilogx(tol_t, fit1, "g--", label=f"deg-1 (max err={res1:.4f})")
ax2.semilogx(tol_t, fit3, "m:", label=f"deg-3 (max err={res3:.4f})")
ax2.set_xlabel("tol")
ax2.set_ylabel("sigma_min")
ax2.set_title("Transition region fits")
ax2.legend()

plt.tight_layout()
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sigma_fit.png"
plt.savefig(out, dpi=120)
print(f"Saved plot to {out}")
