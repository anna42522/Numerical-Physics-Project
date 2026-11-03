"""Fig. 1: initial cat state and evolution under photon loss (mesolve)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from src.cat_states import cat_state, loss_op, check_truncation, N_CUT
from src.plotting import plot_wigner

alpha, kappa = 3.0, 1.0
psi0 = cat_state(alpha, "even")
assert check_truncation(psi0), "Increase N_CUT"

H = qt.qzero(N_CUT)                      # rotating frame, H0 = 0
times = np.array([0.0, 0.02, 0.05, 0.2])  # in units of 1/kappa
res = qt.mesolve(H, psi0, times, [loss_op(kappa)])

fig, axes = plt.subplots(1, len(times), figsize=(14, 3.6))
for ax, t, st in zip(axes, times, res.states):
    plot_wigner(st, ax=ax, title=f"κt = {t:.2f}")
fig.tight_layout()
out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/fig01_cat_wigner.png"
fig.savefig(out, dpi=150)
print("Saved:", out)

alphas=np.linspace(0,5,5)

fig, axes= plt.subplots(1, len(alphas), figsize=(14, 3.6))
for ax, alpha in zip(axes, alphas):
    cat= cat_state(alpha, "even")
    wig= qt.wigner(cat, np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    ax.contourf(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200), wig, 100)
    ax.set_title(f"α = {alpha:.1f}")
fig.tight_layout()

plt.title("Wigner function of the cat state")   
plt.savefig("results/figures/single_mode_state.png")