# Decoherence of Schrödinger Cat States via the Monte Carlo Wavefunction Method

Numerical Physics project — Anna Iranzo & Andriana Misailidi.

We simulate the decoherence of cat states |α⟩ ± |−α⟩ in a single bosonic
cavity mode under photon loss and pure dephasing, using `qutip.mcsolve`
(Monte Carlo Wavefunction, MCWF) and validating against `qutip.mesolve`
(exact Lindblad master equation).

## Installation

```bash
conda env create -f environment.yml
conda activate mcwf-cat
```

Alternatively: `pip install qutip matplotlib pytest`.

## Reproducing the figures

```bash
python scripts/fig01_cat_wigner.py      # initial cat state and evolution under photon loss
```

Each figure has its own script; outputs are saved to `results/figures/`.

## Repository structure

- `src/`: reusable code (states, validation, analysis, plotting)
- `scripts/`: one script per figure in the report
- `notebooks/`: exploration (not the "official" code)
- `tests/`: pytest tests
- `report/`, `slides/`: report and presentation
- `docs/`: proposal and working notes

## Project plan (5 weeks)

1. Environment, cat state, `mesolve`, first Wigner function
2. `mcsolve` + validation against `mesolve` and analytics; convergence in M
3. Sweep in α and fit of the decoherence time τ_D ∝ 1/d²
4. Pure dephasing comparison, Wigner animations
5. Report and presentation

## Workflow

`main` is always working. One branch per task (`feature/...`), merged via
pull request reviewed by the other person. Tasks tracked in Issues / Project board.

## References

1. J. Dalibard, Y. Castin, K. Mølmer, Phys. Rev. Lett. 68, 580 (1992).
2. M. B. Plenio, P. L. Knight, Rev. Mod. Phys. 70, 101 (1998).
3. M. Brune et al., Phys. Rev. Lett. 77, 4887 (1996).
4. H. J. Carmichael, *An Open Systems Approach to Quantum Optics*, Springer (1993).
