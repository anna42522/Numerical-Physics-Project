# scripts/

One script per figure in the report. Each script imports from `src/`, runs from the repo root, and saves its output to `results/figures/`.

```bash
python scripts/fig01_cat_wigner.py
```

| Script | Figure |
|---|---|
| `fig01_cat_wigner.py` | Initial cat state and Wigner function evolution under photon loss (`mesolve`) |
| `fig02_*.py` *(planned)* | `mcsolve` vs `mesolve` validation and 1/√M convergence |
| `fig03_*.py` *(planned)* | Decoherence time vs cat separation (sweep in α) |
| `fig04_*.py` *(planned)* | Photon loss vs pure dephasing comparison |

Naming: `figNN_short_description.py`. Fix the random seed (`seeds=` in `mcsolve`) so figures are reproducible.
