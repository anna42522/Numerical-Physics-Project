# src/

Reusable code shared by all scripts and notebooks. Nothing in here should run on import or produce figures.

| File | Contents |
|---|---|
| `cat_states.py` | Cat state construction (even/odd), photon-loss and dephasing collapse operators, truncation check |
| `plotting.py` | Plotting helpers, e.g. Wigner function heatmaps |
| `validation.py` *(planned)* | `mcsolve` vs `mesolve` comparison, analytical checks, convergence in M |
| `decoherence.py` *(planned)* | Sweep in α, extraction and fit of the decoherence time τ_D |

Convention: the Fock-space truncation `N_CUT` is defined once in `cat_states.py`.
