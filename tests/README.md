# tests/

Automated tests, run with:

```bash
pytest
```

Current tests check that cat states are normalized and have the correct parity. Planned: average of MCWF trajectories agrees with `mesolve` within statistical error, and trace/hermiticity of ρ̄ are preserved.
