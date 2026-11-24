from pathlib import Path
import numpy as np
import yaml


def load_config():
    """Load the project configuration from config.yaml."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def get_time_grid(config=None):
    """Construct the simulation time array from t_start, t_final, and dt in config.yaml."""
    if config is None:
        config = load_config()
    sim = config["simulation"]
    t_start = float(sim["t_start"])
    t_final = float(sim["t_final"])
    dt = float(sim["dt"])
    n_steps = int(round((t_final - t_start) / dt)) + 1
    return np.linspace(t_start, t_final, n_steps)
