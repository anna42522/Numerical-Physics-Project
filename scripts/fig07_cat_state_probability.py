"""Plot the probability of the cat state to be in a Fock state, the formula is taken from Sakurai page 175"""
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys
from math import factorial, cosh, sinh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
output_dir = pathlib.Path(__file__).resolve().parents[1] / "results" / "figures"

from src.config import load_config
config = load_config()
alpha = config["system"]["alpha"]
cat_parity = config["system"]["cat_parity"]
N_cut = config["system"]["N_cut"]
def plot_odd(n, probabilities, alpha, output_dir):
    fig, ax = plt.subplots()
    ax.bar(n, probabilities, width=0.8)
    ax.axvline(x=2*alpha**2, color="blue", linestyle=":", label=r"$2\alpha^2$")
    ax.set_title("Odd cat state")
    ax.set_xlabel("Fock number n")
    ax.set_ylabel("Probability")
    fig.savefig(output_dir / "fig07_cat_state_probability_odd.png")
    plt.close(fig)


def plot_even(n, probabilities, alpha, output_dir):
    fig, ax = plt.subplots()
    ax.bar(n, probabilities, width=0.8)
    ax.axvline(x=2*alpha**2, color="blue", linestyle=":", label=r"$2\alpha^2$")
    ax.set_title("Even cat state")
    ax.set_xlabel("Fock number n")
    ax.set_ylabel("Probability")
    fig.savefig(output_dir / "fig07_cat_state_probability_even.png")
    plt.close(fig)

if cat_parity == "even":
    n = np.arange(0, N_cut, 2)
    probabilities = [
    np.exp(-alpha**2) * alpha**(2 * k) / (factorial(k) * cosh(alpha**2))
    for k in n
    ]

    plot_even(n, probabilities, alpha, output_dir)

elif cat_parity == "odd":
    n = np.arange(1, N_cut, 2)
    probabilities = [
    np.exp(-alpha**2) * alpha**(2 * k) / (factorial(k) * sinh(alpha**2))
    for k in n
    ]
    plot_odd(n, probabilities, alpha, output_dir)