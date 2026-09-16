"""Visualization utilities for VMC radial distributions, blocking, and optimization."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_radial_distribution(
    samples: np.ndarray,
    alpha: float,
    output_path: str | Path = "radial_distribution.png",
) -> None:
    """Plot histogram of sampled radial distances against exact theoretical density.

    Parameters
    ----------
    samples : np.ndarray
        Array of sampled radial distances.
    alpha : float
        Variational parameter of the trial wave function.
    output_path : str or Path, default="radial_distribution.png"
        Target file path for saving the figure.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    # Histogram of Monte Carlo walker positions
    ax.hist(
        samples,
        bins=80,
        range=(0.0, 6.0),
        density=True,
        alpha=0.6,
        color="royalblue",
        edgecolor="black",
        linewidth=0.5,
        label="VMC Metropolis Samples",
    )

    # Exact analytical radial density: 4 * alpha^3 * r^2 * exp(-2 * alpha * r)
    r_grid = np.linspace(0.0, 6.0, 300)
    exact_p = 4.0 * (alpha**3) * (r_grid**2) * np.exp(-2.0 * alpha * r_grid)
    ax.plot(
        r_grid,
        exact_p,
        color="crimson",
        linewidth=2.0,
        label=f"Analytical Density ($\\alpha = {alpha:.2f}$)",
    )

    ax.set_xlabel("Radial Distance $r$ [$a_0$]")
    ax.set_ylabel("Radial Probability Density $P(r)$")
    ax.set_title("Ground State Radial Electron Distribution")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_blocking(
    block_sizes: np.ndarray,
    block_errors: np.ndarray,
    output_path: str | Path = "blocking_analysis.png",
) -> None:
    """Plot standard error on energy expectation vs block size.

    Parameters
    ----------
    block_sizes : np.ndarray
        Array of evaluated block lengths.
    block_errors : np.ndarray
        Estimated standard error for each block size.
    output_path : str or Path, default="blocking_analysis.png"
        Target file path for saving the figure.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)

    ax.plot(block_sizes, block_errors, "o-", color="navy", markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Block Size")
    ax.set_ylabel(r"Standard Error $\sigma_{\bar{E}}$ [Ha]")
    ax.set_title("Data Blocking Analysis (Plateau Check)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_optimization(
    history: dict[str, list[float]],
    output_path: str | Path = "optimization_trajectory.png",
) -> None:
    """Plot parameter alpha and energy trajectory across optimization steps.

    Parameters
    ----------
    history : dict of str to list of float
        Dictionary containing 'alpha' and 'energy' series.
    output_path : str or Path, default="optimization_trajectory.png"
        Target file path for saving the figure.
    """
    iterations = range(len(history["alpha"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True, dpi=150)

    # Alpha trajectory
    ax1.plot(iterations, history["alpha"], "o-", color="darkgreen", ms=4, lw=1.5)
    ax1.axhline(1.0, color="grey", linestyle=":", label=r"Exact $\alpha = 1.0$")
    ax1.set_ylabel(r"Variational Parameter $\alpha$")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Energy trajectory
    ax2.plot(iterations, history["energy"], "s-", color="purple", ms=4, lw=1.5)
    ax2.axhline(-0.5, color="grey", linestyle=":", label="Exact $E = -0.5$ Ha")
    ax2.set_xlabel("Iteration Step")
    ax2.set_ylabel(r"Mean Local Energy $\langle E_L \rangle$ [Ha]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
