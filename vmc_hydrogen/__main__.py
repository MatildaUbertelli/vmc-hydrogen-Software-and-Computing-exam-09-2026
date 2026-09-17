"""Command-line interface entry point for vmc_hydrogen package."""

import argparse
import sys
from pathlib import Path
from vmc_hydrogen.analysis import blocking_analysis, estimate_energy
from vmc_hydrogen.hamiltonian import local_energy
from vmc_hydrogen.optimizer import VariationalOptimizer
from vmc_hydrogen.plots import (
    plot_blocking,
    plot_optimization,
    plot_radial_distribution,
)
from vmc_hydrogen.sampler import MultiWalkerMetropolis
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for CLI execution.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(
        prog="vmc_hydrogen",
        description="Variational Monte Carlo solver for the Hydrogen atom 1s ground state.",
    )
    parser.add_argument(
        "--alpha-init",
        type=float,
        default=0.5,
        help="Initial variational parameter alpha (default: 0.5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.15,
        help="Learning rate for gradient descent (default: 0.15).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Maximum optimization iterations (default: 30).",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=500,
        help="Number of parallel Metropolis walkers (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random number generator seed (default: 42).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Directory where diagnostic plots will be stored (default: 'results').",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run full VMC optimization workflow and produce diagnostics.

    Parameters
    ----------
    argv : list of str or None, default=None
        Command line argument list. If None, sys.argv[1:] is used.

    Returns
    -------
    int
        Exit status code (0 for success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print(" VMC Hydrogen Ground State Optimization (Atomic Units)")
    print("==========================================================")
    print(f"Initial alpha : {args.alpha_init}")
    print(f"Walkers       : {args.walkers}")
    print(f"Iterations    : {args.iterations}")
    print(f"Learning rate : {args.lr}")
    print("----------------------------------------------------------")

    # Step 1: Variational optimization via stochastic gradient descent
    optimizer = VariationalOptimizer(
        initial_alpha=args.alpha_init,
        learning_rate=args.lr,
        num_walkers=args.walkers,
        steps_per_iter=50,
        seed=args.seed,
    )
    history = optimizer.optimize(max_iterations=args.iterations)

    opt_alpha = history["alpha"][-1]
    print(f"Optimal variational parameter alpha* = {opt_alpha:.4f}")

    # Plot optimization curves
    plot_optimization(history, output_path=outdir / "optimization_trajectory.png")

    # Step 2: High-statistics production run with optimal alpha
    print("\nStarting high-statistics production run...")
    best_wf = Hydrogen1sWaveFunction(alpha=opt_alpha)
    sampler = MultiWalkerMetropolis(
        wavefunction=best_wf,
        num_walkers=args.walkers,
        step_size=0.5 / opt_alpha,
        seed=args.seed + 1,
    )

    production_samples = sampler.sample(num_steps=200, thermalization=200)
    energies = local_energy(production_samples, opt_alpha)

    # Step 3: Statistical data blocking analysis
    sizes, errors = blocking_analysis(energies, min_block_size=10)
    plot_blocking(sizes, errors, output_path=outdir / "blocking_analysis.png")

    mean_energy, energy_err = estimate_energy(energies)

    # Plot final radial distribution
    plot_radial_distribution(
        production_samples,
        alpha=opt_alpha,
        output_path=outdir / "radial_distribution.png",
    )

    print("----------------------------------------------------------")
    print(f"Estimated Ground State Energy : {mean_energy:.6f} +/- {energy_err:.6f} Ha")
    print("Exact Analytical Energy       : -0.500000 Ha")
    print(f"Discrepancy                   : {abs(mean_energy - (-0.5)):.6f} Ha")
    print(f"Artifacts successfully saved in directory: '{outdir.resolve()}'")
    print("==========================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
