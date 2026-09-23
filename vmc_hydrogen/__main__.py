"""Command-line interface entry point for the vmc_hydrogen package.

Executes the full Variational Monte Carlo workflow: stochastic gradient
descent optimization of the trial wave function parameter, followed by
an equilibrium production sampling run and statistical data blocking.
"""

from __future__ import annotations

import argparse
import json
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
    """Construct and configure the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance with structured argument groups.
    """
    parser = argparse.ArgumentParser(
        prog="vmc_hydrogen",
        description="Variational Monte Carlo solver for the hydrogen atom 1s ground state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Variational Optimization arguments
    opt_group = parser.add_argument_group("Optimization Parameters")
    opt_group.add_argument(
        "--alpha-init",
        type=float,
        default=0.5,
        help="Initial variational parameter alpha (> 0).",
    )
    opt_group.add_argument(
        "--lr",
        type=float,
        default=0.15,
        help="Learning rate for stochastic gradient descent.",
    )
    opt_group.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Maximum number of gradient descent iterations.",
    )
    opt_group.add_argument(
        "--steps-per-iter",
        type=int,
        default=50,
        help="Number of MCMC sampling steps per walker during each optimization step.",
    )

    # Production sampling arguments
    mcmc_group = parser.add_argument_group("MCMC Sampling Parameters")
    mcmc_group.add_argument(
        "--walkers",
        type=int,
        default=500,
        help="Number of concurrent Metropolis-Hastings random walkers.",
    )
    mcmc_group.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Production sampling steps recorded per walker with optimal alpha.",
    )
    mcmc_group.add_argument(
        "--therm",
        type=int,
        default=200,
        help="Thermalization (burn-in) steps discarded prior to production sampling.",
    )

    # Output and Reproducibility arguments
    io_group = parser.add_argument_group("I/O and Reproducibility")
    io_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for pseudorandom number generation to ensure reproducibility.",
    )
    io_group.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Directory where diagnostic plots and numerical summaries are stored.",
    )

    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate numerical constraints on input arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    parser : argparse.ArgumentParser
        Parser instance used to signal configuration errors.
    """
    if args.alpha_init <= 0.0:
        parser.error("--alpha-init must be strictly positive.")
    if args.lr <= 0.0:
        parser.error("--lr must be strictly positive.")
    if args.iterations <= 0:
        parser.error("--iterations must be a positive integer.")
    if args.steps_per_iter <= 0:
        parser.error("--steps-per-iter must be a positive integer.")
    if args.walkers <= 0:
        parser.error("--walkers must be a positive integer.")
    if args.steps <= 0:
        parser.error("--steps must be a positive integer.")
    if args.therm < 0:
        parser.error("--therm must be a non-negative integer.")


def main(argv: list[str] | None = None) -> int:
    """Execute the complete VMC optimization and analysis pipeline.

    Parameters
    ----------
    argv : list of str or None, default=None
        Command-line argument vector. If None, sys.argv[1:] is used.

    Returns
    -------
    int
        Status code (0 for successful termination).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print(" VMC Hydrogen Ground State Optimization (Atomic Units)")
    print("==========================================================")
    print(f"Initial alpha         : {args.alpha_init:.4f}")
    print(f"Parallel Walkers      : {args.walkers}")
    print(f"Optimization Cycles   : {args.iterations}")
    print(f"Steps per iteration   : {args.steps_per_iter}")
    print(f"Learning Rate         : {args.lr:.4f}")
    print(f"Production Steps      : {args.steps}")
    print(f"Thermalization Steps  : {args.therm}")
    print(f"Random Seed           : {args.seed}")
    print("----------------------------------------------------------")

    # Step 1: Variational optimization via stochastic gradient descent
    optimizer = VariationalOptimizer(
        initial_alpha=args.alpha_init,
        learning_rate=args.lr,
        num_walkers=args.walkers,
        steps_per_iter=args.steps_per_iter,
        seed=args.seed,
    )
    history = optimizer.optimize(max_iterations=args.iterations)

    opt_alpha = history["alpha"][-1]
    print(f"Optimal variational parameter alpha* = {opt_alpha:.4f}")

    # Export optimization convergence trajectories
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

    production_samples = sampler.sample(num_steps=args.steps, thermalization=args.therm)
    energies = local_energy(production_samples, opt_alpha)

    # Step 3: Statistical data blocking analysis
    sizes, errors = blocking_analysis(energies, min_block_size=10)
    plot_blocking(sizes, errors, output_path=outdir / "blocking_analysis.png")

    mean_energy, energy_err = estimate_energy(energies)

    # Export final empirical vs theoretical radial distribution
    plot_radial_distribution(
        production_samples,
        alpha=opt_alpha,
        output_path=outdir / "radial_distribution.png",
    )

    # Step 4: Persist structured numerical artifacts
    summary_data = {
        "optimal_alpha": float(opt_alpha),
        "mean_energy_hartree": float(mean_energy),
        "energy_error_hartree": float(energy_err),
        "exact_energy_hartree": -0.5,
        "absolute_discrepancy_hartree": float(abs(mean_energy - (-0.5))),
        "num_walkers": args.walkers,
        "steps_per_iter": args.steps_per_iter,
        "production_steps": args.steps,
        "thermalization_steps": args.therm,
        "total_configurations": int(args.walkers * args.steps),
    }

    summary_file = outdir / "simulation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print("----------------------------------------------------------")
    print(f"Estimated Ground State Energy : {mean_energy:.6f} +/- {energy_err:.6f} Ha")
    print("Exact Analytical Energy       : -0.500000 Ha")
    print(f"Discrepancy                   : {abs(mean_energy - (-0.5)):.6f} Ha")
    print(f"Numerical summary written to  : '{summary_file.resolve()}'")
    print(f"Artifacts successfully saved in directory: '{outdir.resolve()}'")
    print("==========================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
