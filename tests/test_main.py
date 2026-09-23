"""Integration and regression tests for the VMC CLI execution entry point.

Validates end-to-end execution flow, artifact generation (diagnostic plots
and numerical summaries), and command-line argument validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from vmc_hydrogen.__main__ import build_parser, main, validate_args


@pytest.fixture(autouse=True)
def cleanup_figures() -> None:
    """Ensure all Matplotlib figures are closed after each integration test."""
    yield
    plt.close("all")


def test_main_cli_execution(tmp_path: Path) -> None:
    """Verify execution of the CLI main loop and integrity of generated artifacts."""
    out_dir = tmp_path / "cli_test_out"

    exit_code = main(
        [
            "--alpha-init", "0.8",
            "--iterations", "2",
            "--walkers", "50",
            "--steps-per-iter", "10",
            "--steps", "20",
            "--therm", "10",
            "--outdir", str(out_dir),
        ]
    )

    assert exit_code == 0
    assert out_dir.is_dir()

    # Verify visual diagnostic artifacts
    for plot_name in [
        "optimization_trajectory.png",
        "blocking_analysis.png",
        "radial_distribution.png",
    ]:
        plot_path = out_dir / plot_name
        assert plot_path.is_file(), f"Expected plot {plot_name} was not created."
        assert plot_path.stat().st_size > 0, f"Plot {plot_name} is empty."

    # Verify structured JSON output
    summary_path = out_dir / "simulation_summary.json"
    assert summary_path.is_file(), "Expected simulation summary JSON was not created."

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    expected_keys = {
        "optimal_alpha",
        "mean_energy_hartree",
        "energy_error_hartree",
        "exact_energy_hartree",
        "absolute_discrepancy_hartree",
        "num_walkers",
        "steps_per_iter",
        "production_steps",
        "thermalization_steps",
        "total_configurations",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["num_walkers"] == 50
    assert summary["steps_per_iter"] == 10
    assert summary["production_steps"] == 20
    assert summary["thermalization_steps"] == 10
    assert summary["exact_energy_hartree"] == -0.5
    assert summary["total_configurations"] == 50 * 20


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--alpha-init", "-0.5"),
        ("--lr", "0.0"),
        ("--iterations", "0"),
        ("--steps-per-iter", "0"),
        ("--walkers", "-10"),
        ("--steps", "0"),
        ("--therm", "-1"),
    ],
)
def test_main_cli_invalid_arguments(flag: str, value: str) -> None:
    """Ensure non-physical or out-of-bound CLI arguments trigger SystemExit via parser."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args([flag, value])
        validate_args(args, parser)
