"""Unit tests for the CLI execution point."""

from vmc_hydrogen.__main__ import main


def test_main_cli_execution(tmp_path):
    """Verify execution of CLI main loop with small test parameters."""
    out_dir = tmp_path / "cli_test_out"

    exit_code = main(
        [
            "--alpha-init",
            "0.8",
            "--iterations",
            "2",
            "--walkers",
            "50",
            "--outdir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "optimization_trajectory.png").is_file()
    assert (out_dir / "blocking_analysis.png").is_file()
    assert (out_dir / "radial_distribution.png").is_file()
