"""Unit tests for plotting functions."""

import numpy as np
from vmc_hydrogen.plots import (
    plot_blocking,
    plot_optimization,
    plot_radial_distribution,
)


def test_plot_radial_distribution(tmp_path):
    """Verify radial distribution figure is generated."""
    out_file = tmp_path / "test_radial.png"
    samples = np.random.default_rng(42).exponential(scale=1.5, size=500)

    plot_radial_distribution(samples, alpha=1.0, output_path=out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_plot_blocking(tmp_path):
    """Verify blocking analysis figure is generated."""
    out_file = tmp_path / "test_blocking.png"
    sizes = np.array([10, 20, 50, 100])
    errors = np.array([0.02, 0.03, 0.035, 0.036])

    plot_blocking(sizes, errors, output_path=out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_plot_optimization(tmp_path):
    """Verify optimization trajectories figure is generated."""
    out_file = tmp_path / "test_opt.png"
    history = {
        "alpha": [0.5, 0.7, 0.9, 1.0],
        "energy": [-0.3, -0.42, -0.48, -0.5],
    }

    plot_optimization(history, output_path=out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
