"""Unit tests for the VariationalOptimizer class."""

import numpy as np
import pytest
from vmc_hydrogen.optimizer import VariationalOptimizer


def test_optimizer_invalid_parameters()-> None:
    """Verify that non-positive parameters raise a ValueError."""
    with pytest.raises(ValueError):
        VariationalOptimizer(initial_alpha=-0.1)
    with pytest.raises(ValueError):
        VariationalOptimizer(learning_rate=0.0)
    with pytest.raises(ValueError):
        VariationalOptimizer(num_walkers=0)
    with pytest.raises(ValueError):
        VariationalOptimizer(steps_per_iter=-10)


def test_gradient_oracle_at_analytical_minimum()-> None:
    """Verify energy and gradient evaluate to exact minimum at alpha = 1.0. """
    opt = VariationalOptimizer(initial_alpha=1.0, seed=42)

    rng = np.random.default_rng(42)
    synthetic_samples = rng.gamma(shape=3.0, scale=0.5, size=50000)

    mean_energy, gradient = opt.compute_gradient(synthetic_samples)

    assert mean_energy == pytest.approx(-0.5, abs=1e-10)
    
    assert gradient == pytest.approx(0.0, abs=1e-3)


def test_optimizer_convergence()-> None:
    """Verify optimization trajectory moves towards alpha = 1.0 and E = -0.5 Ha."""
    opt = VariationalOptimizer(
        initial_alpha=0.6,
        learning_rate=0.2,
        num_walkers=300,
        steps_per_iter=40,
        seed=123,
    )

    history = opt.optimize(max_iterations=15)

    final_alpha = history["alpha"][-1]
    final_energy = history["energy"][-1]

    assert final_alpha > 0.6
    assert abs(final_alpha - 1.0) < 0.20
    assert abs(final_energy - (-0.5)) < 0.15
