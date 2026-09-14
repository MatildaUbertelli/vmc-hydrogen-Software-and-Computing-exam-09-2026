"""Unit tests for local energy and Hamiltonian evaluations."""

import numpy as np
import pytest
from vmc_hydrogen.hamiltonian import local_energy


def test_local_energy_exact_oracle():
    """Verify that E_L equals -0.5 Ha identically with zero variance when alpha=1.0."""
    r = np.linspace(0.1, 10.0, 100)
    e_local = local_energy(r, alpha=1.0)

    np.testing.assert_allclose(e_local, -0.5, atol=1e-12)
    assert np.var(e_local) == pytest.approx(0.0, abs=1e-12)


def test_local_energy_arbitrary_alpha():
    """Verify local energy computation for alpha != 1.0."""
    alpha = 0.8
    r = np.array([1.0, 2.0])
    expected = -0.5 * (alpha**2) + (alpha - 1.0) / r
    np.testing.assert_allclose(local_energy(r, alpha), expected)


def test_local_energy_invalid_r():
    """Verify that r <= 0 raises a ValueError."""
    with pytest.raises(ValueError):
        local_energy(np.array([1.0, 0.0]), alpha=1.0)
    with pytest.raises(ValueError):
        local_energy(np.array([-0.5, 2.0]), alpha=1.0)
