"""Unit tests for the Hydrogen1sWaveFunction class."""

import numpy as np
import pytest
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction


def test_invalid_alpha_raises_error():
    """Verify that non-positive alpha values raise a ValueError."""
    with pytest.raises(ValueError):
        Hydrogen1sWaveFunction(alpha=0.0)
    with pytest.raises(ValueError):
        Hydrogen1sWaveFunction(alpha=-1.5)


def test_wavefunction_evaluation():
    """Check evaluation of psi_T(r) = exp(-alpha * r) against known values."""
    alpha = 1.2
    wf = Hydrogen1sWaveFunction(alpha=alpha)
    r = np.array([0.0, 0.5, 1.0, 2.0])

    expected = np.exp(-alpha * r)
    np.testing.assert_allclose(wf.value(r), expected, atol=1e-12)


def test_radial_density():
    """Check calculation of radial density D(r) = r^2 * exp(-2 * alpha * r)."""
    alpha = 0.9
    wf = Hydrogen1sWaveFunction(alpha=alpha)
    r = np.array([0.0, 1.0, 2.5])

    expected = (r**2) * np.exp(-2.0 * alpha * r)
    np.testing.assert_allclose(wf.radial_density(r), expected, atol=1e-12)


def test_log_derivative():
    """Check analytical logarithmic derivative d(ln psi_T) / d(alpha) = -r."""
    wf = Hydrogen1sWaveFunction(alpha=1.0)
    r = np.array([0.1, 1.4, 3.2])

    np.testing.assert_allclose(wf.log_derivative(r), -r, atol=1e-12)
