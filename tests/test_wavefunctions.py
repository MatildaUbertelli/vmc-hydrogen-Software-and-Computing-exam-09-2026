import numpy as np
import pytest
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction


def test_initialization_invalid_alpha():
  with pytest.raises(ValueError):
    Hydrogen1sWaveFunction(alpha=-0.5)
  with pytest.raises(ValueError):
    Hydrogen1sWaveFunction(alpha=0.0)


def test_wavefunction_values():
  wf = Hydrogen1sWaveFunction(alpha=1.0)
  r = np.array([0.0, 1.0, 2.0])

  expected = np.exp(-r)
  np.testing.assert_allclose(wf.value(r), expected)


def test_radial_density():
  wf = Hydrogen1sWaveFunction(alpha=1.0)
  r = np.array([1.0, 2.0])

  expected = (r**2) * np.exp(-2.0 * r)
  np.testing.assert_allclose(wf.radial_density(r), expected)


def test_log_derivative():
  wf = Hydrogen1sWaveFunction(alpha=1.0)
  r = np.array([0.5, 1.5, 3.0])

  np.testing.assert_allclose(wf.log_derivative(r), -r)
