"""Trial wave functions and radial probability densities for the hydrogen atom."""

import numpy as np


class Hydrogen1sWaveFunction:
  """Trial wave function for the hydrogen atom 1s ground state.

  Parameters
  ----------
  alpha : float, default=1.0
      Variational decay parameter. Must be strictly positive.

  Raises
  ------
  ValueError
      If alpha is less than or equal to zero.
  """

  def __init__(self, alpha: float = 1.0) -> None:
    if alpha <= 0.0:
      raise ValueError("Variational parameter alpha must be strictly positive.")
    self.alpha = float(alpha)

  def value(self, r: np.ndarray) -> np.ndarray:
    """Evaluate the trial wave function at given radial positions.

    Parameters
    ----------
    r : np.ndarray
        Radial distance from the nucleus in Bohr radii.

    Returns
    -------
    np.ndarray
        Values of the trial wave function exp(-alpha * r).
    """
    return np.exp(-self.alpha * r)

  def radial_density(self, r: np.ndarray) -> np.ndarray:
    """Calculate the unnormalized radial target probability density.

    Parameters
    ----------
    r : np.ndarray
        Radial distance array in Bohr radii.

    Returns
    -------
    np.ndarray
        Radial probability density values.
    """
    return (r**2) * np.exp(-2.0 * self.alpha * r)

  def log_derivative(self, r: np.ndarray) -> np.ndarray:
    """Evaluate the logarithmic derivative with respect to alpha.

    Parameters
    ----------
    r : np.ndarray
        Radial distance array in Bohr radii.

    Returns
    -------
    np.ndarray
        Values of the logarithmic derivative (-r).
    """
    return -r
