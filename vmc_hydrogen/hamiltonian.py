"""Local energy evaluation and Hamiltonian operators for the hydrogen atom."""

import numpy as np


def local_energy(r: np.ndarray, alpha: float) -> np.ndarray:
    """Calculate the local energy for the hydrogen atom trial wave function.

    Parameters
    ----------
    r : np.ndarray
        Radial distance array in Bohr radii. All values must be strictly positive.
    alpha : float
        Variational parameter of the trial wave function.

    Returns
    -------
    np.ndarray
        Local energy values computed at radial positions r.

    Raises
    ------
    ValueError
        If any element in r is less than or equal to zero.
    """
    r_arr = np.asarray(r, dtype=float)
    if np.any(r_arr <= 0.0):
        raise ValueError("Radial coordinate r must be strictly positive.")
    return -0.5 * (alpha**2) + (alpha - 1.0) / r_arr
