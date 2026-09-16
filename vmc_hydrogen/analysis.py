"""Statistical analysis and error estimation for Monte Carlo trajectories."""

import numpy as np


def blocking_analysis(data: np.ndarray, min_block_size: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Compute the standard error of the mean as a function of block size.

    Parameters
    ----------
    data : np.ndarray
        One-dimensional array of correlated measurements (e.g., local energies).
    min_block_size : int, default=10
        Minimum block size to evaluate.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - block_sizes: 1D array of evaluated block lengths.
        - block_errors: 1D array of estimated standard errors for each block size.

    Raises
    ------
    ValueError
        If input array is empty or min_block_size is too large.
    """
    arr = np.asarray(data, dtype=float)
    n_samples = len(arr)

    if n_samples == 0:
        raise ValueError("Input data array cannot be empty.")
    if min_block_size >= n_samples // 2:
        raise ValueError("min_block_size is too large compared to dataset length.")

    max_block_size = n_samples // 4
    block_sizes = np.unique(
        np.geomspace(min_block_size, max_block_size, num=25, dtype=int)
    )

    block_errors = []
    for b_size in block_sizes:
        n_blocks = n_samples // b_size
        truncated = arr[: n_blocks * b_size]
        block_means = truncated.reshape((n_blocks, b_size)).mean(axis=1)

        # Standard error of the block means: std / sqrt(N_blocks - 1)
        err = float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))
        block_errors.append(err)

    return block_sizes, np.array(block_errors)


def estimate_energy(
    local_energies: np.ndarray, block_size: int | None = None
) -> tuple[float, float]:
    """Estimate mean energy and its statistical uncertainty using blocking.

    Parameters
    ----------
    local_energies : np.ndarray
        Array of sampled local energy values.
    block_size : int or None, default=None
        Size of blocks to use. If None, automatically sets block size
        to N / 100 (minimum 20 samples per block).

    Returns
    -------
    tuple of (float, float)
        - mean_energy: Sample mean of the energy.
        - error: Estimated standard error on the mean taking correlations into account.
    """
    arr = np.asarray(local_energies, dtype=float)
    mean_val = float(np.mean(arr))

    if block_size is None:
        block_size = max(20, len(arr) // 100)

    n_blocks = len(arr) // block_size
    if n_blocks < 2:
        # Fallback to standard error if not enough samples for blocking
        return mean_val, float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

    truncated = arr[: n_blocks * block_size]
    block_means = truncated.reshape((n_blocks, block_size)).mean(axis=1)
    error = float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))

    return mean_val, error
