"""Unit test suite for statistical analysis and error estimation utilities."""

import numpy as np
import pytest
from vmc_hydrogen.analysis import blocking_analysis, estimate_energy


def test_estimate_energy_exact_oracle():
    """Verify zero-variance principle and asymptotic consistency.

    Ensures that for an identical constant sample (zero variance), the estimated 
    expectation value equals the constant and the statistical error is identically zero.
    """
    constant_energies = np.full(1000, -0.5)
    mean, err = estimate_energy(constant_energies)

    assert mean == pytest.approx(-0.5, abs=1e-12)
    assert err == pytest.approx(0.0, abs=1e-12)


def test_blocking_analysis_uncorrelated():
    """Verify standard error invariance under blocking for uncorrelated samples.

    In the limit of independent and identically distributed (i.i.d.) data, 
    the standard error evaluated via block averaging must remain invariant 
    and match the theoretical limit sigma / sqrt(N), irrespective of block length.
    """
    rng = np.random.default_rng(42)
    n_samples = 40000
    white_noise = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    sizes, errors = blocking_analysis(white_noise, min_block_size=10)
    expected_error = 1.0 / np.sqrt(n_samples)

    # Validate error plateau across early partition regimes with high block counts (Nb >= 100)
    assert np.allclose(errors[:5], expected_error, rtol=0.10)

    # Compare fixed single-block estimate against theoretical standard error
    _, err_estimate = estimate_energy(white_noise, block_size=20)
    assert err_estimate == pytest.approx(expected_error, rel=0.10)


def test_analysis_invalid_inputs():
    """Verify defensive input validation and exception handling.

    Ensures a ValueError is raised when handling empty datasets or when 
    the minimum block size exceeds the dataset capacity.
    """
    with pytest.raises(ValueError):
        blocking_analysis(np.array([]))
    with pytest.raises(ValueError):
        blocking_analysis(np.ones(10), min_block_size=8)
