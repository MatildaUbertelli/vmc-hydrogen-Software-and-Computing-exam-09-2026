"""Unit tests for the MultiWalkerMetropolis sampler."""

import numpy as np
import pytest
from vmc_hydrogen.sampler import MultiWalkerMetropolis
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction


def test_sampler_invalid_parameters():
    """Verify that non-positive parameters raise ValueError."""
    wf = Hydrogen1sWaveFunction(alpha=1.0)
    with pytest.raises(ValueError):
        MultiWalkerMetropolis(wavefunction=wf, num_walkers=0)
    with pytest.raises(ValueError):
        MultiWalkerMetropolis(wavefunction=wf, step_size=-0.2)


def test_acceptance_rate_range():
  """Check that Metropolis acceptance rate stays within an expected physical window."""
  wf = Hydrogen1sWaveFunction(alpha=1.0)
  sampler = MultiWalkerMetropolis(
      wavefunction=wf, num_walkers=500, step_size=0.5, seed=42
  )

  acceptance_rates = [sampler.step() for _ in range(50)]
  mean_rate = np.mean(acceptance_rates)

  assert 0.30 <= mean_rate <= 0.85


def test_sample_output_shape_and_positivity():
    """Check output array shape and ensure radial distances remain strictly positive."""
    wf = Hydrogen1sWaveFunction(alpha=1.0)
    num_walkers = 50
    num_steps = 10
    sampler = MultiWalkerMetropolis(
        wavefunction=wf, num_walkers=num_walkers, step_size=0.5, seed=123
    )

    samples = sampler.sample(num_steps=num_steps, thermalization=10)

    # Flattened shape must match num_steps * num_walkers
    assert samples.shape == (num_steps * num_walkers,)
    # Radius must never be negative or zero
    assert np.all(samples > 0.0)
