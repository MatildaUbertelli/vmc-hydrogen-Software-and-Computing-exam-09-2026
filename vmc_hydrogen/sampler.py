"""Vectorized multi-walker Metropolis-Hastings radial sampler."""

import numpy as np
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction


class MultiWalkerMetropolis:
    """Multi-walker Metropolis-Hastings sampler for 1D radial distributions.

    Parameters
    ----------
    wavefunction : Hydrogen1sWaveFunction
        Trial wave function instance providing the probability density.
    num_walkers : int, default=1000
        Number of independent random walkers evolved in parallel.
    step_size : float, default=0.5
        Standard deviation of Gaussian displacement proposals.
    seed : int or None, default=None
        Seed for the NumPy random number generator.

    Raises
    ------
    ValueError
        If num_walkers <= 0 or step_size <= 0.0.
    """

    def __init__(
        self,
        wavefunction: Hydrogen1sWaveFunction,
        num_walkers: int = 1000,
        step_size: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if num_walkers <= 0:
            raise ValueError("num_walkers must be strictly positive.")
        if step_size <= 0.0:
            raise ValueError("step_size must be strictly positive.")

        self.wavefunction = wavefunction
        self.num_walkers = int(num_walkers)
        self.step_size = float(step_size)
        self.rng = np.random.default_rng(seed)

        # Initialize walker radial positions in interval [0.5, 2.0]
        self.positions = self.rng.uniform(0.5, 2.0, size=self.num_walkers)

    def step(self) -> float:
        """Perform a single Metropolis step across all walkers.

        Returns
        -------
        float
            Acceptance ratio of the step across all walkers.
        """
        # Gaussian proposal for each walker
        proposals = self.positions + self.rng.normal(
            loc=0.0, scale=self.step_size, size=self.num_walkers
        )

        valid_mask = proposals > 0.0

        p_current = self.wavefunction.radial_density(self.positions)
        p_proposed = np.zeros_like(proposals)
        p_proposed[valid_mask] = self.wavefunction.radial_density(proposals[valid_mask])

        # Metropolis acceptance probability: min(1, P(r') / P(r))
        ratio = np.divide(
            p_proposed,
            p_current,
            out=np.zeros_like(p_proposed),
            where=p_current > 0.0,
        )
        acceptance_prob = np.clip(ratio, 0.0, 1.0)

        random_draws = self.rng.uniform(0.0, 1.0, size=self.num_walkers)
        accepted = random_draws < acceptance_prob

        self.positions[accepted] = proposals[accepted]
        return float(np.mean(accepted))

    def sample(self, num_steps: int, thermalization: int = 200) -> np.ndarray:
        """Collect radial samples after an initial thermalization phase.

        Parameters
        ----------
        num_steps : int
            Number of production steps to record.
        thermalization : int, default=200
            Number of initial steps to discard (burn-in).

        Returns
        -------
        np.ndarray
            Flattened array of radial positions with shape (num_steps * num_walkers,).
        """
        for _ in range(thermalization):
            self.step()

        history = np.empty((num_steps, self.num_walkers), dtype=float)
        for i in range(num_steps):
            self.step()
            history[i] = self.positions

        return history.flatten()
