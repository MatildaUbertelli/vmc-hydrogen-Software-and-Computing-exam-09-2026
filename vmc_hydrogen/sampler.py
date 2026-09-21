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
    seed : int, np.random.Generator or None, default=None
        Seed or Generator instance for NumPy RNG.
    """

    def __init__(
        self,
        wavefunction: Hydrogen1sWaveFunction,
        num_walkers: int = 1000,
        step_size: float = 0.5,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if num_walkers <= 0:
            raise ValueError("num_walkers must be strictly positive.")
        if step_size <= 0.0:
            raise ValueError("step_size must be strictly positive.")

        self.wavefunction = wavefunction
        self.num_walkers = int(num_walkers)
        self.step_size = float(step_size)

        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

        # Initialize walker radial positions uniformly in [0.5, 2.0]
        self.positions = self.rng.uniform(0.5, 2.0, size=self.num_walkers)

    def step(self) -> float:
        """Perform a single Metropolis step across all walkers.

        Returns
        -------
        float
            Acceptance ratio of the step across all walkers.
        """
        proposals = self.positions + self.rng.normal(
            loc=0.0, scale=self.step_size, size=self.num_walkers
        )

        # Physical condition: r must be strictly positive
        valid = proposals > 1e-12

        # Ratio P(r') / P(r) = (r'/r)^2 * exp(-2 * alpha * (r' - r))
        # Evaluated safely via log-ratio to prevent numeric overflow/underflow
        log_ratio = np.full(self.num_walkers, -np.inf)

        r_curr = self.positions[valid]
        r_prop = proposals[valid]
        alpha = self.wavefunction.alpha

        log_ratio[valid] = 2.0 * np.log(r_prop / r_curr) - 2.0 * alpha * (r_prop - r_curr)

        # Metropolis acceptance test: log(u) < log_ratio
        u = self.rng.uniform(0.0, 1.0, size=self.num_walkers)
        accepted = np.log(u) < np.minimum(0.0, log_ratio)

        self.positions[accepted] = proposals[accepted]
        return float(np.mean(accepted))

    def thermalize(self, steps: int = 200) -> None:
        """Evolve the walkers without recording to reach the equilibrium distribution."""
        for _ in range(steps):
            self.step()

    def sample(self, num_steps: int, thermalization: int = 0) -> np.ndarray:
        """Collect radial samples after an optional thermalization phase.

        Parameters
        ----------
        num_steps : int
            Number of production steps to record.
        thermalization : int, default=0
            Number of initial steps to discard (burn-in).

        Returns
        -------
        np.ndarray
            Flattened array of radial positions with shape (num_steps * num_walkers,).
        """
        if thermalization > 0:
            self.thermalize(thermalization)

        history = np.empty((num_steps, self.num_walkers), dtype=float)
        for i in range(num_steps):
            self.step()
            history[i] = self.positions

        return history.flatten()
