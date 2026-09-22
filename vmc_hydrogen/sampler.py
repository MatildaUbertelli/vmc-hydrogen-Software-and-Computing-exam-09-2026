"""Vectorized multi-walker Metropolis-Hastings radial sampler.

This module implements an ensemble Markov Chain Monte Carlo (MCMC) algorithm
using vectorized random walkers to sample from the radial probability density
distribution of the hydrogenic trial state.
"""

import numpy as np
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction


class MultiWalkerMetropolis:
    """Multi-walker Metropolis-Hastings sampler for 1D radial distributions.
    
    Propagates an ensemble of walkers in parallel across the configuration space, employing Gaussian displacement proposals subject to the boundary condition r > 0.
    
    Parameters
    ----------
    wavefunction : Hydrogen1sWaveFunction
        Trial wave function instance providing the target distribution parameters.
    num_walkers : int, default=1000
        Number of independent random walkers evolved concurrently.
    step_size : float, default=0.5
        Standard deviation of the Gaussian transition kernel.
    seed : int, np.random.Generator or None, default=None
        Seed or random number generator instance for reproducible stochastic sampling.
        
    Raises
    ------
    ValueError
        If `num_walkers` is not strictly positive or `step_size` is non-positive.
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

        # Initialize walker radial positions uniformly in [0.5, 2.0] a.u.
        self.positions = self.rng.uniform(0.5, 2.0, size=self.num_walkers)

    def step(self) -> float:
        """Perform a single Metropolis step across all walkers.

        Generates displacement proposals from a symmetric Gaussian kernel, evaluates acceptance probabilities in logarithmic space to prevent floating-point underflow/overflow, and updates walker positions.

        Returns
        -------
        float
            Empirical acceptance fraction across the ensemble for this step.
        """
        
        proposals = self.positions + self.rng.normal(
            loc=0.0, scale=self.step_size, size=self.num_walkers
        )

        # Physical boundary condition: radial distance r must be strictly positive
        valid = proposals > 1e-12

        # Logarithmic acceptance ratio:
        # ln[ P(r') / P(r) ] = 2 * ln(r' / r) - 2 * alpha * (r' - r)
        log_ratio = np.full(self.num_walkers, -np.inf)

        r_curr = self.positions[valid]
        r_prop = proposals[valid]
        alpha = self.wavefunction.alpha

        log_ratio[valid] = 2.0 * np.log(r_prop / r_curr) - 2.0 * alpha * (r_prop - r_curr)

        # Metropolis acceptance criterion evaluated via uniform stochastic threshold
        u = self.rng.uniform(0.0, 1.0, size=self.num_walkers)
        accepted = np.log(u) < np.minimum(0.0, log_ratio)

        self.positions[accepted] = proposals[accepted]
        return float(np.mean(accepted))

    def thermalize(self, steps: int = 200) -> None:
        """Evolve the ensemble to achieve statistical equilibrium (burn-in phase).

        Parameters
        ----------
        steps : int, default=200
            Number of unrecorded transition steps performed to decouple configurations from the initial spatial distribution.
        """
        for _ in range(steps):
            self.step()

    def sample(self, num_steps: int, thermalization: int = 0) -> np.ndarray:
        """Generate configurations from the stationary distribution.

        Performs optional burn-in steps followed by production sampling, aggregating walker coordinates at each step.

        Parameters
        ----------
        num_steps : int
            Number of production steps to record along the Markov chain.
        thermalization : int, default=0
            Number of initial discard steps (burn-in period).

        Returns
        -------
        np.ndarray
            One-dimensional array of accumulated radial coordinates of shape (num_steps * num_walkers,).
        """
        if thermalization > 0:
            self.thermalize(thermalization)

        history = np.empty((num_steps, self.num_walkers), dtype=float)
        for i in range(num_steps):
            self.step()
            history[i] = self.positions

        return history.flatten()
