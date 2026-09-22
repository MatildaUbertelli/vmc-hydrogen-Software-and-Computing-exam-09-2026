"""Variational parameter optimization via stochastic gradient descent.

This module implements a stochastic gradient descent algorithm to minimize
the energy expectation value of the hydrogenic trial state with respect to
the variational parameter alpha, using Markov Chain Monte Carlo estimates.
"""

import numpy as np
from vmc_hydrogen.hamiltonian import local_energy
from vmc_hydrogen.sampler import MultiWalkerMetropolis
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction


class VariationalOptimizer:
    """Optimizes the trial wave function parameter alpha minimizing energy expectation.

    Parameters
    ----------
    initial_alpha : float, default=0.5
        Initial value for the variational decay parameter. Must be strictly positive.
    learning_rate : float, default=0.15
        Step size (learning rate) for gradient descent parameter updates.
    num_walkers : int, default=500
        Number of concurrent Metropolis walkers deployed per iteration.
    steps_per_iter : int, default=50
        Number of production sampling steps recorded per walker at each iteration.
    seed : int or None, default=None
        Seed for the pseudorandom number generator to ensure reproducibility.

    Raises
    ------
    ValueError
        If `initial_alpha`, `learning_rate`, `num_walkers`, or `steps_per_iter` do not satisfy positivity constraints.
    """

    def __init__(
        self,
        initial_alpha: float = 0.5,
        learning_rate: float = 0.15,
        num_walkers: int = 500,
        steps_per_iter: int = 50,
        seed: int | None = None,
    ) -> None:
        if initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be strictly positive.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        if num_walkers <= 0:
            raise ValueError("num_walkers must be strictly positive.")
        if steps_per_iter <= 0:
            raise ValueError("steps_per_iter must be strictly positive.")

        self.alpha = float(initial_alpha)
        self.learning_rate = float(learning_rate)
        self.num_walkers = int(num_walkers)
        self.steps_per_iter = int(steps_per_iter)
        self.seed = seed

        self.history_alpha: list[float] = []
        self.history_energy: list[float] = []
        self.history_gradient: list[float] = []

    def compute_gradient(self, r_samples: np.ndarray) -> tuple[float, float]:
        """Compute energy expectation and stochastic energy gradient with respect to alpha.

        Parameters
        ----------
        r_samples : np.ndarray
            Radial coordinate samples drawn from the current trial probability density.

        Returns
        -------
        tuple of (float, float)
            Tuple containing:
            - mean_e : Estimated energy expectation value <E> in Hartree.
            - gradient : Estimated energy gradient d<E>/d(alpha) with respect to alpha.
        """
        e_loc = local_energy(r_samples, self.alpha)
        d_log_psi = -r_samples

        mean_e = float(np.mean(e_loc))
        mean_dlog = float(np.mean(d_log_psi))
        mean_cross = float(np.mean(e_loc * d_log_psi))

        gradient = 2.0 * (mean_cross - mean_e * mean_dlog)
        return mean_e, gradient

    def optimize(
        self, max_iterations: int = 30, tolerance: float = 1e-3
    ) -> dict[str, list[float]]:
        """Run iterative gradient descent minimization.

        At each iteration, configurations are sampled for the current alpha value, the gradient is evaluated, and alpha is updated following: alpha_{k+1} = alpha_k - learning_rate * d<E>/d(alpha)

        Parameters
        ----------
        max_iterations : int, default=30
            Maximum number of optimization cycles to perform.
        tolerance : float, default=1e-3
            Convergence threshold on the absolute value of the estimated gradient.

        Returns
        -------
        dict of str to list of float
            Dictionary recording optimization trajectories:
            - "alpha": Sequence of variational parameters per iteration.
            - "energy": Sequence of mean local energies per iteration.
            - "gradient": Sequence of estimated gradients per iteration.
        """
        for _ in range(max_iterations):
            wf = Hydrogen1sWaveFunction(alpha=self.alpha)
            # Scale proposal step size inversely with alpha for stable acceptance
            sampler = MultiWalkerMetropolis(
                wavefunction=wf,
                num_walkers=self.num_walkers,
                step_size=0.5 / self.alpha,
                seed=self.seed,
            )

            samples = sampler.sample(
                num_steps=self.steps_per_iter, thermalization=100
            )
            energy, grad = self.compute_gradient(samples)

            self.history_alpha.append(self.alpha)
            self.history_energy.append(energy)
            self.history_gradient.append(grad)

            if abs(grad) < tolerance:
                break

            self.alpha = max(1e-3, self.alpha - self.learning_rate * grad)

        return {
            "alpha": self.history_alpha,
            "energy": self.history_energy,
            "gradient": self.history_gradient,
        }
