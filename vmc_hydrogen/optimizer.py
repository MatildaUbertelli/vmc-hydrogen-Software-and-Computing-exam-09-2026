"""Variational parameter optimization via stochastic gradient descent."""

import numpy as np
from vmc_hydrogen.hamiltonian import local_energy
from vmc_hydrogen.sampler import MultiWalkerMetropolis
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction


class VariationalOptimizer:
    """Optimizes the trial wave function parameter alpha minimizing energy expectation.

    Parameters
    ----------
    initial_alpha : float, default=0.5
        Initial guess for the variational parameter alpha.
    learning_rate : float, default=0.15
        Step size (learning rate) for gradient descent updates.
    num_walkers : int, default=500
        Number of Metropolis walkers used during sampling.
    steps_per_iter : int, default=50
        Number of sampling steps recorded per optimization iteration.
    seed : int or None, default=None
        Seed for the NumPy random number generator.

    Raises
    ------
    ValueError
        If initial_alpha <= 0, learning_rate <= 0, or num_walkers <= 0.
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
            Array of radial coordinates sampled from the trial distribution.

        Returns
        -------
        tuple of (float, float)
            Tuple containing estimated mean energy and its stochastic gradient.
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

        Parameters
        ----------
        max_iterations : int, default=30
            Maximum number of optimization cycles.
        tolerance : float, default=1e-3
            Convergence threshold on absolute gradient value.

        Returns
        -------
        dict of str to list of float
            Dictionary tracking 'alpha', 'energy', and 'gradient' histories.
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
