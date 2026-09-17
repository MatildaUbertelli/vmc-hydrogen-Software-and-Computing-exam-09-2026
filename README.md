# Variational Monte Carlo for the Hydrogen Atom Ground State

This project implements the Variational Monte Carlo (VMC) method to determine the ground-state properties of the hydrogen atom. The software combines an ensemble-based multi-walker Metropolis-Hastings sampling algorithm with an automated damped steepest descent optimization routine to find the optimal variational parameter alpha.

## Theoretical Background

### Physical Model and Atomic Units
The problem is formulated in atomic units where $\hbar = m_e = e = 4\pi\varepsilon_0 = 1$. Distances are measured in Bohr radii ($a_0$) and energies in Hartrees ($E_h$).

Assuming an infinitely heavy nucleus fixed at the origin, the non-relativistic Hamiltonian for the single electron moving in the central Coulomb potential is:

$$\hat{H} = -\frac{1}{2}\nabla^2 - \frac{1}{r}$$

where $r$ is the electron-nucleus radial distance. The exact analytical solution for the 1s ground state is $\psi_0(r) = e^{-r}$ with ground-state energy $E_0 = -0.5\text{ Ha}$.

### Variational Principle
According to the Rayleigh-Ritz variational principle, for any normalizable trial wave function $\psi_\alpha(r)$ parameterized by $\alpha > 0$, the energy expectation value provides a rigorous upper bound to the ground-state eigenvalue $E_0$[cite: 1]:

$$\langle E(\alpha) \rangle = \frac{\langle \psi_\alpha | \hat{H} | \psi_\alpha \rangle}{\langle \psi_\alpha | \psi_\alpha \rangle} \ge E_0$$

The equality holds if and only if $\psi_\alpha(r)$ matches the exact ground state $\psi_0(r)$[cite: 1].

### Trial Wave Function and Local Energy
We consider a spherically symmetric exponential trial wave function:

$$\psi_T(\alpha, r) = e^{-\alpha r}$$

The local energy $E_L(\alpha, r)$ is defined as:

$$E_L(\alpha, r) \equiv \frac{\hat{H}\psi_T(\alpha, r)}{\psi_T(\alpha, r)}$$

Using the radial Laplacian $\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}(r^2 \frac{\partial}{\partial r})$, the analytical action of the Hamiltonian yields:

$$E_L(\alpha, r) = -\frac{\alpha^2}{2} + \frac{\alpha - 1}{r}$$

When $\alpha = 1.0$, the Coulomb singularity term vanishes identically for any $r > 0$, collapsing the local energy to a constant value of $E_L \equiv -0.5\text{ Ha}$ with zero variance. This zero-variance condition serves as the analytical oracle for testing the implementation.

### Stochastic Gradient and Damped Steepest Descent

To optimize the variational parameter $\alpha$, we compute the gradient of the energy expectation value $\langle E(\alpha) \rangle$ with respect to $\alpha$:

$$\frac{\partial \langle E(\alpha) \rangle}{\partial \alpha} = 2 \left[ \langle E_L(\alpha, r) \cdot (-r) \rangle - \langle E_L(\alpha, r) \rangle \langle -r \rangle \right]$$

The updates are performed iteratively via damped steepest descent:

$$\alpha_{k+1} = \alpha_k - \eta_k \frac{\partial \langle E \rangle}{\partial \alpha}\Big|_{\alpha_k}$$

where $\eta_k$ is the learning rate. At each step $k$, the expectation values are evaluated empirically using an ensemble of parallel Metropolis-Hastings random walkers.

---

## Software Architecture

The package follows a strictly modular layout complying with academic software standards:

* **`vmc_hydrogen.wavefunctions`**: Implements the trial wave function $\psi_T(\alpha, r)$ and logarithmic derivatives.
* **`vmc_hydrogen.hamiltonian`**: Computes the analytical local energy $E_L(\alpha, r)$.
* **`vmc_hydrogen.sampler`**: Ensemble multi-walker Metropolis-Hastings sampling in 3D Cartesian coordinates.
* **`vmc_hydrogen.optimizer`**: Stochastic gradient descent engine tracking convergence histories of $\alpha$ and $\langle E_L \rangle$.
* **`vmc_hydrogen.plots`**: Visual diagnostic tools for radial electron probability distributions and parameter trajectories.
* **`vmc_hydrogen.__main__`**: Unified CLI interface for executing runs from the terminal.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone [https://github.com/YOUR_USERNAME/vmc_hydrogen.git](https://github.com/YOUR_USERNAME/vmc_hydrogen.git)
cd vmc_hydrogen
pip install -e .
pip install -r requirements.txt
