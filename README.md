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
