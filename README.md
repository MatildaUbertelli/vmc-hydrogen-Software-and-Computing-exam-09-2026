# Variational Monte Carlo for the Hydrogen Atom Ground State

This project implements the Variational Monte Carlo (VMC) method to determine the ground-state properties of the hydrogen atom. The software combines an ensemble-based multi-walker Metropolis-Hastings sampling algorithm with an automated damped steepest descent optimization routine to find the optimal variational parameter alpha.

## Theoretical Background

### Physical Model and Atomic Units
The problem is formulated in atomic units where $\hbar = m_e = e = 4\pi\varepsilon_0 = 1$[cite: 1]. Distances are measured in Bohr radii ($a_0$) and energies in Hartrees ($E_h$)[cite: 1].

Assuming an infinitely heavy nucleus fixed at the origin, the non-relativistic Hamiltonian for the single electron moving in the central Coulomb potential is:

$$\hat{H} = -\frac{1}{2}\nabla^2 - \frac{1}{r}$$

where $r$ is the electron-nucleus radial distance[cite: 1]. The exact analytical solution for the 1s ground state is $\psi_0(r) = e^{-r}$ with ground-state energy $E_0 = -0.5\text{ Ha}$[cite: 1].

### Variational Principle
According to the Rayleigh-Ritz variational principle, for any normalizable trial wave function $\psi_\alpha(r)$ parameterized by $\alpha > 0$, the energy expectation value provides a rigorous upper bound to the ground-state eigenvalue $E_0$[cite: 1]:

$$\langle E(\alpha) \rangle = \frac{\langle \psi_\alpha | \hat{H} | \psi_\alpha \rangle}{\langle \psi_\alpha | \psi_\alpha \rangle} \ge E_0$$

The equality holds if and only if $\psi_\alpha(r)$ matches the exact ground state $\psi_0(r)$[cite: 1].

### Trial Wave Function and Local Energy
We consider a spherically symmetric exponential trial wave function:

$$\psi_T(\alpha, r) = e^{-\alpha r}$$

The local energy $E_L(\alpha, r)$ is defined as[cite: 1]:

$$E_L(\alpha, r) \equiv \frac{\hat{H}\psi_T(\alpha, r)}{\psi_T(\alpha, r)}$$

Using the radial Laplacian $\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}(r^2 \frac{\partial}{\partial r})$, the analytical action of the Hamiltonian yields[cite: 1]:

$$E_L(\alpha, r) = -\frac{\alpha^2}{2} + \frac{\alpha - 1}{r}$$

When $\alpha = 1.0$, the Coulomb singularity term vanishes identically for any $r > 0$, collapsing the local energy to a constant value of $E_L \equiv -0.5\text{ Ha}$ with zero variance[cite: 1]. This zero-variance condition serves as the analytical oracle for testing the implementation[cite: 1].

### Multi-Walker Metropolis Sampling
To evaluate the 3D expectation value in spherical coordinates, the integration includes the spherical volume element factor $4\pi r^2$[cite: 1]. The radial target probability density is therefore[cite: 1]:

$$D(r) \propto r^2 |\psi_T(\alpha, r)|^2 = r^2 e^{-2\alpha r}$$

Instead of relying on a single trajectory, this code evolves a population of $N_w$ parallel random walkers. At each step:
1. Every walker proposes a new radial coordinate $r_{\text{prop}} = r_{\text{current}} + \eta$, where $\eta \sim \mathcal{U}(-L/2, L/2)$[cite: 1].
2. Proposed radii with $r_{\text{prop}} \le 0$ are rejected immediately.
3. The move is accepted according to the probability[cite: 1]:
   $$A(r_{\text{current}} \to r_{\text{prop}}) = \min\left(1, \frac{r_{\text{prop}}^2 e^{-2\alpha r_{\text{prop}}}}{r_{\text{current}}^2 e^{-2\alpha r_{\text{current}}}}\right)$$
4. The expectation value $\langle E(\alpha) \rangle$ is estimated as the sample mean of $E_L(\alpha, r)$ across all active walkers after thermalization[cite: 1].

### Parameter Optimization via Damped Steepest Descent
The variational parameter $\alpha$ is optimized automatically without using finite-difference approximations[cite: 1]. Using the logarithmic derivative relation $\frac{\partial \ln \psi_T}{\partial \alpha} = -r$, the energy gradient is evaluated on the fly as[cite: 1]:

$$\frac{dE}{d\alpha} = 2 \left[ \langle E_L \rangle \langle r \rangle - \langle E_L r \rangle \right]$$

The parameter is updated iteratively according to[cite: 1]:

$$\alpha^{(k+1)} = \alpha^{(k)} - \gamma \frac{dE}{d\alpha}$$

where $\gamma$ is a damping factor introduced to prevent numerical instability from stochastic fluctuations[cite: 1]. The routine terminates once the local energy variance drops below a set threshold $\varepsilon$, indicating convergence to the zero-variance analytical ground state[cite: 1].
