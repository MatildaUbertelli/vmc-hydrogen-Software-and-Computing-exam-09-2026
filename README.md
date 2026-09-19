# Variational Monte Carlo for the Hydrogen Atom Ground State

This project implements the Variational Monte Carlo (VMC) method to determine the ground-state properties of the hydrogen atom. The software combines an ensemble-based multi-walker Metropolis-Hastings sampling algorithm with an automated damped steepest descent optimization routine to find the optimal variational parameter alpha.

## Table of Contents

- [Theoretical Background](#theoretical-background)
  - [Physical Model and Atomic Units](#physical-model-and-atomic-units)
  - [Variational Principle](#variational-principle)
  - [Trial Wave Function and Local Energy](#trial-wave-function-and-local-energy)
  - [Multi-Walker Metropolis-Hastings Sampling](#multi-walker-metropolis-hastings-sampling)
  - [Stochastic Gradient and Damped Steepest Descent](#stochastic-gradient-and-damped-steepest-descent)
  - [Statistical Error Estimation and Data Blocking](#statistical-error-estimation-and-data-blocking)
- [Software Architecture](#software-architecture)
- [Installation](#installation)
- [Command-Line Usage](#command-line-usage)
  - [CLI Parameters and Flags](#cli-parameters-and-flags)
- [Python API Example](#python-api-example)
- [Running Unit Tests](#running-unit-tests)
  - [Key Validation Test Cases](#key-validation-test-cases)
  - [Executing Tests via Pytest](#executing-tests-via-pytest)
- [Diagnostic Visualizations](#diagnostic-visualizations)

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

### Multi-Walker Metropolis-Hastings Sampling

To evaluate the quantum-mechanical expectation values without computing high-dimensional analytical integrals, configurations are sampled from the probability density:

$$\rho(\mathbf{r}) = \frac{|\psi_T(\alpha, \mathbf{r})|^2}{\int |\psi_T(\alpha, \mathbf{r}')|^2 d^3\mathbf{r}'}$$

Rather than propagating a single sequential Markov chain, the simulation employs an **ensemble of $N_w$ parallel walkers** in 3D Cartesian space, initialized uniformly within a bounding volume.

At each Monte Carlo iteration:
1. **Trial Displacement:** Every walker position $\mathbf{r}_i$ ($i = 1, \dots, N_w$) is updated by proposing a random displacement drawn uniformly from $[-\delta, \delta]^3$:
   $$\mathbf{r}'_i = \mathbf{r}_i + \Delta \mathbf{r}_i$$
2. **Metropolis Acceptance Ratio:** The transition probability is governed by the squared amplitude ratio:
   $$A(\mathbf{r}_i \to \mathbf{r}'_i) = \min\left(1, \frac{|\psi_T(\alpha, \mathbf{r}'_i)|^2}{|\psi_T(\alpha, \mathbf{r}_i)|^2}\right) = \min\left(1, e^{-2\alpha (\|\mathbf{r}'_i\| - \|\mathbf{r}_i\|)}\right)$$
3. **Parallel Acceptance:** A uniform random variable $u_i \sim \mathcal{U}(0, 1)$ is evaluated simultaneously across all walkers. The step is accepted if $u_i < A$, and rejected otherwise.

The step size $\delta$ is adaptively scaled ($\delta \approx 0.5 / \alpha$) to maintain an acceptance rate within the optimal $40\%\text{--}60\%$ range, ensuring efficient exploration and rapid thermalization.

### Stochastic Gradient and Damped Steepest Descent
To optimize the variational parameter $\alpha$, we compute the gradient of the energy expectation value $\langle E(\alpha) \rangle$ with respect to $\alpha$:

$$\frac{\partial \langle E(\alpha) \rangle}{\partial \alpha} = 2 \left[ \langle E_L(\alpha, r) \cdot (-r) \rangle - \langle E_L(\alpha, r) \rangle \langle -r \rangle \right]$$

The updates are performed iteratively via damped steepest descent:

$$\alpha_{k+1} = \alpha_k - \eta_k \frac{\partial \langle E \rangle}{\partial \alpha}\Big|_{\alpha_k}$$

where $\eta_k$ is the learning rate. At each step $k$, the expectation values are evaluated empirically using an ensemble of parallel Metropolis-Hastings random walkers.

### Statistical Error Estimation and Data Blocking
In Markov Chain Monte Carlo (MCMC) simulations, successive configurations along walker trajectories are temporally correlated. Computing the standard error via naive independent sample statistics:

$$\sigma_{\text{naive}} = \frac{\sigma}{\sqrt{N}}$$

underestimates the real uncertainty because it omits the integrated autocorrelation time $\tau_{\text{int}}$.

To produce an unbiased statistical error on the estimated ground-state energy, the package applies the **Flyvbjerg-Petersen block averaging** technique (data blocking):
1. The chronological series of local energy samples $E_L$ of size $N$ is divided into $N_b$ consecutive, non-overlapping blocks of length $B$ ($N = N_b \cdot B$).
2. The sample mean of each block is evaluated:
   $$\bar{E}_k = \frac{1}{B} \sum_{i=1}^B E_L^{(k, i)}, \quad k = 1, \dots, N_b$$
3. The standard error on the mean is calculated across block averages:
   $$\sigma_{\bar{E}}(B) = \frac{1}{\sqrt{N_b (N_b - 1)}} \sqrt{\sum_{k=1}^{N_b} (\bar{E}_k - \langle E_L \rangle)^2}$$

As the block length $B$ surpasses the correlation window ($B \gg 2\tau_{\text{int}}$), consecutive block averages become mutually uncorrelated and $\sigma_{\bar{E}}(B)$ reaches a plateau. The value on this plateau represents the genuine standard error of the Monte Carlo simulation.

---

## Software Architecture

The package follows a strictly modular layout complying with academic software standards:

* **`vmc_hydrogen.wavefunctions`**: Implements the trial wave function $\psi_T(\alpha, r)$ and logarithmic derivatives.
* **`vmc_hydrogen.hamiltonian`**: Computes the analytical local energy $E_L(\alpha, r)$.
* **`vmc_hydrogen.sampler`**: Ensemble multi-walker Metropolis-Hastings sampling in 3D Cartesian coordinates.
* **`vmc_hydrogen.optimizer`**: Stochastic gradient descent engine tracking convergence histories of $\alpha$ and $\langle E_L \rangle$.
* **`vmc_hydrogen.analysis`**: Implements Flyvbjerg-Petersen block averaging (data blocking) and statistical error estimation to correct for finite autocorrelation times in Markov chains.
* **`vmc_hydrogen.plots`**: Visual diagnostic tools for radial electron probability distributions and parameter trajectories.
* **`vmc_hydrogen.__main__`**: Unified CLI interface for executing runs from the terminal.

---

## Installation

Clone the repository and install the package locally in editable mode:

```bash
git clone https://github.com/MatildaUbertelli/vmc-hydrogen-Software-and-Computing-exam-09-2026.git
cd vmc-hydrogen-Software-and-Computing-exam-09-2026
pip install -r requirements.txt
pip install -e .
```

---

## Command-Line Usage

The solver provides a fully configurable Command-Line Interface (CLI) implemented via Python's standard `argparse` module. You can execute runs directly from the terminal without modifying the source files:

```bash
python -m vmc_hydrogen --alpha-init 0.5 --walkers 500 --iterations 30 --lr 0.15 --production-steps 100 --outdir results/
```
---

## Running Unit Tests

The repository includes a comprehensive unit testing suite managed through `pytest`. The test files verify algorithmic edge cases, parameter bounds, stochastic convergence, and theoretical physics oracles.

### Key Validation Test Cases

* **Zero-Variance Principle Oracle (`test_hamiltonian.py`)**: Asserts that when $\alpha = 1.0$, the local energy collapses identically to $E_L \equiv -0.5\text{ Ha}$ across all random sample points in space, with numerical standard deviation vanishing identically ($\sigma \approx 0$).
* **Wave Function Integrity (`test_wavefunctions.py`)**: Checks boundary conditions $\lim_{r \to \infty} \psi(r) = 0$, strictly positive normalization domains, and exact values of analytical logarithmic derivatives.
* **Metropolis Sampling Acceptance (`test_sampler.py`)**: Validates that the multi-walker 3D Metropolis-Hastings chain exhibits an empirical acceptance rate within the optimal range ($40\%\text{--}60\%$) and correctly ergodically explores configuration space.
* **Flyvbjerg-Petersen Data Blocking (`test_analysis.py`)**: Asserts that on uncorrelated white noise, the blocking error remains consistent with the theoretical $1/\sqrt{N}$ behavior, while on correlated Markov chains it successfully reaches a stationary error plateau.
* **CLI & Pipeline Integration (`test_main.py`)**: Verifies parsing of flags and guarantees that end-to-end execution completes generating the requested output artifacts without raising exceptions.

### Executing Tests via Pytest

To execute all test modules with verbose reporting, run from the root directory:
```bash
pytest -v```

To run a single test module:
```bash
pytest tests/test_hamiltonian.py -v```

To generate a test coverage report directly in the terminal:
```bash
pytest --cov=vmc_hydrogen -v```

To export an interactive HTML coverage report to the `htmlcov/` directory:
```bash
pytest --cov=vmc_hydrogen --cov-report=html```

---

## Diagnostic Visualizations

Running the simulation automatically exports three diagnostic figures into the output folder (default: `results/`):

* **`optimization_trajectory.png`**: Tracks the iterative convergence of $\alpha \to 1.0$ and $\langle E_L \rangle \to -0.5\text{ Ha}$, validating algorithmic stability during stochastic gradient descent.
* **`radial_distribution.png`**: Compares the histogram of sampled electron radii against the exact analytical distribution $P(r) = 4\alpha^3 r^2 e^{-2\alpha r}$, proving correct spatial exploration of the Metropolis Markov chain.
* **`blocking_analysis.png`**: Displays standard error $\sigma_{\bar{E}}$ versus block size on a logarithmic scale, validating that the correlation plateau is reached and error bars are statistically robust.
