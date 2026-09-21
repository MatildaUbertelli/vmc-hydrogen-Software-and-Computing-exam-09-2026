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
- [Lincese](#license)

## Theoretical Background

### Physical Model and Atomic Units
The problem is formulated in atomic units where $\[hbar] = m_e = e = 4\pi\varepsilon_0 = 1$. Distances are measured in Bohr radii ($a_0$) and energies in Hartrees ($E_h$).

Assuming an infinitely heavy nucleus fixed at the origin, the non-relativistic Hamiltonian for the single electron moving in the central Coulomb potential is:

$$\hat{H} = -\frac{1}{2}\nabla^2 - \frac{1}{r}$$

where $r$ is the electron-nucleus radial distance. The exact analytical solution for the 1s ground state is $\psi_0(r) = e^{-r}$ with ground-state energy $E_0 = -0.5\text{ Ha}$.

### Variational Principle
According to the Rayleigh-Ritz variational principle, for any normalizable trial wave function $\psi_\alpha(r)$ parameterized by $\alpha > 0$, the energy expectation value provides a rigorous upper bound to the ground-state eigenvalue $E_0$:

$$\langle E(\alpha) \rangle = \frac{\langle \psi_\alpha | \hat{H} | \psi_\alpha \rangle}{\langle \psi_\alpha | \psi_\alpha \rangle} \ge E_0$$

The equality holds if and only if $\psi_\alpha(r)$ matches the exact ground state $\psi_0(r)$.

### Trial Wave Function and Local Energy
We consider a spherically symmetric exponential trial wave function:

$$\psi_T(\alpha, r) = e^{-\alpha r}$$

The local energy $E_L(\alpha, r)$ is defined as:

$$E_L(\alpha, r) \equiv \frac{\hat{H}\psi_T(\alpha, r)}{\psi_T(\alpha, r)}$$

Using the radial Laplacian $\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}(r^2 \frac{\partial}{\partial r})$, the analytical action of the Hamiltonian yields:

$$E_L(\alpha, r) = -\frac{\alpha^2}{2} + \frac{\alpha - 1}{r}$$

When $\alpha = 1.0$, the Coulomb singularity term vanishes identically for any $r > 0$, collapsing the local energy to a constant value of $E_L \equiv -0.5\text{ Ha}$ with zero variance. This zero-variance condition serves as the analytical oracle for testing the implementation.

### Multi-Walker Metropolis-Hastings Radial Sampling

Exploiting the spherical symmetry of the $1s$ orbital, the three-dimensional volume element integrates analytically over the solid angle ($d^3\mathbf{r} = 4\pi r^2 dr$). The electron configuration is therefore directly sampled along the scalar radial coordinate $r \in (0, \infty)$ from the radial probability density:

$$P(r) = 4\alpha^3 r^2 e^{-2\alpha r}$$

The simulation propagates an **ensemble of $N_w$ parallel walkers** along the radial coordinate $r_i$ ($i = 1, \dots, N_w$), initialized uniformly in the interval $[0.5, 2.0]\,a_0$.

At each Monte Carlo iteration:
1. **Gaussian Trial Displacement:** Every walker proposes a new radial coordinate:
   $$r'_i = r_i + \Delta r_i, \quad \Delta r_i \sim \mathcal{N}(0, \delta^2)$$
   where $\delta$ is the displacement standard deviation (`step_size`). Any proposed step with $r'_i \le 0$ is strictly unphysical and assigned $P(r'_i) = 0$.
2. **Radial Metropolis Acceptance Ratio:** The acceptance probability includes both the radial wave function squared and the spherical volume factor $(r')^2$:
   $$A(r_i \to r'_i) = \min\left(1, \frac{P(r'_i)}{P(r_i)}\right) = \min\left(1, \left(\frac{r'_i}{r_i}\right)^2 e^{-2\alpha(r'_i - r_i)}\right)$$
3. **Vectorized Decision:** A uniform random variable $u_i \sim \mathcal{U}(0, 1)$ is drawn simultaneously for all walkers. The proposed state is accepted if $u_i < A(r_i \to r'_i)$ and retained otherwise.

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
* **`vmc_hydrogen.sampler`**: Ensemble multi-walker Metropolis-Hastings sampling along the 1D radial coordinate.
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

The package provides an integrated Command-Line Interface (CLI) based on Python's standard `argparse` module to run parameter optimization, production sampling, and automated diagnostic plot generation:

```bash
python -m vmc_hydrogen --alpha-init 0.5 --walkers 500 --iterations 30 --lr 0.15 --seed 42 --outdir results
```

### CLI Parameters and Flags

The application accepts the following command-line flags and parameters via standard `argparse`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--alpha-init` | `float` | `0.5` | Initial guess for the variational parameter $\alpha$. Must be strictly positive. |
| `--lr` | `float` | `0.15` | Learning rate ($\eta$) used for damped steepest descent parameter updates. |
| `--iterations` | `int` | `30` | Number of stochastic gradient descent steps during parameter optimization. |
| `--walkers` | `int` | `1000` | Number of simultaneous walkers propagating along the radial coordinate $r$. |
| `--seed` | `int` | `None` | Seed for NumPy RNG reproducibility. Pass an integer for deterministic runs. |
| `--outdir` | `str` | `results/` | Output directory where diagnostic `.png` figures and reports are saved. |

---

## Python API Example

In addition to the CLI interface, `vmc_hydrogen` can be imported and executed programmatically as a standard Python package. Below is a minimal working example showing how to initialize the trial wave function, propagate radial walkers, evaluate local energies, and estimate the statistical error via Flyvbjerg-Petersen data blocking:

```python
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction
from vmc_hydrogen.sampler import MultiWalkerMetropolis
from vmc_hydrogen.hamiltonian import local_energy
from vmc_hydrogen.analysis import blocking_analysis, estimate_energy

# 1. Initialize trial wave function (e.g. at the exact ground state alpha=1.0)
wf = Hydrogen1sWaveFunction(alpha=1.0)

# 2. Configure the 1D multi-walker radial Metropolis sampler
sampler = MultiWalkerMetropolis(
    wavefunction=wf,
    num_walkers=1000,
    step_size=0.5,
    seed=42,
)

# 3. Collect radial samples (includes 200 burn-in/thermalization steps)
radii = sampler.sample(num_steps=100, thermalization=200)

# 4. Evaluate local energy values on sampled configurations
energies = local_energy(radii, alpha=wf.alpha)

# 5. Flyvbjerg-Petersen block averaging analysis
block_sizes, block_errors = blocking_analysis(energies, min_block_size=10)
mean_energy, energy_error = estimate_energy(energies)

print(f"Estimated Ground State Energy : {mean_energy:.6f} +/- {energy_error:.6f} Ha")
print(f"Exact Analytical Energy       : -0.500000 Ha")
print(f"Discrepancy                   : {abs(mean_energy - (-0.5)):.6f} Ha")
```
> **Note on Minimal API Usage:**  
> The snippet above illustrates the core computational workflow: evaluating the expectation value and statistical uncertainty for a fixed trial wave function without invoking the optimization engine or generating visual artifacts. For full automated parameter optimization and diagnostic figure generation, use the integrated command-line interface (`python -m vmc_hydrogen ...`) or invoke `vmc_hydrogen.optimizer.VariationalOptimizer` directly.

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
pytest -v

```

To run a single test module:

```bash
pytest tests/test_hamiltonian.py -v

```

---

## Diagnostic Visualizations

Running the simulation automatically exports three diagnostic figures into the output folder (default: `results/`):

* **`optimization_trajectory.png`**: Tracks the iterative convergence of $\alpha \to 1.0$ and $\langle E_L \rangle \to -0.5\text{ Ha}$, validating algorithmic stability during stochastic gradient descent.
* **`radial_distribution.png`**: Compares the histogram of sampled electron radii against the exact analytical distribution $P(r) = 4\alpha^3 r^2 e^{-2\alpha r}$, proving correct spatial exploration of the Metropolis Markov chain.
* **`blocking_analysis.png`**: Displays standard error $\sigma_{\bar{E}}$ versus block size on a logarithmic scale, validating that the correlation plateau is reached and error bars are statistically robust.

---

## License

This project is licensed under the terms of the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file in the root directory for the complete license text.
