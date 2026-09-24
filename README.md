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
  - [Executing the Test Suite](#executing-the-test-suite)
  - [Test Suite Structure](#test-suite-structure)
- [Diagnostic Visualizations](#diagnostic-visualizations)
- [Lincese](#license)
- [References](#references)

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

### Statistical Error & Data Blocking

Successive MCMC samples are serially correlated, meaning naive error estimation $\sigma / \sqrt{N}$ systematically underestimates the uncertainty. To obtain an unbiased standard error, the simulation applies **Flyvbjerg-Petersen block averaging**:

1. Divide the $N$ chronological samples into $N_b$ non-overlapping blocks of size $B$ ($N = N_b \cdot B$).
2. Compute each block mean:
   $$\bar{E}_k = \frac{1}{B} \sum_{i=1}^B E_L^{(k, i)}, \quad k = 1, \dots, N_b$$
3. Evaluate the standard error across block averages:
   $$\sigma_{\bar{E}}(B) = \sqrt{\frac{1}{N_b(N_b - 1)} \sum_{k=1}^{N_b} (\bar{E}_k - \langle E_L \rangle)^2}$$

When $B \gg 2\tau_{\text{int}}$, the block averages become decorrelated and $\sigma_{\bar{E}}(B)$ reaches a plateau, identifying the genuine statistical uncertainty of the estimated energy.

---

## Software Architecture

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
python -m vmc_hydrogen --alpha-init 0.5 --walkers 500 --iterations 30 --steps-per-iter 50 --lr 0.15 --steps 1000 --therm 200 --seed 42 --outdir results
```

### CLI Parameters and Flags

The application accepts the following command-line flags and parameters via standard `argparse`, organized into functional argument groups:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--alpha-init` | `float` | `0.5` | Initial guess for the variational decay parameter $\alpha$. Must be strictly positive ($\alpha > 0$). |
| `--lr` | `float` | `0.15` | Learning rate ($\eta$) used for stochastic gradient descent parameter updates. Must be $> 0$. |
| `--iterations` | `int` | `30` | Maximum number of gradient descent iterations during variational parameter optimization. |
| `--steps-per-iter` | `int` | `50` | Number of MCMC sampling steps per walker accumulated during each optimization iteration. Must be $> 0$. |
| `--walkers` | `int` | `500` | Number of concurrent Metropolis-Hastings random walkers propagating along $r$. |
| `--steps` | `int` | `200` | Production sampling steps per walker accumulated with the optimal parameter $\alpha^*$.|
| `--therm` | `int` | `200` | Thermalization (burn-in) steps discarded during the final high-statistics production run at $\alpha^*$.|
| `--seed` | `int` | `42` | Seed for pseudorandom number generation to ensure deterministic reproducibility. |
| `--outdir` | `Path` | `results/` | Target directory where diagnostic figures (`.png`) and numerical summaries (`.json`) are stored. |

---

## Python API Example

In addition to the CLI interface, `vmc_hydrogen` can be imported and executed programmatically as a standard Python package. Below is an updated minimal working example showing how to initialize the trial wave function, configure production and thermalization steps, evaluate local energies, and estimate the statistical error via Flyvbjerg-Petersen data blocking:

```python
from vmc_hydrogen.wavefunctions import Hydrogen1sWaveFunction
from vmc_hydrogen.sampler import MultiWalkerMetropolis
from vmc_hydrogen.hamiltonian import local_energy
from vmc_hydrogen.analysis import blocking_analysis, estimate_energy

# 1. Initialize trial wave function (e.g., at the exact ground state alpha = 1.0)
wf = Hydrogen1sWaveFunction(alpha=1.0)

# 2. Configure the 1D multi-walker radial Metropolis sampler
sampler = MultiWalkerMetropolis(
    wavefunction=wf,
    num_walkers=500,
    step_size=0.5 / wf.alpha,
    seed=42,
)

# 3. Collect radial samples using configurable thermalization (burn-in) and production steps
#    Corresponding to CLI flags --therm 200 and --steps 200
radii = sampler.sample(num_steps=200, thermalization=200)

# 4. Evaluate local energy values on sampled configurations (atomic units)
energies = local_energy(radii, alpha=wf.alpha)

# 5. Flyvbjerg-Petersen block averaging analysis
block_sizes, block_errors = blocking_analysis(energies, min_block_size=10)
mean_energy, energy_error = estimate_energy(energies)

print(f"Estimated Ground State Energy : {mean_energy:.6f} +/- {energy_error:.6f} Ha")
print(f"Exact Analytical Energy       : -0.500000 Ha")
print(f"Discrepancy                   : {abs(mean_energy - (-0.5)):.6f} Ha")
```
> **Note on Minimal API Usage:**  
> The snippet above illustrates the core computational workflow: evaluating the expectation value and statistical uncertainty for a fixed trial wave function without invoking the optimization engine or generating visual artifacts (such as diagnostic plots). For full automated parameter optimization and diagnostic figure generation, use the integrated command-line interface (`python -m vmc_hydrogen ...`) or invoke `vmc_hydrogen.optimizer.VariationalOptimizer` directly.

---

## Running Unit Tests

The test suite is built on top of `pytest` and verifies mathematical consistency, analytical oracles, numerical stability, and CLI workflows across all modules.

### Executing the Test Suite

Run the full test suite with verbose output:

```bash
pytest -v
```

To run tests along with line-by-line code coverage reporting (requires `pytest-cov`):

```bash
pytest --cov=vmc_hydrogen --cov-report=term-missing -v
```

Generate an interactive HTML coverage breakdown in `htmlcov/index.html`:

```bash
pytest --cov=vmc_hydrogen --cov-report=html -v
```

You can also target individual submodules:

```bash
pytest tests/test_wavefunctions.py -v   # Wavefunction values, densities, log-derivatives
pytest tests/test_hamiltonian.py -v     # Local energy evaluations and zero-variance oracle
pytest tests/test_sampler.py -v         # Metropolis acceptance, proposal positivity, shapes
pytest tests/test_optimizer.py -v       # Gradient oracle and variational convergence
pytest tests/test_analysis.py -v        # Flyvbjerg-Petersen blocking and standard error
pytest tests/test_plots.py -v           # Diagnostic figure generation
pytest tests/test_main.py -v            # CLI pipeline integration via tmp_path
```

---

### Test Suite Structure

The test suite contains 27 unit and integration tests structured as follows:

| Module | Tests | Target / Physics Verified |
| :--- | :---: | :--- |
| `test_wavefunctions.py` | 4 | $1s$ ansatz, radial density $P(r) \propto r^2 e^{-2\alpha r}$, log-derivative, and $\alpha > 0$ boundary guard. |
| `test_hamiltonian.py` | 3 | **Zero-variance principle** ($\mathrm{Var}(E_L) = 0$ at $\alpha = 1.0$), local energy formula, and $r \le 0$ error handling. |
| `test_sampler.py` | 3 | Multi-walker Metropolis, acceptance window $[0.30, 0.85]$, coordinate positivity ($r > 0$), and input checks. |
| `test_optimizer.py` | 3 | **Stationary point oracle** ($\nabla_\alpha \langle E \rangle = 0$ at $\alpha = 1.0$), SGD convergence, and parameter bounds. |
| `test_analysis.py` | 3 | Exact energy estimation, Flyvbjerg-Petersen blocking invariance on white noise, and empty-array guards. |
| `test_plots.py` | 3 | Non-empty diagnostic PNG generation with memory-safe teardown fixture (`plt.close`). |
| `test_main.py` | 8 | End-to-end CLI workflow with JSON export validation and 7 parametrized boundary tests raising `SystemExit`. |
| **Total** | **27** | **100% pass rate with zero warnings.** |

---

## Diagnostic Visualizations and Output Artifacts

Executing the simulation automatically exports three publication-quality diagnostic figures (saved as 150 DPI PNG files) and a structured JSON summary into the specified output directory (default: `results/`):

* **`optimization_trajectory.png`**:  
  Tracks the convergence histories across optimization iterations for both the variational parameter $\alpha_k \to \alpha^* \approx 1.0$ and the sample energy expectation value $\langle E_L \rangle \to -0.5\text{ Ha}$. Includes visual reference lines for the analytical minimum and illustrates the progressive suppression of the stochastic gradient as the algorithm approaches the stationary ground state.

* **`radial_distribution.png`**:  
  Validates the spatial exploration of the Markov chain during the final production run. The normalized histogram of accumulated radial walker coordinates $r$ is plotted against the exact theoretical probability density:
  $$P(r) = 4(\alpha^*)^3 r^2 e^{-2\alpha^* r}$$
  confirming that the equilibrium distribution has been reached following the `--therm` burn-in phase.

* **`blocking_analysis.png`**:  
  Presents the statistical error $\sigma_{\bar{E}}$ on the energy expectation value as a function of block size $B$ (Flyvbjerg-Petersen rebinning). Displays the initial underestimation due to serial autocorrelation and highlights the asymptotic plateau where block averages become independent identically distributed (i.i.d.) variables, confirming that the reported error bar is statistically robust and uncorrupted by correlation.

* **`simulation_summary.json`**:  
  A machine-readable numerical summary recording the optimal variational parameter $\alpha^*$, the estimated ground-state energy $\langle E \rangle \pm \sigma_{\bar{E}}$, the analytical discrepancy $\vert{} \langle E \rangle - (-0.5) \vert{}$, the total configuration count ($N_{\text{walkers}} \times N_{\text{steps}}$), and the full CLI execution metadata to ensure reproducibility.

---

## License

This project is licensed under the terms of the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file in the root directory for the complete license text.

---

## References

1. **Thijssen, J. M.** (2007).  
   *Computational Physics.*  
   Cambridge University Press (2nd Edition).

2. **Massachusetts Institute of Technology (MIT OpenCourseWare)** (2006).  
   *The Hydrogen Atom: Fundamentals of Photonics and Quantum Electronics.*  
   Available at: [MIT OCW 6-974 Course Material](https://ocw.mit.edu/courses/6-974-fundamentals-of-photonics-quantum-electronics-spring-2006/8a3eb732190cc7fc2520fa122bed8dcd_hydrogen_atom.pdf)

3. **Jain, S. K., & Singh, V. P.** (2003).  
   *Chapter 5 – Systems Analysis Techniques.*  
   In: *Water Resources Systems Planning and Management* (Vol. 51, Developments in Water Science, pp. 279–350). Elsevier.  
   DOI: [10.1016/S0167-5648(03)80059-X](https://www.sciencedirect.com/science/article/pii/S016756480380059X)
