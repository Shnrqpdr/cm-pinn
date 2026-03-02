# Classical Mechanics Physics-Informed Neural Networks (CM-PINN)

A research project for solving classical mechanics and physics problems using **Physics-Informed Neural Networks (PINNs)**, embedding physical laws (differential equations, conservation laws) directly into neural network training.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Key Resources](#key-resources)
- [Implementation Status](#implementation-status)
- [Contributing](#contributing)
- [Related Work](#related-work)

---

## Overview

### What are PINNs?

Physics-Informed Neural Networks (PINNs) are neural networks trained to solve supervised learning tasks while respecting governing physical laws encoded as differential equations (PDEs/ODEs). Rather than learning from data alone, PINNs encode physical constraints directly into the loss function during training, enabling accurate predictions with minimal data.

**Key advantages:**
- **Data-efficient:** Compensates for scarce data using physical laws as regularizers
- **Mesh-free:** No need for domain discretization; evaluates solutions at collocation points
- **Unified framework:** Solves both forward and inverse problems with the same code
- **Differentiable physics:** Leverages automatic differentiation to compute spatial/temporal derivatives

### Project Focus: Classical Mechanics

This project applies PINNs to classical mechanics problems, particularly:

- **Störmer problem:** Motion of charged particles in dipole magnetic fields
- **Orbital mechanics:** Trajectories under gravitational and electromagnetic forces
- **Constrained motion:** Systems with conservation laws (energy, angular momentum)
- **Multi-scale dynamics:** Phenomena across different temporal/spatial scales

The goal is to develop robust PINN implementations that respect Lagrangian/Hamiltonian mechanics and preserve fundamental conservation laws.

---

## Quick Start

### Prerequisites

- **Python 3.8+**
- **PyTorch** (or JAX) for automatic differentiation
- **NumPy, SciPy** for numerical computing
- **Matplotlib** for visualization

### Installation

```bash
# Clone the repository
git clone https://github.com/Shnrqpdr/cm-pinn.git
cd cm-pinn

# Install dependencies
pip install torch numpy scipy matplotlib

# Optional: For advanced features
pip install jax jaxlib  # JAX backend
pip install jupyterlab  # Interactive notebooks
```

### Next Steps

1. **Read the theory:** Start with [`notes/pinns.md`](notes/pinns.md) for comprehensive PINN theory
2. **Review conventions:** Check [`CLAUDE.md`](CLAUDE.md) for development guidelines
3. **Explore examples:** Browse [`problems/`](problems/) for implementation examples
4. **Contribute:** See [Contributing](#contributing) for how to add new problems

---

## Repository Structure

```
cm-pinn/
├── README.md                    # This file
├── CLAUDE.md                    # Development guidelines & conventions
│
├── references/                  # Research papers & books (not in git)
│   ├── pinn/                   # PINN methodology papers
│   │   ├── PINN-original.pdf   # Raissi et al. (2019) - foundational
│   │   ├── 1808.04327v1.pdf    # Hidden Fluid Mechanics
│   │   └── ... (8 additional papers)
│   └── stormer-problem/        # Störmer problem references
│
├── problems/                    # Problem implementations
│   ├── [problem-name]/         # One directory per problem
│   │   ├── model.py            # PINN architecture
│   │   ├── train.py            # Training pipeline
│   │   ├── data/               # Datasets & initial conditions
│   │   └── results/            # Training logs, plots, saved models
│   └── ... (future problems)
│
├── notes/                       # Documentation & analysis
│   ├── pinns.md               # Comprehensive PINN knowledge base
│   │                          # (13 sections, 650+ lines)
│   └── [problem-specific]/    # Notes for each problem (dated)
│
└── .gitignore                  # Excludes references/, large data files
```

### Directory Descriptions

#### `references/`
Organized repository of research papers and books:
- **`pinn/`:** Papers on PINN methodology (9 papers covering theory, applications, analysis)
- **`stormer-problem/`:** Research on Störmer problem and related dynamics
- Papers are organized by topic for easy discovery

#### `problems/`
Implementation of each classical mechanics problem:
- **Self-contained:** Each problem folder is independent
- **Complete pipeline:** Model definition, training, validation, results
- **Reproducible:** Documentation of hyperparameters, data sources, success metrics
- Example structure:
  ```
  problems/stormer-2d/
  ├── model.py       # PINN class (MLP with tanh activation)
  ├── train.py       # Training loop (Adam + L-BFGS)
  ├── utils.py       # Helper functions (sampling, plotting)
  ├── data/
  │   └── initial_conditions.py
  └── results/
      ├── trained_model.pt
      ├── loss_history.png
      └── trajectory_comparison.png
  ```

#### `notes/`
Documentation of theory, methods, and progress:
- **`pinns.md`:** Complete reference (Theory, implementations, applications, debugging)
- **`[problem-name].md`:** Problem-specific analysis, results, lessons learned
- **Format:** Date-stamped sections for tracking evolution
- **Standards:** All equations in LaTeX (see CLAUDE.md)

---

## Key Resources

### Essential Reading

| Resource | Purpose | Content |
|----------|---------|---------|
| **[notes/pinns.md](notes/pinns.md)** | PINN Theory & Practice | 13 sections covering fundamentals, math, architectures, training, applications, implementation |
| **[CLAUDE.md](CLAUDE.md)** | Development Guidelines | Code style, documentation standards, conventions, LaTeX usage |
| **[PINN-original.pdf](references/pinn/PINN-original.pdf)** | Foundational Paper | Raissi, Perdikaris, Karniadakis (2019) — The original PINN framework |

### Quick Reference: PINN Papers in Repository

All papers are in `references/pinn/`:

1. **PINN-original.pdf** — Raissi et al. (2019) — Foundational framework
2. **1808.04327v1.pdf** — Hidden Fluid Mechanics (Navier-Stokes inverse problems)
3. **2201.05624v4.pdf** — Comprehensive survey ("Where we are and What's next")
4. **Physics_Informed_Neural_Networks.pdf** — Pedagogical chapter with PyTorch examples
5. **BDCC-06-00140-v2.pdf** — Systematic review & bibliometric analysis
6. **numerical_analysis_of_physicsinformed_neural_networks...pdf** — Rigorous error analysis (Acta Numerica)
7. **Physics informed neural networks as solver of the schrodinger equation.pdf** — Quantum mechanics application
8. **Physics informed neural networks in nano-optics and metamaterials.pdf** — Inverse problems in photonics
9. **Physics informed neural networks with scarce data.pdf** — Equation discovery from sparse data

---

## Implementation Status

### Current State

This project is in **early-stage development**. The foundation has been established:

- ✅ **Complete PINN knowledge base** (`notes/pinns.md` — 650+ lines, 13 sections)
- ✅ **Development conventions** (`CLAUDE.md` — style, documentation standards)
- ✅ **Repository structure** (organized for scalability)
- ✅ **Reference collection** (9 key PINN papers organized by topic)

### Roadmap

#### Phase 1: Core Infrastructure (Current)
- [x] PINN theory documentation
- [x] Development guidelines
- [ ] Unit testing framework
- [ ] Base PINN classes (architecture templates)

#### Phase 2: Initial Problems (Next)
- [ ] Störmer problem (2D charged particle in dipole field)
- [ ] Harmonic oscillator (validating against analytical solutions)
- [ ] Simple pendulum (nonlinear ODE system)

#### Phase 3: Advanced Features
- [ ] Energy/momentum conservation enforcement
- [ ] Multi-fidelity training (combining different data sources)
- [ ] Domain decomposition for larger systems
- [ ] Transfer learning between related problems

#### Phase 4: Publication & Benchmarking
- [ ] Performance comparison with classical methods
- [ ] Benchmark suite for classical mechanics problems
- [ ] Research papers on novel PINN variants

---

## Contributing

### Adding a New Problem

To implement a new classical mechanics problem:

1. **Create problem directory:**
   ```bash
   mkdir problems/[problem-name]
   ```

2. **Structure your code:**
   ```
   problems/[problem-name]/
   ├── README.md           # Problem description & results
   ├── model.py            # PINN class definition
   ├── train.py            # Training loop & main script
   ├── data/
   │   └── generate_data.py # Create initial conditions, reference solutions
   └── results/            # Outputs (models, plots, logs)
   ```

3. **Follow conventions:**
   - Read `CLAUDE.md` for code style and documentation standards
   - Use LaTeX for equations in markdown (per CLAUDE.md)
   - Document hyperparameters and architectural choices
   - Compare against analytical/numerical reference solutions

4. **Documentation:**
   - Write `problems/[problem-name]/README.md` with:
     - Problem statement (equations, initial conditions)
     - PINN architecture details
     - Training results & validation
     - Performance metrics
   - Create `notes/[problem-name].md` with dated sections for progress

5. **Validation:**
   - Compare against known solutions (analytical if available)
   - Report error metrics (L2 norm, relative error)
   - Include convergence plots (loss vs. epoch)
   - Visualize predictions vs. ground truth

### Development Setup

```bash
# Create a development branch
git checkout -b feature/[problem-name]

# Make changes following CLAUDE.md conventions
# Commit regularly with clear messages

# Create a pull request when ready
gh pr create --title "Add [problem name] problem"
```

### Code Standards

- **Language:** Python 3.8+
- **Framework:** PyTorch (primary) or JAX (alternative)
- **Style:** Follow PEP 8
- **Documentation:** Docstrings + inline comments for complex logic
- **Equations:** Use LaTeX in docstrings and markdown files
- **Testing:** Validate against reference solutions

---

## Related Work

### Foundational PINN Papers

**Original PINN Framework (2019):**
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707. [[PDF]](references/pinn/PINN-original.pdf)

**Key Extensions & Applications:**
- Hidden Fluid Mechanics (Raissi et al., 2018) — Navier-Stokes from visualization data
- Comprehensive Survey (Cuomo et al., 2022) — State-of-the-art review
- Error Analysis (De Ryck & Mishra, 2024) — Rigorous numerical analysis

### Related Frameworks

- **DeepXDE:** High-level framework for PINNs (supports TensorFlow/PyTorch/JAX)
- **NVIDIA Modulus:** Industrial physics-informed ML framework
- **NeuralPDE.jl:** Julia implementation for scientific computing

### Classical Mechanics Applications

This project focuses on extending PINN techniques to:
- Lagrangian/Hamiltonian systems
- Conservation law enforcement
- Multi-scale dynamics
- Charged particle motion in fields

---

## License

This project is open source. See repository for license details.

## Contact & Support

For questions, issues, or contributions:
- Open a GitHub issue for bugs or feature requests
- Submit pull requests with improvements
- Check `CLAUDE.md` for development guidelines
- Consult `notes/pinns.md` for PINN theory

---

**Last Updated:** March 2026
**Status:** Early-stage research project — foundational work complete, active development ongoing
