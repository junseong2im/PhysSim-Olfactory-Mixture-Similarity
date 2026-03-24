# UMN: Universe Multi Neural Network

**Olfactory Mixture Similarity Prediction via Differentiable N-body Simulation**

Junseong Kim, University of Suwon

[![Paper](https://img.shields.io/badge/Paper-ChemRxiv-blue)](https://chemrxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

UMN (Universe Multi Neural network) is a physics-simulation-based deep learning architecture for predicting perceptual similarity between olfactory mixtures. Unlike existing GNN-based approaches, UMN maps molecular fingerprints to celestial bodies with learnable mass, position, and velocity, performs differentiable N-body gravitational simulation, and extracts orbital stability features as mixture-level representations.

UMN achieves Pearson r = 0.780 (std = 0.039) on the Snitz et al. (2013) mixture similarity dataset using 5-seed x 5-fold cross-validation without any manual feature engineering.

![Architecture](figures/architecture.png)

## Key Results

| Method | Pearson r | Evaluation | Description |
|--------|:---------:|:----------:|-------------|
| Snitz et al. (2013) optimized | 0.85 | Train/Test split | 21 hand-selected from 1433 descriptors |
| **UMN (ours)** | **0.780** | **5-seed x 5-fold CV** | **End-to-end, no feature engineering** |
| Snitz et al. (2013) simple | >= 0.49 | Train/Test split | Single structural vector, no feature selection |
| Tanimoto Similarity | 0.439 | Full data | Average pairwise Tanimoto similarity |

## Architecture

UMN consists of three modules:

1. **InputHardwareLayer**: Maps 2048-dim Morgan fingerprints to 128-dim atom vectors via grid mapping and channel transformation
2. **PhysicsProcessingEngine**: Performs differentiable N-body gravitational simulation
   - ConstellationToCelestial: Linear projection to mass (1D), position (3D), velocity (3D)
   - GravitationalEngine: Verlet integration with learnable gravitational constant
   - OrbitalStabilityEvaluator: Extracts 20D physics embeddings from trajectories
3. **SimilarityHead**: Predicts similarity from absolute difference of projected embeddings

## Installation

```bash
# Clone the repository
git clone https://github.com/junseong2im/UMN.git
cd UMN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install rdkit-pypi scipy scikit-learn pandas numpy matplotlib
```

### Requirements
- Python >= 3.10
- PyTorch >= 2.0 (CUDA recommended)
- RDKit
- NumPy, SciPy, Scikit-learn, Pandas

## Repository Structure

```
UMN/
├── models/                          # Core model modules
│   ├── olfabind_engine.py          # Physics engine (GravitationalEngine, OrbitalStabilityEvaluator)
│   ├── olfabind_input.py           # InputHardwareLayer
│   ├── olfabind_pipeline.py        # Full model pipeline
│   ├── olfabind_contrastive.py     # Contrastive learning module (v19-v21)
│   └── olfabind_ghost.py           # Ghost molecule augmentation
├── experiments/                     # Experiment scripts (v17-v25)
│   ├── v17_olfabind_validation.py  # Initial validation
│   ├── v18_olfabind_validation.py  # T-sweep + multi-restart baseline
│   ├── v19_contrastive_validation.py
│   ├── v20_triplet_physics_validation.py
│   ├── v21_enhanced_triplet.py
│   ├── v22_physics_native.py       # Physics-based loss (HNN, PINN)
│   ├── v23_freedom_stability.py    # Multi-scale simulation
│   ├── v24_internal_improvement.py # Internal mapper replacement
│   └── v25_optimization_trio.py    # Final optimization comparison
├── results/                         # Experiment results (JSON)
├── figures/                         # Paper figures
├── paper.tex                        # LaTeX source
├── UMN.pdf                         # Compiled paper
└── README.md                       # This file
```

## Experiments

We conducted systematic experiments across 9 model versions (v17-v25):

| Version | Method | Pearson r | Std |
|---------|--------|:---------:|:---:|
| v25 baseline | Original + 10-restart | **0.780** | 0.039 |
| v25-C | HP grid search | 0.732 | 0.038 |
| v25-A | +SWA, warmup | 0.727 | 0.037 |
| v18 | Original + 3-restart | 0.680 | - |
| v19 | +InfoNCE contrastive | 0.594 | 0.085 |
| v22 | +Physics-based loss | 0.532 | 0.119 |
| v24 | +MLP mapper | 0.440 | 0.090 |

### Key Findings
1. The physics engine architecture is near-optimal in its simplest form
2. Multi-restart training is the most effective optimization strategy
3. Contrastive learning and physics-based losses degrade performance in small-data regimes
4. Learned physical quantities show partial chemical interpretability (LogP: r=+0.141, HBA: r=-0.148)

## Running Experiments

```bash
# Run the best-performing configuration (v25 baseline, 10-restart)
python experiments/v25_optimization_trio.py

# Results are saved to results/v25_optimization_trio.json
```

## Dataset

This research uses the following publicly available datasets:

- **Snitz et al. (2013)**: 360 mixture pairs with perceptual similarity ratings (primary training data)
- **Ravia et al. (2020)**: 50 mixture pairs (zero-shot transfer evaluation)
- **Bushdid et al. (2014)**: 6,864 discrimination data (augmentation attempt, found ineffective)

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{kim2026umn,
  title={UMN: Universe Multi Neural Network for Olfactory Mixture Similarity Prediction via Differentiable N-body Simulation},
  author={Kim, Junseong},
  journal={ChemRxiv preprint},
  year={2026},
  doi={10.26434/chemrxiv-2026-XXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work uses the Snitz et al. (2013) mixture similarity dataset from the DREAM Olfaction Challenge. We thank the original authors for making their data publicly available.