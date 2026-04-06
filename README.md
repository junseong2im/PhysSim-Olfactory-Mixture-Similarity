# PhysSim: Physics-Simulated Molecular Interactions Predict Olfactory Perception Without Pretrained Models

Junseong Kim, University of Suwon

Preprint: https://doi.org/10.26434/chemrxiv.15001664/v1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

PhysSim is a physics simulation engine that predicts olfactory mixture similarity by evolving molecular embeddings under four fundamental forces in a learned latent space. Unlike GNN-based approaches that require pretrained molecular representations and large datasets, PhysSim encodes physical laws as inductive biases, enabling effective learning from only 360 training pairs with 162K parameters.

The four simulated forces are:

- Gravity with general relativistic (Schwarzschild) corrections
- Coulombic electrostatic interactions
- Van der Waals forces via a soft-core Lennard-Jones potential
- Spin-orbit coupling

All seven physical constants are learned end-to-end from data.


## Results

### SOTA comparison (Snitz 2013, molecule-level CV, n = 50 folds)

| Model | Spearman rho | Delta | p-value | Cohen's d | Wins |
|:---|:---:|:---:|:---:|:---:|:---:|
| PhysSim (ours) | 0.613 +/- 0.046 | --- | --- | --- | --- |
| Attention Mix | 0.602 +/- 0.037 | -0.011 | 0.050 | 0.25 | 29/50 |
| Morgan FP + MLP | 0.584 +/- 0.047 | -0.029 | 0.0002 | 0.62 | 34/50 |
| MPNN (GCN) | 0.560 +/- 0.042 | -0.053 | <1e-7 | 1.20 | 42/50 |
| RDKit Desc + MLP | 0.508 +/- 0.037 | -0.105 | <1e-14 | 2.51 | 50/50 |

Attention Mix uses Morgan fingerprints with multi-head self-attention mixture aggregation, inspired by POMMix (Nguyen et al., 2025).

### Zero-shot transfer

| Validation | Dataset | n | PhysSim rho | RDKit cos rho | Ratio |
|:---|:---|:---:|:---:|:---:|:---:|
| Within-dataset CV | Snitz 2013 | 360 | 0.613 | ~0.49 | 1.3x |
| Cross-dataset | Ravia 2020 | 182 | 0.408 | 0.206 | 2.0x |
| Cross-task | Bushdid 2014 | 264 | -0.492 | -0.454 | 1.1x |

### Ablation study (n = 50 folds)

| Configuration | rho | Delta | p | d |
|:---|:---:|:---:|:---:|:---:|
| Full (4 forces + GR) | 0.613 | --- | --- | --- |
| - Coulomb | 0.601 | -0.012 | 0.022 | 0.26 |
| Gravity only | 0.599 | -0.014 | 0.021 | 0.32 |

Coulombic interactions provide the largest unique contribution, consistent with the role of electrostatic complementarity in olfactory receptor binding.


## Architecture

```
SMILES --> RDKit descriptors (217d)
       --> Chemical encoder (MLP, 217 -> 128)
       --> Property extraction (mass, charge, radius, position, velocity, spin)
       --> Physics simulation (16 timesteps, 4 forces, 7 learnable constants)
       --> Trajectory features (134d mixture fingerprint)
       --> Similarity head (|diff|, product, cosine -> MLP -> sigmoid)
```

The simulation operates in a learned 128-dimensional latent space. Forces serve as structured inductive biases whose functional forms constrain the model's hypothesis space far more tightly than unconstrained neural network layers.


## Installation

```bash
git clone https://github.com/junseong2im/UMN-GR-Modeling-Olfactory-Mixture-Perception-via-General-Relativistic-N-body-Simulation.git
cd UMN-GR-Modeling-Olfactory-Mixture-Perception-via-General-Relativistic-N-body-Simulation

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install rdkit-pypi scipy scikit-learn pandas numpy matplotlib torch-geometric
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0 (CUDA recommended)
- RDKit
- PyTorch Geometric
- NumPy, SciPy, Scikit-learn, Pandas


## Repository structure

```
PhysSim/
├── models/
│   ├── olfabind_engine.py           # Physics engine (forces, simulation loop)
│   ├── olfabind_input.py            # Chemical encoder and property extraction
│   ├── olfabind_pipeline.py         # Full model pipeline
│   ├── olfabind_contrastive.py      # Contrastive learning module
│   └── olfabind_ghost.py            # Ghost molecule augmentation
├── experiments/
│   ├── v38_extended_validation.py   # 10-seed ablation and SOTA (n=50)
│   └── v39_final_experiments.py     # MPNN, AttnMix, Bushdid transfer
├── paper/
│   ├── main.tex                     # LaTeX source
│   ├── supplementary_information.md # Supplementary tables and notes
│   └── PhysSim_Olfactory_Prediction.pdf
├── results/                         # Experiment results (JSON)
├── figures/                         # Paper figures
└── README.md
```


## Running experiments

All experiments run on Google Colab (T4 GPU). Training scripts are self-contained.

```bash
# 10-seed ablation and SOTA comparison (~7.4 GPU-hours)
python experiments/v38_extended_validation.py

# MPNN, Attention Mix, and Bushdid transfer (~2.2 GPU-hours)
python experiments/v39_final_experiments.py
```

Results are saved as JSON files in the results directory.


## Datasets

All datasets are publicly available from the DREAM Olfaction Challenge repository.

- Snitz et al. (2013): 360 mixture pairs with perceptual similarity ratings. Primary training and CV data.
- Ravia et al. (2020): 182 mixture pairs. Zero-shot cross-dataset transfer evaluation.
- Bushdid et al. (2014): 264 mixture pairs measuring discriminability. Zero-shot cross-task transfer evaluation.


## Learned physical constants

| Constant | Symbol | Value |
|:---|:---:|:---:|
| Gravitational | G | 0.999 |
| Speed of light | c | 1.182 |
| Coulomb | k_e | 0.947 |
| Lennard-Jones | eps_LJ | 0.481 |
| Spin coupling | lambda_s | 0.920 |
| Mass decay | kappa | 0.405 |
| GR boost | beta | 1.093 |


## Citation

```bibtex
@article{kim2026physsim,
  title={Physics-Simulated Molecular Interactions Predict Olfactory Perception Without Pretrained Models},
  author={Kim, Jun-Seong},
  journal={ChemRxiv preprint},
  year={2026}
}
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgments

This work uses datasets from the DREAM Olfaction Challenge. We thank the original authors for making their data publicly available. Computational resources were provided by Google Colab.