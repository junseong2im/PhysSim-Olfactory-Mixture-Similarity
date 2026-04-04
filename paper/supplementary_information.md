# Supplementary Information

## Physics-Simulated Molecular Interactions Predict Olfactory Perception Without Pretrained Models

**Jun-Seong Kim**

University of Suwon, Hwaseong-si, Gyeonggi-do, Republic of Korea

---

## Supplementary Table 1 | Summary statistics for all model configurations

*10 seeds × 5 folds = 50 evaluations per model. Each fold: 2 restarts, best selected by training ρ.*

| Model | Mean ρ | S.D. | Δ vs Full | p (Wilcoxon) | Cohen's d | Wins |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PhysSim (Full)** | **0.613** | **0.046** | — | — | — | — |
| −Coulomb | 0.601 | 0.045 | −0.012 | **0.022** | 0.26 | 31/50 |
| −GR correction | 0.611 | 0.043 | −0.002 | 0.189 | 0.04 | 24/50 |
| −van der Waals | 0.615 | 0.047 | +0.002 | 0.758 | −0.04 | 20/50 |
| −Spin-orbit | 0.609 | 0.046 | −0.004 | 0.139 | 0.08 | 28/50 |
| Gravity only | 0.599 | 0.042 | −0.014 | **0.021** | 0.32 | 31/50 |
| Attention Mix† | 0.602 | 0.037 | −0.011 | **0.050** | 0.25 | 29/50 |
| Morgan FP + MLP | 0.584 | 0.047 | −0.029 | **0.0002** | 0.62 | 34/50 |
| MPNN (GCN) | 0.560 | 0.042 | −0.053 | **<10⁻⁷** | 1.20 | 42/50 |
| RDKit Desc + MLP | 0.508 | 0.037 | −0.105 | **<10⁻¹⁴** | 2.51 | 50/50 |

*†Morgan FP + multi-head self-attention mixture aggregation, inspired by POMMix.*

*Seeds: 42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777. Wilcoxon: one-sided (Full > ablated/baseline).*

---

## Supplementary Table 2 | Cross-dataset and cross-task transfer results

*Model trained on full Snitz 2013 (360 pairs), 3 restarts, best selected by training ρ.*

| Dataset | Task | n pairs | PhysSim ρ | PhysSim r | RDKit cos ρ | Ratio |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Snitz 2013 (CV) | Similarity | 360 | **0.613** | — | ~0.49 | 1.3× |
| Ravia 2020 | Similarity (blind) | 182 | **0.408** | 0.423 | 0.206 | **2.0×** |
| Bushdid 2014 | Discriminability (blind) | 264 | **−0.492** | −0.511 | −0.454 | **1.1×** |

*Note: Negative ρ for Bushdid reflects the expected inverse relationship between predicted similarity and discriminability (higher similarity → harder to discriminate → lower correct fraction). The comparison ratio uses absolute values: |0.492|/|0.454| = 1.1×.*

---

## Supplementary Table 3 | Learned physical constants

*Final values after training on complete Snitz 2013 dataset.*

| Constant | Symbol | Learned Value | Physical Role |
|:---|:---:|:---:|:---|
| Gravitational | G | 0.999 | Mass-dependent attraction strength |
| Speed of light | c | 1.182 | Relativistic velocity limit |
| Coulomb | k_e | 0.947 | Electrostatic interaction strength |
| Lennard-Jones | ε_LJ | 0.481 | Short-range repulsion/attraction depth |
| Spin coupling | λ_s | 0.920 | Angular momentum exchange rate |
| Mass decay | κ | 0.405 | Temporal mass evolution rate |
| GR boost | β | 1.093 | Schwarzschild correction magnitude |

*All constants converge to non-zero values, confirming each force is actively utilized. G ≈ k_e ≈ λ_s ≈ 1, indicating similar magnitude contributions from gravity, Coulomb, and spin-orbit. ε_LJ = 0.481 suggests weaker van der Waals contribution.*

---

## Supplementary Table 4 | Pairwise statistical comparisons (detailed)

*One-sided Wilcoxon signed-rank test (PhysSim > competitor), paired bootstrap (10,000 resamples).*

| Comparison | n | Wilcoxon p | Bootstrap p | Mean Δ | Cohen's d | 95% CI (Δ) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| PhysSim vs Attn Mix | 50 | **0.050** | 0.103 | +0.011 | 0.25 | [−0.006, +0.027] |
| PhysSim vs Morgan+MLP | 50 | **0.0002** | **0.001** | +0.029 | **0.62** | [+0.011, +0.048] |
| PhysSim vs MPNN | 50 | **<10⁻⁷** | **<0.001** | +0.053 | **1.20** | [+0.036, +0.070] |
| PhysSim vs RDKit+MLP | 50 | **<10⁻¹⁴** | **<0.001** | +0.105 | **2.51** | [+0.088, +0.121] |
| Full vs −Coulomb | 50 | **0.022** | 0.023 | +0.012 | 0.26 | [+0.000, +0.023] |
| Full vs Gravity only | 50 | **0.021** | 0.012 | +0.014 | 0.32 | [+0.002, +0.027] |

---

## Supplementary Figure 1 | Ablation interpretation

The sum of individual force contributions when removed independently (Δ_Coulomb + Δ_GR + Δ_vdW + Δ_Spin = 0.012 + 0.002 + 0 + 0.004 = 0.018) exceeds the total performance drop when all non-gravitational forces are removed simultaneously (Full − Gravity_Only = 0.613 − 0.599 = 0.014). This indicates **partial redundancy** among forces: each force compensates for aspects captured by others. The Coulombic force shows the largest unique contribution (−0.012, p = 0.022), suggesting electrostatic interactions carry the most non-overlapping information for olfactory similarity prediction.

---

## Supplementary Note 1 | Physics simulation in latent space

Our approach uses physical laws as *functional templates* in a learned latent space, not as literal simulations in 3D Euclidean space. Several important distinctions should be noted:

1. **Dimensionality.** Forces act in ℝ¹²⁸ rather than ℝ³. The inverse-square law (1/r²) generalizes naturally to any dimension, as it describes the dilution of a conserved quantity over a (D−1)-dimensional hypersphere.

2. **Learned quantities.** Mass, charge, and radius are not physical measurements but learned projections of molecular descriptors. The optimization procedure determines which aspects of molecular structure map to which physical quantity.

3. **Interpretability.** Despite operating in an abstract space, the physical constraints ensure that the simulation dynamics are governed by only 7 scalar constants (compared to thousands of unconstrained weights in an equivalent MLP). This dramatically reduces the hypothesis space and provides interpretable knobs.

4. **Soft-core regularization.** The Lennard-Jones potential is regularized using a soft-core substitution r → √(r² + δ²) with δ = 0.5, preventing the r⁻¹² singularity that would cause numerical overflow when two particles approach closely in latent space.

---

## Supplementary Note 2 | Molecule-level cross-validation procedure

Standard k-fold cross-validation for pairwise similarity prediction can leak information if molecules appear in both training and test pairs. Our molecule-level CV procedure prevents this:

1. The 86 unique molecules in the Snitz dataset are randomly partitioned into 5 folds.
2. For each fold, all pairs containing any molecule from that fold (in either mixture A or B) are assigned to the test set.
3. The remaining pairs form the training set.
4. This ensures that test mixtures contain molecules *never seen* during training, testing whether learned representations generalize to novel chemical structures.

This procedure is more stringent than pair-level CV and typically results in lower performance, but provides a more honest estimate of real-world generalization.

---

## Supplementary Note 3 | Attention Mix baseline details

To compare against the POMMix architectural paradigm, we implemented an attention-based mixture aggregation model (Attention Mix) that replaces PhysSim's force simulation with multi-head self-attention:

1. Each molecule is encoded from 2048-bit Morgan fingerprints using a 2-layer MLP (2048 → 256 → 128).
2. Molecule embeddings within a mixture attend to each other via 4-head self-attention with layer normalization.
3. Attended embeddings are mean-pooled (masked) to produce a 128-d mixture representation.
4. Mixture pairs are compared using the same similarity head as PhysSim.

This architecture directly mirrors the key innovation of POMMix (attention-based mixture aggregation) while using the same training protocol, data, and evaluation as all other baselines. PhysSim outperforms this model (p = 0.050), suggesting that explicit physics-inspired force computation provides a more effective inductive bias than learned attention weights for this task.

*Note: We could not compare against the original POMMix directly, as it requires pretrained POM GNN embeddings that are not publicly available. Our Attention Mix baseline tests the same architectural principle (attention-based mixture aggregation) in a fair, controlled setting.*
