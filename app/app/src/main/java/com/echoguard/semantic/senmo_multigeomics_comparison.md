# SeNMo vs MultiGEOmics: Benchmark Comparison & Merged Architecture Proposal

## Part 1: Performance Benchmark

Results from running the benchmark suites in `senmo_survival/run_all.py` and `multigeomics_survival/run_all.py`:

| Model | Cohort | Survival C-Index | Classification Metric |
|---|---|---|---|
| SeNMo Dedicated MLP | TCGA-BRCA | **0.8637** | N/A (survival only) |
| SeNMo Dedicated MLP | TCGA-LGG | **0.8514** | N/A (survival only) |
| SeNMo Pan-Cancer MLP | 5 cancers combined | **0.6557** (overall) / 0.7024 (mean) | N/A (survival only) |
| MultiGEOmics Graph Network | 5 cancers combined | **0.7606 ± 0.0076** | **99.52% accuracy** / 0.9999 AUROC |

### Comparison with original papers

**SeNMo** — The paper reports C-Indices between 0.62–0.85 depending on cancer type and fold. Our replication for LGG (0.8514) and BRCA (0.8637) matches or slightly beats those numbers. When pooled into a pan-cancer cohort, the C-Index drops to 0.6557, consistent with the paper's Table 5 — standard MLPs struggle to reconcile the different baseline hazard distributions across organs without any relational structure.

**MultiGEOmics** — Achieves a notably higher pan-cancer C-Index (0.7606). This tracks with the paper's results because patient similarity graphs group patients with similar molecular subtypes together, which stabilizes relative hazard rankings across tissues. The 99.52% classification accuracy makes sense too, since tissue-of-origin signatures are highly separable.

---

## Part 2: Proposed Merged Architecture — SeNMo-GNN

Idea: combine MultiGEOmics' relational clustering with SeNMo's self-normalizing convergence to beat both baselines.

```
Multi-Omics Input (Meth, miRNA, mRNA)
        │
   ┌────┴─────┐
   │           │
Dense Path   Graph Path
(SeNMo       (Adaptive
7-layer MLP,  Similarity
SELU +        Graph)
AlphaDropout)     │
   │         Self-Normalizing
   │         GraphSAGE (Max-
   │         Pooling + SELU)
   │              │
   │         Self-Normalizing
   │         Cross-Omics
   │         Attention
   │              │
   └──────┬───────┘
          │
   Dynamic Gating & Fusion
          │
   Unified Patient Embedding
          │
     ┌────┴────┐
Survival Head  Classifier Head
(log hazard)   (class logits)
```

### Core components

**1. Self-Normalizing Graph Layers (SNGNN)**
MultiGEOmics currently uses ReLU + LayerNorm in `GraphSAGELayer`. Under CoxPH loss, batch sorting introduces temporal bias that makes LayerNorm/BatchNorm unstable. Fix: swap ReLU for SELU, add AlphaDropout, and initialize weights with SeNMo's LeCun-normal scheme (W ~ N(0, 1/√d_in)). This keeps self-normalization intact through message passing.

**2. Dual-Path Fusion (Relational + Cell-Autonomous)**
- *Path A (relational)*: aggregates neighborhood features to place a patient's tumor subtype relative to the cohort.
- *Path B (dense)*: runs the raw multi-omics vector through a deep MLP, preserving individual signal that GNN averaging tends to smooth out.
- *Fusion*: a gated linear unit blends both paths —
  `h_final = g ⊙ h_GNN + (1 − g) ⊙ h_SNN`

**3. On top of the base model**
- *Self-normalized cross-omics attention*: bidirectional dot-product attention (upstream ↔ downstream) applied directly on the normalized GNN embeddings, to keep information flow stable across genomic layers.
- *GradNorm / uncertainty-based multi-task loss*: replace the static α/β weights in `JointSurvivalClassificationLoss` with dynamic weighting based on homoscedastic uncertainty, so the (easier) classification task doesn't dominate gradients over the survival objective.
