# Week 4: Two-Tower Architecture & Contrastive Training

**Milestone:** Two-Tower Network Architecture, Symmetric InfoNCE Loss, and Multimodal Embedding Alignment  
**Project:** SENTINEL-RS — Cross-Modal Remote Sensing Image Retrieval  
**Execution Environment:** Local Windows PC (PARAM SHAVAK Segregated)  
**Status:** IMPLEMENTED & AUDITED (Phase B Controlled Implementation)

---

## 1. Overview & Research Objectives

Week 4 implements the multimodal deep learning architecture and contrastive training pipeline for SENTINEL-RS. The network projects heterogeneous remote sensing observations from Sentinel-1 Synthetic Aperture Radar (SAR, 2 channels) and Sentinel-2 Multispectral imagery (MS, 3 channels) into a shared, sensor-agnostic 512-dimensional metric embedding space.

```
                    SAR STREAM                                   MS STREAM
            [B, 2, 224, 224] (VV, VH)                     [B, 3, 224, 224] (RGB)
                        │                                             │
               Modified ResNet50                             Standard ResNet50
           (conv1: 2->64, fc: Identity)                 (conv1: 3->64, fc: Identity)
                        │                                             │
                 [B, 2048] features                           [B, 2048] features
                        │                                             │
               SAR Projection Head                           MS Projection Head
              (2048 -> 1024 -> 512)                         (2048 -> 1024 -> 512)
                        │                                             │
                 [B, 512] vectors                              [B, 512] vectors
                        │                                             │
                        └───────────────┬─────────────────────────────┘
                                        │
                               L2 Normalization
                              (F.normalize dim=1)
                                        │
                              Symmetric InfoNCE Loss
                                   (tau = 0.07)
```

---

## 2. Directory Structure & File Provenance

```
weeks/week_04/
├── models/
│   ├── two_tower_jainam.py             # Canonical TwoTowerNetwork & InfoNCELoss (Jainam - P1)
│   ├── two_tower_adarsh.py             # Historical original TwoTower copy (Adarsh - P2)
│   ├── two_tower_meet.py               # Historical original TwoTower copy (Meet - P3)
│   ├── two_tower_fateh.py              # Historical original TwoTower copy (Fateh - P4)
│   └── resnet_meet.py                  # Standalone ResNet50 SAR encoder with Strategy B conv1 (Meet - P3)
├── training/
│   ├── train_jainam.py                 # Canonical PyTorch Lightning training script (Jainam - P1)
│   ├── train_adarsh.py                 # Historical original training script (Adarsh - P2)
│   └── train_two_tower_fateh.py        # Prototype contrastive training script (Fateh - P4)
├── evaluation/
│   └── monitor_overfitting_adarsh.py   # Generalization gap and overfitting detection (Adarsh - P2)
├── visualization/
│   └── plot_training_loss_adarsh.py    # Training and validation loss curve plotting (Adarsh - P2)
├── .backup_pre_correction_w4/          # Pristine pre-correction backups and SHA256 records
└── README.md                           # This technical documentation
```

### Contributor Original Preservation
All original contributor scripts (`two_tower_adarsh.py`, `two_tower_meet.py`, `two_tower_fateh.py`, `train_adarsh.py`, `train_two_tower_fateh.py`, `resnet_meet.py`) remain **100% untouched** to preserve individual contributor attribution and historical integrity.

---

## 3. Architecture Specifications

### 3.1 Backbones
- **SAR Tower**: Torchvision ResNet-50 modified to accept 2 input channels (VV, VH polarizations). Conv1 pretrained weights are initialized via Strategy C (spatial channel mean repeated across both input channels).
- **Multispectral Tower**: Standard 3-channel Torchvision ResNet-50 accepting standardized RGB composites (B04, B03, B02).
- **Backbone Outputs**: Both backbones replace `fc` with `nn.Identity()`, yielding unnormalized 2048-dimensional feature vectors.
- **Independence**: The two towers are separate modules with independent parameters (no weight sharing).

### 3.2 Projection Heads & BatchNorm Modes
The projection head maps 2048-D backbone features to 512-D metric embeddings. Two architectural modes are supported via `use_batchnorm`:

1. **Historical Default (`use_batchnorm=False`)**:
   $$\text{Linear}(2048 \to 1024) \longrightarrow \text{ReLU}() \longrightarrow \text{Linear}(1024 \to 512)$$
   - State-Dict Keys: `layers.0.weight`, `layers.0.bias`, `layers.2.weight`, `layers.2.bias`.
   - **Maintains 100% strict key compatibility with historical `best_model-v1.ckpt`.**
2. **Optional Research Specification (`use_batchnorm=True`)**:
   $$\text{Linear}(2048 \to 1024) \longrightarrow \text{BatchNorm1d}(1024) \longrightarrow \text{ReLU}() \longrightarrow \text{Linear}(1024 \to 512)$$
   - State-Dict Keys: `layers.0.*` (Linear), `layers.1.*` (BatchNorm1d), `layers.3.*` (Linear).
   - **BatchNorm Constraint**: Requires batch size $B \ge 2$ during training (`BatchNorm1d` cannot compute variance across a single sample).

> [!IMPORTANT]
> The two projection head modes are **not parameter-compatible**. A checkpoint trained with `use_batchnorm=False` cannot be strict-loaded into a network with `use_batchnorm=True`. The chosen mode must be explicitly specified and tracked in run metadata.

---

## 4. Contrastive Learning Formulation

### 4.1 Symmetric InfoNCE Loss
For a mini-batch of size $B$ with paired embeddings $\mathbf{Z}^S \in \mathbb{R}^{B \times 512}$ and $\mathbf{Z}^M \in \mathbb{R}^{B \times 512}$:

1. **L2 Hyperspherical Normalization**:
   $$\hat{\mathbf{z}}_i^S = \frac{\mathbf{z}_i^S}{\|\mathbf{z}_i^S\|_2}, \quad \hat{\mathbf{z}}_i^M = \frac{\mathbf{z}_i^M}{\|\mathbf{z}_i^M\|_2}$$
2. **Scaled Similarity Matrix** $\mathbf{S} \in \mathbb{R}^{B \times B}$:
   $$S_{i, j} = \frac{\langle \hat{\mathbf{z}}_i^S, \hat{\mathbf{z}}_j^M \rangle}{\tau}, \quad \tau = 0.07$$
3. **Bidirectional Cross-Entropy with Diagonal Targets** $y_i = i$:
   $$\mathcal{L}_{S \to M} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(S_{i, i})}{\sum_{j=1}^B \exp(S_{i, j})}$$
   $$\mathcal{L}_{M \to S} = -\frac{1}{B} \sum_{j=1}^B \log \frac{\exp(S_{j, j})}{\sum_{i=1}^B \exp(S_{i, j})}$$
   $$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2} \left( \mathcal{L}_{S \to M} + \mathcal{L}_{M \to S} \right)$$

### 4.2 Verified Mathematical Properties
- **Swap Invariance**: $\mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}^S, \mathbf{Z}^M) \equiv \mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}^M, \mathbf{Z}^S)$.
- **Theoretical Chance Baseline**: When similarity logits are uniform across $B$ items, the loss analytically evaluates to:
  $$\mathcal{L}_{\text{chance}} = -\ln\left(\frac{1}{B}\right) = \ln(B)$$
  For $B=16$: $\ln(16) = 2.772589...$  
  For $B=4$: $\ln(4) = 1.386294...$
- **Temperature Status**: $\tau = 0.07$ is a constant float parameter, not a learnable tensor.

---

## 5. Training Pipeline & Configurations

### 5.1 Historical Reproduction vs. New Experimental Modes

| Parameter | Historical Reproduction Setting | New Experimental Alternative |
|---|---|---|
| **Optimizer** | `Adam` | `AdamW` (selectable via `--optimizer AdamW`) |
| **Learning Rate** | `0.0001` (`1e-4`) | Configurable via `--learning-rate` |
| **Weight Decay** | `0.0` | Configurable via `--weight-decay` (e.g. `1e-4`) |
| **Scheduler** | `CosineAnnealingLR` ($T_{\text{max}}=20$) | `CosineAnnealingLR` ($T_{\text{max}}=\text{epochs}$) |
| **Projection Head** | `use_batchnorm=False` | `use_batchnorm=True` (via `--use-batchnorm`) |
| **Physical Batch Size**| `16` (adapted for 4 GB VRAM) | Configurable via `--batch-size` |
| **Precision** | `16-mixed` (Automatic Mixed Precision) | Configurable via `--precision` |
| **Gradient Clipping** | `1.0` | Configurable via `--gradient-clip-val` |
| **Random Seed** | `42` (`seed_everything(42)`) | Configurable via `--seed` |

### 5.2 Isolated Output Invariant
New training runs must specify an output directory (default: `results/local_pc/week_04/runs/canonical_run/` or user-defined `--run-dir`). Historical artifacts in `results/local_pc/week_04/` (including historical `best_model-v1.ckpt.readme.txt`, `metrics.csv`, and `Week_4_training_record.txt`) are strictly protected and **never overwritten**.

---

## 6. Historical Checkpoint Provenance & Current Availability

- **Historical Artifact**: `best_model-v1.ckpt` (598.95 MB, 628,041,728 bytes).
- **Format**: Complete PyTorch Lightning state dictionary (644 parameter tensors prefixed with `network.`, epoch 19, optimizer states, hyperparameters).
- **Provenance**: Trained by Jainam on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) using batch size 16, fp16-mixed, `Adam` optimizer ($lr=1e-4$), and `use_batchnorm=False`.
- **Current Workstation Status**: The physical 598.95 MB binary is **not present locally** on this machine; it is represented by `best_model-v1.ckpt.readme.txt` linking to Google Drive. Local compatibility testing utilizes simulated state dictionaries matching the exact keys and shapes.

---

## 7. Execution Commands

### 7.1 Running Overfitting Analysis
```bash
python weeks/week_04/evaluation/monitor_overfitting_adarsh.py
```

### 7.2 Running Training Loss Plotting
```bash
python weeks/week_04/visualization/plot_training_loss_adarsh.py
```

### 7.3 Canonical Training Script (Dry-Run / Help)
```bash
python weeks/week_04/training/train_jainam.py --help
```

