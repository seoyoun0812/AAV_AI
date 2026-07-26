# AAV_AI

Code for a manuscript under peer review, released anonymously for review.

## Overview

A machine learning framework for engineering AAV9 capsids with enhanced CNS
tropism. Two surface loops (VR-IV, VR-VIII) are randomized and screened in
vivo, and models predict two properties of a variant directly from its
sequence:

- **Production fitness** — how well the capsid assembles and packages.
- **Tissue enrichment** — how strongly it is enriched in a target tissue.

Both are learned on top of frozen protein language model embeddings, and both
output a per-residue attention signal for biological interpretation.

| Model | Approach | Best for |
| --- | --- | --- |
| **GraphAAV** | Graph transformer over a residue graph + attention pooling | Production fitness |
| **LightAAV** | Residual light attention over per-residue embeddings (optional LoRA) | Tissue enrichment |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── common.py     # dataset, metrics, plots, prediction export (shared)
│   ├── graphaav.py   # GraphAAV
│   └── lightaav.py   # LightAAV
└── data/             # inputs (not distributed; see data/README.md)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch-scatter` / `torch-geometric` must match your PyTorch/CUDA build. If a
plain install fails, use the prebuilt wheel index:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

LightAAV reuses the `ResidualLightAttention` head from EpHod
(Gado et al., *Nat. Mach. Intell.* 2025):

```bash
git clone https://github.com/jafetgado/EpHod.git
export PYTHONPATH="$PWD/EpHod:$PYTHONPATH"
```

## Data

Both scripts read three CSVs (train / validation / test) and one FASTA. Each
CSV needs a sequence column (`full_sequence`) and a target column
(`--target_column`); the FASTA holds the same sequences keyed by variant ID.
See `data/README.md` for the layout.

## Usage

```bash
python src/graphaav.py --target_column YOUR_TARGET_COLUMN --extract_attention
python src/lightaav.py --target_column YOUR_TARGET_COLUMN --extract_attention

# LightAAV with LoRA adaptation of the top encoder layers
python src/lightaav.py --target_column YOUR_TARGET_COLUMN \
    --finetune_esm --lora_target_layers 29 30 31 32
```

Use `--help` for the full argument list.

## Defaults

The two models were tuned independently.

| | GraphAAV | LightAAV |
| --- | --- | --- |
| Learning rate | 1e-4 | 1e-4 (LoRA: 1e-5) |
| Batch size | 64 | 32 |
| LR schedule | ReduceLROnPlateau (0.7, patience 10) | none |
| Early stopping | patience 15, after epoch 80 | patience 8, after epoch 80 |
| Encoder | `esm2_t12_35M_UR50D` | `esm2_t12_35M_UR50D` |

## Outputs

Each run writes a checkpoint, training history, training curves, a parity
plot, per-variant predictions, and (with `--extract_attention`) per-residue
attention tables to `--save_dir`.
