# AAV_AI — sequence-based prediction of AAV9 capsid production fitness and tissue enrichment

Code accompanying the manuscript *"Unveiling the structural basis of
adeno-associated virus CNS tropism via geometric protein language modeling"*
(under review). This repository is released anonymously for peer review.

Two attention-based regressors predict a scalar property (tissue-specific
enrichment or production fitness) from a full-length AAV9 capsid sequence
carrying randomized VR-IV (aa 452–456, 5-mer) and VR-VIII (aa 586–592, 7-mer)
loops:

| Model | Head | Interpretation output |
| --- | --- | --- |
| **GraphAAV** | Graph transformer over a sequence-local residue graph, followed by global attention pooling | Per-residue importance from message magnitude × pooling gate |
| **LightAAV** | Residual light attention over per-residue embeddings | Per-residue local attention weights |

Both take per-residue embeddings from a frozen protein language model as node /
token features. LightAAV can additionally adapt the encoder with LoRA adapters
on selected transformer layers.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── common.py     # dataset, metrics, plots, checkpoint I/O shared by both models
│   ├── graphaav.py   # GraphAAV: graph transformer encoder + global attention pooling
│   └── lightaav.py   # LightAAV: residual light attention (+ optional LoRA)
├── data/             # input CSV / FASTA (not distributed; see below)
└── results/          # created at runtime
```

`common.py` holds everything the two models share — the dataset, the metric
definitions, the parity plot and the training-curve plot — so that both are
evaluated identically.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch-scatter` and `torch-geometric` must match your installed PyTorch and
CUDA build. If `pip install torch-scatter` fails, install the prebuilt wheel:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```


## Usage

### GraphAAV

```bash
python src/graphaav.py \
    --target_column production_fitness \
    --save_dir results/graphaav_fitness \
    --extract_attention
```

`--mode all` additionally exports sequence-level representations before and
after training (raw pooled ESM embeddings, pre-training GNN output,
post-training GNN output), which is what the representation-space comparison
figure is built from:

```bash
python src/graphaav.py --mode all --compare_representations \
    --target_column production_fitness
```

### LightAAV

```bash
python src/lightaav.py \
    --target_column striatum_enrichment \
    --save_dir results/lightaav_striatum \
    --extract_attention
```

With LoRA adaptation of the top four ESM1v layers:

```bash
python src/lightaav.py \
    --target_column striatum_enrichment \
    --finetune_esm --lora_target_layers 29 30 31 32 \
    --lora_rank 16 --lora_alpha 32 --esm_lr 1e-5
```

Run either script with `--help` for the full argument list.

---

## Default hyperparameters

Defaults in both scripts follow the manuscript's training section:

| Setting | Value |
| --- | --- |
| Optimiser | AdamW |
| Learning rate | 5e-5 (LoRA adapters: 1e-5) |
| Weight decay | 5e-5 (LoRA adapters: 1e-4) |
| Batch size | 64 |
| LR schedule | `ReduceLROnPlateau`, factor 0.7, patience 10 |
| Early stopping | patience 15, only considered after epoch 80 |
| Max epochs | 400 |
| Loss | MSE |
| Seed | 42 (`--seed`) |

The best checkpoint by validation loss is saved and restored before final
evaluation. Seeding covers Python, NumPy and PyTorch and sets cuDNN to
deterministic mode; scatter and atomic GPU kernels remain non-deterministic,
so repeated runs agree to within small floating-point differences rather than
bitwise.

---

## Outputs

Each run writes to `--save_dir`:

| File | Contents |
| --- | --- |
| `best_model.pt` | Best checkpoint (weights, optimiser state, metrics, resolved arguments) |
| `training_history.csv` | Per-epoch loss, R², MAE and learning rate |
| `training_curves.png` | Loss / R² / MAE / LR curves with the best epoch marked |
| `parity_plot_test.png` | Test parity plot with marginal histograms and KDEs |
| `test_predictions.csv` | Per-variant true and predicted values with error columns |
| `metrics.json` | R², RMSE and MAE for train / validation / test |
| `attention_analysis/` | Per-residue and per-variant attention tables (with `--extract_attention`) |

R² is the coefficient of determination (`sklearn.metrics.r2_score`) everywhere,
including on the parity plot.

---

## Notes

- The protein language model is frozen unless `--finetune_esm` is passed, and
  its weights are excluded from the saved checkpoints; only the trained head
  (and any LoRA adapters) are stored.
- GraphAAV builds edges between residues within a ±3 position window. Each
  directed edge (i, j) carries
  `[|i-j|/3, sign(j-i), 1{|i-j|=1}, 1{|i-j|=2}, 1{|i-j|=3}]`,
  giving position-aware learning from sequence topology alone, with no 3D
  coordinates required.
- Sequencing data underlying the target values are available from the
  corresponding author on reasonable request and are therefore not included in
  this repository.
