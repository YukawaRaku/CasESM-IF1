# Cas-IF1

Cas protein inverse folding fine-tuning pipeline built around ESM-IF1. This repository provides a minimal runnable project for:

- Cas structure/sequence acquisition from public sources
- preprocessing PDB/mmCIF files into ESM-IF1-ready JSONL records
- parameter-efficient fine-tuning under 16GB VRAM
- inverse folding inference for RFdiffusion-style structures
- held-out evaluation against the frozen ESM-IF1 baseline

The default training strategy is LoRA-style PEFT instead of full-parameter fine-tuning. On a 16GB GPU, this is the more stable option because ESM-IF1 is non-trivial in memory once sequence length, batch size, and optimizer state are included.

## Project Tree

```text
.
├── README.md
├── requirements.txt
├── configs
│   ├── fetch_cas.yaml
│   ├── preprocess.yaml
│   └── train_lora.yaml
├── scripts
│   ├── evaluate.py
│   ├── fetch_cas_data.py
│   ├── infer.py
│   ├── preprocess_dataset.py
│   ├── summarize_results.py
│   └── train.py
└── cas_if1
    ├── __init__.py
    ├── config.py
    ├── data
    │   ├── __init__.py
    │   ├── acquisition.py
    │   ├── dataset.py
    │   └── preprocess.py
    ├── eval
    │   ├── __init__.py
    │   ├── metrics.py
    │   └── runner.py
    ├── models
    │   ├── __init__.py
    │   ├── esm_if1.py
    │   └── lora.py
    ├── train
    │   ├── __init__.py
    │   └── engine.py
    └── utils
        ├── __init__.py
        ├── io.py
        ├── logging.py
        └── protein.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Stage 1: Acquire Cas Structures

The fetch stage queries the RCSB Search API using Cas-related keywords, downloads PDB or mmCIF files, and writes normalized metadata.

```bash
python scripts/fetch_cas_data.py \
  --config configs/fetch_cas.yaml \
  --output-dir data/raw/cas_rcsb
```

Important notes:

- Cas entries are defined by keyword matching against polymer/entity descriptions and entry annotations.
- The default keyword set includes `CRISPR-associated`, `Cas9`, `Cas12`, `Cas13`, `Cpf1`, `Csm`, `Cmr`, `Cascade`, and related terms.
- Deduplication is sequence-based after metadata retrieval.

Expected outputs:

- `data/raw/cas_rcsb/structures/`
- `data/raw/cas_rcsb/metadata.jsonl`
- `data/raw/cas_rcsb/summary.json`

## Stage 2: Preprocess into Training Records

This stage parses PDB/mmCIF files, extracts backbone coordinates and aligned sequences, filters low-quality samples, then creates train/val/test splits with an approximate homology-aware grouping strategy based on sequence k-mer Jaccard clustering.

```bash
python scripts/preprocess_dataset.py \
  --config configs/preprocess.yaml \
  --input-dir data/raw/cas_rcsb \
  --output-dir data/processed/cas_if1
```

Expected outputs:

- `records.jsonl`: one record per chain
- `train.jsonl`, `val.jsonl`, `test.jsonl`
- `splits.json`
- `preprocess_report.json`

Record format:

```json
{
  "sample_id": "7S4X_A",
  "entry_id": "7S4X",
  "chain_id": "A",
  "sequence": "MNNK...",
  "length": 534,
  "coords": [
    [[12.1, 8.4, 3.0], [11.4, 9.2, 3.8], [10.1, 9.0, 3.4]],
    ...
  ],
  "source_path": "data/raw/cas_rcsb/structures/7S4X.cif",
  "keywords": ["Cas9", "CRISPR-associated"],
  "cluster_id": 12
}
```

Each residue stores `N`, `CA`, `C` coordinates in that order. Residues missing any backbone atom are removed.

## Stage 3: Train

Default config targets 16GB VRAM:

- mixed precision enabled
- gradient accumulation enabled
- max residue length cropping
- configurable dataloader workers
- checkpoint save/resume
- frozen base model with LoRA adapters

```bash
python scripts/train.py \
  --config configs/train_lora.yaml \
  --train-jsonl data/processed/cas_if1/train.jsonl \
  --val-jsonl data/processed/cas_if1/val.jsonl \
  --output-dir outputs/cas_if1_lora
```

Resume:

```bash
python scripts/train.py \
  --config configs/train_lora.yaml \
  --train-jsonl data/processed/cas_if1/train.jsonl \
  --val-jsonl data/processed/cas_if1/val.jsonl \
  --output-dir outputs/cas_if1_lora \
  --resume outputs/cas_if1_lora/checkpoints/last.pt
```

## Stage 4: Inference on RFdiffusion Structures

```bash
python scripts/infer.py \
  --checkpoint outputs/cas_if1_lora/checkpoints/best.pt \
  --input data/rfdiffusion_outputs \
  --output-dir outputs/inference_run \
  --num-samples 8 \
  --temperature 0.8
```

Outputs:

- `sequences.fasta`
- `scores.csv`
- `per_structure/*.json`

## Stage 5: Evaluation

Run both baseline and finetuned model on the held-out test set:

```bash
python scripts/evaluate.py \
  --test-jsonl data/processed/cas_if1/test.jsonl \
  --finetuned-checkpoint outputs/cas_if1_lora/checkpoints/best.pt \
  --output-dir outputs/eval_test
```

Then summarize:

```bash
python scripts/summarize_results.py \
  --results-dir outputs/eval_test \
  --output-dir outputs/eval_summary
```

## Evaluation Design

This project evaluates more than training loss:

- Negative log-likelihood and perplexity on a held-out test set
- Native sequence recovery
- Top-k residue recovery (`k=3` and `k=5`)
- Baseline frozen ESM-IF1 versus fine-tuned checkpoint
- Per-family or per-cluster reporting to verify gains are not concentrated in near-duplicates
- Length-bucket reporting to ensure improvements are not just from short sequences

Cas-specific suggestions included in the code and report:

- stratify by Cas subtype keyword where available, such as `Cas9`, `Cas12`, `Cas13`
- separately inspect nuclease active-site neighborhoods if curated annotations are added later
- compare recovery for guide-binding versus non-binding regions once domain annotations are available

## Practical Notes

- The fetch script depends on public RCSB APIs. Large downloads may take time.
- ESM-IF1 loading depends on `fair-esm`. If the exact upstream API changes, adjust `cas_if1/models/esm_if1.py`.
- The current split method is an approximate anti-leakage strategy, not a strict structural family split. For a stronger split, replace the k-mer clustering stage with MMseqs2 or Foldseek clusters.
- For long Cas proteins, training crops sequence windows. Evaluation can use full length or the same crop policy depending on config.

