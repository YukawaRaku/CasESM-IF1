
# Cas-IF1: Parameter-Efficient Inverse Folding for Cas Proteins

## 1. Project Goal and Motivation

This repository implements an end-to-end workflow to adapt **ESM-IF1** for **Cas protein inverse folding**, with a specific focus on structure-conditioned sequence generation for *de novo* protein design.

The core problem addressed in this project is: given a protein backbone structure, predict amino-acid sequences that are structurally compatible with that backbone. While general inverse folding models such as ESM-IF1 are trained on broad protein structure distributions, they are not specifically biased toward functionally relevant protein families such as Cas proteins.

This project focuses on introducing a **Cas-specific structural bias** into the inverse folding process. In particular, it aims to improve sequence generation for backbone structures produced by generative models such as **RFdiffusion**, where the target application is the design of novel Cas-like proteins.

The motivation is twofold:

- Enable **structure-conditioned sequence generation that is more consistent with Cas protein characteristics**, including size, fold patterns, and functional constraints.
- Increase the feasibility of **de novo Cas protein design**, by bridging generative structure models (e.g., RFdiffusion) with sequence prediction models that are adapted to the Cas protein family.

This repository therefore provides a lightweight but complete pipeline that connects:

- structural data acquisition and preprocessing,
- Cas-focused fine-tuning of an inverse folding model,
- and downstream sequence generation and evaluation,

with the goal of making Cas-oriented inverse folding workflows more reproducible and practically usable in protein design settings.

## 2. Design Rationale: Cas Adaptation, ESM-IF1, and Parameter-Efficient Fine-Tuning

This project adopts a **structure-conditioned sequence design strategy specialized for Cas proteins**, implemented through the adaptation of a pretrained inverse folding model.

Instead of treating inverse folding as a general protein modeling task, the workflow explicitly introduces a **family-specific inductive bias** toward Cas proteins. Cas proteins exhibit distinct structural and functional characteristics, including large size, modular domains, and conserved nuclease-related motifs. A generic inverse folding model, although broadly trained, does not preferentially model these features. Fine-tuning on Cas-focused structural data therefore shifts the model distribution toward sequences that are more consistent with Cas-like folds and functional constraints.

The base model used is **ESM-IF1 (`esm_if1_gvp4_t16_142M_UR50`)**, a pretrained inverse folding architecture that maps backbone coordinates to amino acid sequences. Leveraging ESM-IF1 provides a strong prior over protein geometry–sequence relationships, allowing this project to focus on **distribution adaptation rather than de novo model training**. This significantly reduces both data requirements and training complexity while preserving general structural reasoning capability.

To make this adaptation practical, the project employs **parameter-efficient fine-tuning (PEFT) via LoRA**. Instead of updating all model parameters, low-rank adaptation modules are inserted into selected linear layers while the pretrained backbone remains frozen. This design has several advantages:

- It preserves the pretrained structural knowledge of ESM-IF1 while introducing targeted modifications specific to Cas proteins.
- It reduces the number of trainable parameters by orders of magnitude, leading to more stable optimization and lower memory overhead.
- It minimizes overfitting risks when fine-tuning on a relatively narrow protein family.

Overall, this design balances **model capacity, domain adaptation, and computational efficiency**, enabling effective specialization of a general inverse folding model toward Cas protein design tasks, particularly in workflows coupled with generative backbone models such as RFdiffusion.

## 3. Actual Training Setup (from config + implementation)

Configuration source: `configs/train_lora.yaml` and `cas_if1/train/engine.py`.

- Pretrained model: `esm_if1_gvp4_t16_142M_UR50`
- Base model freezing: enabled (`freeze_base: true`)
- PEFT: LoRA enabled
- LoRA rank/alpha/dropout: `r=8`, `alpha=16`, `dropout=0.05`
- LoRA target modules: `[]` (implementation treats empty as “match all linear submodules”)
- Batch size: `1`
- Gradient accumulation: `8` steps (effective batch size ~8 when divisible)
- Mixed precision: enabled (`torch.cuda.amp.autocast` + `GradScaler`)
- Sequence crop/max length: random crop to `max_length=512` during training
- Epochs: `8`
- Optimizer: `AdamW`, learning rate `3e-4`, weight decay `0.01`
- Gradient clipping: `max_grad_norm=1.0`
- Label smoothing: `0.0`
- Checkpoint strategy: save `last.pt` every epoch; update `best.pt` when validation NLL improves
- Resume: supported via `--resume`


## 4. End-to-End Pipeline

### Stage A: Data acquisition from RCSB

Script: `scripts/fetch_cas_data.py` using `cas_if1/data/acquisition.py`.

- Queries RCSB Search API with Cas-related keywords.
- Downloads mmCIF/PDB structures (`download_format: cif` by default).
- Filters by deposited polymer monomer count (`min_length=80`, `max_length=4000`).
- Writes `metadata.jsonl` and `summary.json`.

Current run report (`data/raw/cas_rcsb/summary.json`):

- `num_unique_entries=659`
- `num_downloaded_records=621`

### Stage B: Preprocessing to training JSONL

Script: `scripts/preprocess_dataset.py` using `cas_if1/data/preprocess.py`.

- Parses structures with Biopython.
- Extracts per-chain sequence and backbone coordinates (`N`, `CA`, `C`).
- Drops residues with missing required backbone atoms.
- Filters chains by length and missing-fraction.
- Assigns approximate homology clusters via k-mer Jaccard (`k=3`, threshold `0.55`).
- Splits by cluster into train/val/test.

Current preprocess report (`data/processed/cas_if1/preprocess_report.json`):

- total records: `2153`
- clusters: `344`
- split sizes: train `1719`, val `218`, test `216`

### Stage C: Fine-tuning

Script: `scripts/train.py`.

- Builds dataloaders from JSONL.
- Applies LoRA to ESM-IF1 linear layers, freezes non-LoRA weights.
- Trains with cross-entropy token loss.
- Evaluates each epoch on validation set with NLL/perplexity/recovery/top-k.

### Stage D: Inference on RFdiffusion-generated structures

Script: `scripts/infer.py` via `inference_main`.

- Input: one `.pdb`/`.cif` or a directory.
- For each structure, parses chains and currently uses the **longest chain**.
- Autoregressive sampling with configurable `num_samples` and `temperature`.
- Outputs `sequences.fasta`, `scores.csv`, and per-structure JSON payloads.

### Stage E: Evaluation against frozen baseline

Script: `scripts/evaluate.py` + `scripts/summarize_results.py`.

- Evaluates both:
  - frozen baseline ESM-IF1 (no LoRA),
  - finetuned checkpoint.
- Test-time crop policy: center crop to length `1024`.
- Produces per-sample metrics and aggregate summaries:
  - NLL, perplexity,
  - native recovery,
  - top-3 and top-5 recovery,
  - bucketed by length, cluster, and simple Cas subtype tags.

## 5. Practical Significance

This workflow is useful as a reproducible Cas-focused inverse-folding baseline that:

- can run on commodity single-GPU hardware,
- supports RFdiffusion structure-to-sequence handoff,
- and quantifies gains versus a frozen pretrained baseline.

## 6. Limitations (Important for Honest Reporting)

- Homology control is approximate (k-mer Jaccard clustering), not strict family-level decontamination.
- Fetch-time deduplication uses a proxy hash (`entry_id + length`), not full sequence-level identity clustering.
- Training crops long chains to 512 residues; full-length behavior is only partially observed in training.
- Inference currently samples from the longest chain only when multiple chains exist.
- `grad_accum_steps` remainder batches at epoch end are not explicitly stepped if not divisible by accumulation factor.
- ESM-IF1 environment compatibility can be sensitive to PyTorch/CUDA and geometric stack versions.

## 7. Future Extensions

- Replace approximate homology split with MMseqs2/Foldseek-based clustering.
- Add chain-selection policies for complexes and interface-aware inference.
- Add domain/active-site-aware evaluation for Cas subtypes.
- Add memory-efficient long-context strategies (curriculum crop, chunked decoding, or selective full-length fine-tuning).
- Export LoRA-only checkpoints for easier sharing and deployment.

## 8. Dependency and Environment Guidance

The project imports the following libraries directly in code:

- Core: `torch`, `fair-esm`, `numpy`, `pandas`, `pyyaml`, `requests`, `tqdm`, `biopython`, `matplotlib`

Version-sensitive stack for ESM-IF1 inverse folding in practice:

- `fair-esm` (ESM-IF1 model loading and inverse-folding utilities)
- `biotite`, `torch-geometric`, `torch-scatter` (commonly required by ESM-IF1/GVP-related components depending on install path)

Because `torch-scatter` and `torch-geometric` are tightly coupled to the installed PyTorch/CUDA build, install them with a wheel/index matching your Torch version.



