
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
## Acknowledgements

Parts of this project were developed with the assistance of OpenAI Codex, which contributed to code implementation, debugging, and system design. Final decisions and integration were carried out by the author.




