from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import esm
import pandas as pd
import torch
import torch.nn.functional as F
from esm.inverse_folding.util import CoordBatchConverter

from cas_if1.models.lora import LoRAConfig, apply_lora, freeze_non_lora_parameters
from cas_if1.utils.io import ensure_dir, list_structure_files
from cas_if1.utils.protein import extract_chain_records, fasta_format


def load_pretrained_esm_if1(model_name: str = "esm_if1_gvp4_t16_142M_UR50"):
    if hasattr(esm.pretrained, model_name):
        loader = getattr(esm.pretrained, model_name)
        model, alphabet = loader()
    else:
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
    return model, alphabet


class ESMIF1Wrapper(torch.nn.Module):
    def __init__(self, model_name: str, freeze_base: bool = True, lora_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.model, self.alphabet = load_pretrained_esm_if1(model_name)
        self.batch_converter = CoordBatchConverter(self.alphabet)
        self.lora_cfg = LoRAConfig.from_dict(lora_cfg or {})
        self.lora_modules: list[str] = []

        if self.lora_cfg.enabled:
            self.lora_modules = apply_lora(self.model, self.lora_cfg)
        if freeze_base:
            freeze_non_lora_parameters(self.model)

    def featurize(self, batch: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
        # CoordBatchConverter expects (coords, confidence, sequence).
        tuples = [(row["coords"].cpu().numpy(), None, row["sequence"]) for row in batch]
        coords, confidence, _, tokens, padding_mask = self.batch_converter(tuples)
        return {
            "coords": coords.to(device),
            "confidence": confidence.to(device),
            "tokens": tokens.to(device),
            "padding_mask": padding_mask.to(device),
        }

    def forward(self, batch: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
        features = self.featurize(batch, device=device)
        prev_output_tokens = features["tokens"][:, :-1]
        target = features["tokens"][:, 1:]
        logits, _ = self.model(
            features["coords"],
            features["padding_mask"],
            features["confidence"],
            prev_output_tokens,
        )
        # ESM-IF1 returns logits as [batch, vocab, length]; normalize to [batch, length, vocab].
        logits = logits.transpose(1, 2).float()
        return {"logits": logits, "target": target, "padding_mask": features["padding_mask"][:, 1:]}

    def state_dict_for_save(self) -> dict[str, Any]:
        return {
            "model": self.state_dict(),
            "alphabet": {
                "padding_idx": self.alphabet.padding_idx,
                "mask_idx": getattr(self.alphabet, "mask_idx", None),
            },
        }


def compute_sequence_metrics(logits: torch.Tensor, target: torch.Tensor, pad_idx: int, topk: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1), ignore_index=pad_idx, reduction="none")
    valid_mask = target.ne(pad_idx)
    n_tokens = valid_mask.sum().item()
    nll = (loss.reshape_as(target) * valid_mask).sum().item() / max(n_tokens, 1)

    pred = logits.argmax(dim=-1)
    recovery = ((pred == target) & valid_mask).sum().item() / max(n_tokens, 1)

    metrics = {"nll": nll, "perplexity": float(torch.exp(torch.tensor(nll))), "recovery": recovery}
    for k in topk:
        topk_idx = logits.topk(k, dim=-1).indices
        topk_hit = ((topk_idx == target.unsqueeze(-1)).any(dim=-1) & valid_mask).sum().item() / max(n_tokens, 1)
        metrics[f"top{k}_recovery"] = topk_hit
    return metrics


def sample_sequences(
    wrapper: ESMIF1Wrapper,
    coords: torch.Tensor,
    num_samples: int,
    temperature: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    dummy_sequence = "A" * coords.size(0)
    batch = [{"sample_id": "sample", "coords": coords, "sequence": dummy_sequence}]
    features = wrapper.featurize(batch, device=device)
    base_prev = torch.full_like(features["tokens"][:, :-1], fill_value=wrapper.alphabet.padding_idx)
    base_prev[:, 0] = features["tokens"][:, 0]
    results = []

    for sample_idx in range(num_samples):
        prev_output_tokens = base_prev.clone()
        generated = []
        log_probs = []
        for step in range(coords.size(0)):
            logits, _ = wrapper.model(
                features["coords"],
                features["padding_mask"],
                features["confidence"],
                prev_output_tokens,
            )
            logits_step = logits[:, :, step] / max(temperature, 1e-6)
            probs = torch.softmax(logits_step, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            if step + 1 < prev_output_tokens.size(1):
                prev_output_tokens[:, step + 1] = token.squeeze(-1)
            generated.append(wrapper.alphabet.get_tok(int(token.item())))
            log_probs.append(torch.log(probs[0, token.item()] + 1e-8).item())
        seq = "".join(tok for tok in generated if len(tok) == 1 and tok.isalpha())
        results.append({"sample_index": sample_idx, "sequence": seq, "avg_log_prob": sum(log_probs) / max(len(log_probs), 1)})
    return results


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[ESMIF1Wrapper, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location=device)
    config = state["config"]
    model_cfg = config["model"]
    wrapper = ESMIF1Wrapper(
        model_name=model_cfg["pretrained_name"],
        freeze_base=model_cfg.get("freeze_base", True),
        lora_cfg=model_cfg.get("lora", {}),
    )
    wrapper.load_state_dict(state["model"], strict=False)
    wrapper.to(device)
    wrapper.eval()
    return wrapper, state


def inference_main(
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
    num_samples: int,
    temperature: float,
    device: str = "cuda",
) -> None:
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    wrapper, _ = load_checkpoint(checkpoint_path, device=device_obj)
    output_dir = ensure_dir(output_dir)
    per_structure_dir = ensure_dir(Path(output_dir) / "per_structure")

    fasta_entries = []
    score_rows = []
    structure_files = list_structure_files(input_path)
    for structure_file in structure_files:
        chain_records = extract_chain_records(structure_file)
        if not chain_records:
            continue
        chain = max(chain_records, key=lambda row: row["length"])
        coords = torch.tensor(chain["coords"], dtype=torch.float32)
        samples = sample_sequences(wrapper, coords=coords, num_samples=num_samples, temperature=temperature, device=device_obj)
        sample_payload = {
            "structure_file": str(structure_file),
            "chain_id": chain["chain_id"],
            "length": chain["length"],
            "samples": samples,
        }
        (per_structure_dir / f"{structure_file.stem}.json").write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")
        for row in samples:
            header = f"{structure_file.stem}|chain={chain['chain_id']}|sample={row['sample_index']}|score={row['avg_log_prob']:.4f}"
            fasta_entries.append((header, row["sequence"]))
            score_rows.append(
                {
                    "structure_file": str(structure_file),
                    "chain_id": chain["chain_id"],
                    "sample_index": row["sample_index"],
                    "sequence": row["sequence"],
                    "avg_log_prob": row["avg_log_prob"],
                }
            )

    Path(output_dir, "sequences.fasta").write_text(fasta_format(fasta_entries), encoding="utf-8")
    pd.DataFrame(score_rows).to_csv(Path(output_dir) / "scores.csv", index=False)
