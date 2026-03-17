from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cas_if1.data.dataset import CropConfig, ProteinRecordDataset, collate_records
from cas_if1.models.esm_if1 import ESMIF1Wrapper, compute_sequence_metrics, load_checkpoint
from cas_if1.utils.io import ensure_dir
from cas_if1.utils.protein import length_bucket


@torch.no_grad()
def run_eval(wrapper: ESMIF1Wrapper, loader: DataLoader, device: torch.device, model_label: str) -> pd.DataFrame:
    wrapper.eval()
    rows = []
    for batch in tqdm(loader, desc=f"Eval {model_label}"):
        outputs = wrapper(batch, device=device)
        metrics = compute_sequence_metrics(outputs["logits"], outputs["target"], pad_idx=wrapper.alphabet.padding_idx)
        row = {
            "sample_id": batch[0]["sample_id"],
            "length": batch[0]["length"],
            "length_bucket": length_bucket(batch[0]["length"]),
            "cluster_id": batch[0].get("cluster_id", -1),
            "keywords": ",".join(batch[0].get("keywords", [])),
            "model": model_label,
            **metrics,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_main(test_jsonl: str, finetuned_checkpoint: str, output_dir: str, device: str = "cuda") -> None:
    output_dir = ensure_dir(output_dir)
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    test_dataset = ProteinRecordDataset(test_jsonl, crop=CropConfig(max_length=1024, crop_mode="center"))
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_records)

    finetuned_wrapper, state = load_checkpoint(finetuned_checkpoint, device=device_obj)
    model_cfg = state["config"]["model"]
    baseline_wrapper = ESMIF1Wrapper(
        model_name=model_cfg["pretrained_name"],
        freeze_base=True,
        lora_cfg={"enabled": False},
    ).to(device_obj)
    baseline_wrapper.eval()

    baseline_df = run_eval(baseline_wrapper, loader, device=device_obj, model_label="baseline")
    finetuned_df = run_eval(finetuned_wrapper, loader, device=device_obj, model_label="finetuned")
    combined = pd.concat([baseline_df, finetuned_df], ignore_index=True)
    combined.to_csv(Path(output_dir) / "per_sample_metrics.csv", index=False)

    summary = (
        combined.groupby("model")[["nll", "perplexity", "recovery", "top3_recovery", "top5_recovery"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(Path(output_dir) / "summary_metrics.csv", index=False)

    summary_idx = summary.set_index("model")
    delta = {
        "nll_improvement": float(summary_idx.loc["baseline", "nll"] - summary_idx.loc["finetuned", "nll"]),
        "perplexity_improvement": float(summary_idx.loc["baseline", "perplexity"] - summary_idx.loc["finetuned", "perplexity"]),
        "recovery_gain": float(summary_idx.loc["finetuned", "recovery"] - summary_idx.loc["baseline", "recovery"]),
        "top3_recovery_gain": float(summary_idx.loc["finetuned", "top3_recovery"] - summary_idx.loc["baseline", "top3_recovery"]),
        "top5_recovery_gain": float(summary_idx.loc["finetuned", "top5_recovery"] - summary_idx.loc["baseline", "top5_recovery"]),
    }
    Path(output_dir, "delta_vs_baseline.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")

    by_bucket = (
        combined.groupby(["model", "length_bucket"])[["nll", "recovery", "top3_recovery", "top5_recovery"]]
        .mean()
        .reset_index()
    )
    by_bucket.to_csv(Path(output_dir) / "length_bucket_metrics.csv", index=False)

    by_cluster = combined.groupby(["model", "cluster_id"])[["nll", "recovery"]].mean().reset_index()
    by_cluster.to_csv(Path(output_dir) / "cluster_metrics.csv", index=False)

    cas_subtype_rows = []
    for _, row in combined.iterrows():
        labels = [token for token in row["keywords"].split(",") if token.startswith("Cas")]
        subtype = labels[0] if labels else "unknown"
        cas_subtype_rows.append({"model": row["model"], "subtype": subtype, "nll": row["nll"], "recovery": row["recovery"]})
    subtype_df = pd.DataFrame(cas_subtype_rows)
    subtype_df.groupby(["model", "subtype"])[["nll", "recovery"]].mean().reset_index().to_csv(
        Path(output_dir) / "cas_subtype_metrics.csv", index=False
    )

    Path(output_dir, "eval_config.json").write_text(json.dumps({"test_jsonl": test_jsonl, "checkpoint": finetuned_checkpoint}, indent=2), encoding="utf-8")


def summarize_main(results_dir: str, output_dir: str) -> None:
    results_dir = Path(results_dir)
    output_dir = ensure_dir(output_dir)
    summary = pd.read_csv(results_dir / "summary_metrics.csv")
    per_sample = pd.read_csv(results_dir / "per_sample_metrics.csv")

    summary.to_csv(Path(output_dir) / "summary_metrics.csv", index=False)
    pivot = per_sample.pivot(index="sample_id", columns="model", values="recovery").reset_index()
    pivot["delta_recovery"] = pivot["finetuned"] - pivot["baseline"]
    pivot.to_csv(Path(output_dir) / "per_sample_recovery_delta.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["model"], summary["recovery"], color=["#999999", "#2a7fff"])
    ax.set_ylabel("Native Sequence Recovery")
    ax.set_title("Baseline vs Finetuned")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "recovery_barplot.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, group in per_sample.groupby("model"):
        ax.hist(group["nll"], bins=20, alpha=0.5, label=model_name)
    ax.set_xlabel("Per-sample NLL")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "nll_histogram.png", dpi=200)
    plt.close(fig)
