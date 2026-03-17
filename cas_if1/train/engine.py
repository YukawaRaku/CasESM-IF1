from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from cas_if1.data.dataset import CropConfig, ProteinRecordDataset, collate_records
from cas_if1.models.esm_if1 import ESMIF1Wrapper, compute_sequence_metrics
from cas_if1.utils.io import ensure_dir
from cas_if1.utils.logging import setup_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(jsonl_path: str, data_cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    dataset = ProteinRecordDataset(
        jsonl_path,
        crop=CropConfig(max_length=int(data_cfg["max_length"]), crop_mode=data_cfg.get("crop_mode", "random")),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=collate_records,
    )


def compute_loss(logits: torch.Tensor, target: torch.Tensor, pad_idx: int, label_smoothing: float = 0.0) -> torch.Tensor:
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        target.reshape(-1),
        ignore_index=pad_idx,
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def evaluate(wrapper: ESMIF1Wrapper, loader: DataLoader, device: torch.device, mixed_precision: bool) -> dict[str, float]:
    wrapper.eval()
    metrics_list = []
    for batch in tqdm(loader, desc="Eval", leave=False):
        with autocast(enabled=mixed_precision):
            outputs = wrapper(batch, device=device)
        metrics_list.append(
            compute_sequence_metrics(outputs["logits"], outputs["target"], pad_idx=wrapper.alphabet.padding_idx)
        )
    keys = metrics_list[0].keys() if metrics_list else []
    return {key: float(np.mean([row[key] for row in metrics_list])) for key in keys}


def save_checkpoint(
    path: Path,
    wrapper: ESMIF1Wrapper,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    best_val: float,
) -> None:
    torch.save(
        {
            "model": wrapper.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "best_val": best_val,
        },
        path,
    )


def train_main(config: dict[str, Any], train_jsonl: str, val_jsonl: str, output_dir: str, resume_path: str | None = None) -> None:
    output_dir = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(Path(output_dir) / "checkpoints")
    logger = setup_logger(output_dir)

    set_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(train_jsonl, config["data"], shuffle=True)
    val_loader = build_dataloader(val_jsonl, config["data"], shuffle=False)

    wrapper = ESMIF1Wrapper(
        model_name=config["model"]["pretrained_name"],
        freeze_base=bool(config["model"].get("freeze_base", True)),
        lora_cfg=config["model"].get("lora", {}),
    ).to(device)

    trainable_params = [param for param in wrapper.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scaler = GradScaler(enabled=bool(config["training"].get("mixed_precision", True)))
    start_epoch = 0
    global_step = 0
    best_val = math.inf

    if resume_path:
        state = torch.load(resume_path, map_location=device)
        wrapper.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_val = float(state["best_val"])
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    logger.info("Trainable parameters: %d", sum(param.numel() for param in trainable_params))
    logger.info("LoRA modules: %d", len(wrapper.lora_modules))

    history: list[dict[str, Any]] = []
    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
    mixed_precision = bool(config["training"].get("mixed_precision", True))
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))

    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}")):
            with autocast(enabled=mixed_precision):
                outputs = wrapper(batch, device=device)
                loss = compute_loss(
                    outputs["logits"],
                    outputs["target"],
                    pad_idx=wrapper.alphabet.padding_idx,
                    label_smoothing=label_smoothing,
                ) / grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(config["training"].get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_losses.append(loss.item() * grad_accum_steps)
            if global_step and global_step % int(config["training"].get("log_every", 10)) == 0:
                logger.info("epoch=%d step=%d train_loss=%.4f", epoch, global_step, np.mean(epoch_losses[-10:]))

        val_metrics = evaluate(wrapper, val_loader, device=device, mixed_precision=mixed_precision)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        logger.info("epoch=%d train_loss=%.4f val_nll=%.4f val_ppl=%.4f val_recovery=%.4f", epoch, train_loss, val_metrics["nll"], val_metrics["perplexity"], val_metrics["recovery"])

        history_row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(history_row)
        Path(output_dir, "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        save_checkpoint(
            checkpoints_dir / "last.pt",
            wrapper,
            optimizer,
            scaler,
            epoch,
            global_step,
            config,
            best_val,
        )
        if val_metrics["nll"] < best_val:
            best_val = val_metrics["nll"]
            save_checkpoint(
                checkpoints_dir / "best.pt",
                wrapper,
                optimizer,
                scaler,
                epoch,
                global_step,
                config,
                best_val,
            )

