from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cas_if1.utils.io import read_jsonl


@dataclass
class CropConfig:
    max_length: int = 512
    crop_mode: str = "random"


class ProteinRecordDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, crop: CropConfig | None = None) -> None:
        self.rows = list(read_jsonl(jsonl_path))
        self.crop = crop or CropConfig()

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, coords: np.ndarray, sequence: str) -> tuple[np.ndarray, str]:
        if len(sequence) <= self.crop.max_length:
            return coords, sequence
        if self.crop.crop_mode == "center":
            start = (len(sequence) - self.crop.max_length) // 2
        else:
            start = random.randint(0, len(sequence) - self.crop.max_length)
        end = start + self.crop.max_length
        return coords[start:end], sequence[start:end]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        coords = np.asarray(row["coords"], dtype=np.float32)
        coords, sequence = self._crop(coords, row["sequence"])
        return {
            "sample_id": row["sample_id"],
            "coords": torch.tensor(coords, dtype=torch.float32),
            "sequence": sequence,
            "length": len(sequence),
            "keywords": row.get("keywords", []),
            "cluster_id": row.get("cluster_id", -1),
        }


def collate_records(batch: list[dict]) -> list[dict]:
    return batch

