from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "LoRAConfig":
        return cls(**data)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def _match_name(name: str, targets: list[str] | None) -> bool:
    if not targets:
        return True
    return any(target in name for target in targets)


def apply_lora(module: nn.Module, config: LoRAConfig) -> list[str]:
    replaced: list[str] = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and _match_name(name, config.target_modules):
            setattr(module, name, LoRALinear(child, config.rank, config.alpha, config.dropout))
            replaced.append(name)
        else:
            nested = apply_lora(child, config)
            replaced.extend([f"{name}.{inner}" for inner in nested])
    return replaced


def freeze_non_lora_parameters(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
