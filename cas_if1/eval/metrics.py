from __future__ import annotations

from typing import Iterable

import numpy as np


def mean_metric(rows: Iterable[dict], key: str) -> float:
    values = [row[key] for row in rows]
    return float(np.mean(values)) if values else float("nan")

