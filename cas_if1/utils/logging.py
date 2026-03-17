from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(output_dir: str | Path | None = None, name: str = "cas_if1") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_dir) / "run.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

