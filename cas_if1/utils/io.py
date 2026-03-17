from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def write_json(path: str | Path, obj: Mapping) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(dict(obj), handle, indent=2, ensure_ascii=True)


def list_structure_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    files = sorted(root.rglob("*.cif")) + sorted(root.rglob("*.pdb"))
    return sorted(set(files))

