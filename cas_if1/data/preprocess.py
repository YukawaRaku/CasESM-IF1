from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cas_if1.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from cas_if1.utils.protein import extract_chain_records, jaccard, sequence_kmers


@dataclass
class PreprocessConfig:
    min_length: int = 60
    max_length: int = 2048
    max_missing_fraction: float = 0.2
    split: dict[str, float] | None = None
    homology: dict[str, Any] | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.split is None:
            self.split = {"train": 0.8, "val": 0.1, "test": 0.1}
        if self.homology is None:
            self.homology = {"kmer_size": 3, "jaccard_threshold": 0.55}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessConfig":
        return cls(**data)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_clusters(records: list[dict], kmer_size: int, jaccard_threshold: float) -> list[int]:
    kmers = [sequence_kmers(record["sequence"], kmer_size) for record in records]
    uf = UnionFind(len(records))
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if jaccard(kmers[i], kmers[j]) >= jaccard_threshold:
                uf.union(i, j)
    roots = {}
    cluster_ids = []
    for i in range(len(records)):
        root = uf.find(i)
        if root not in roots:
            roots[root] = len(roots)
        cluster_ids.append(roots[root])
    return cluster_ids


def split_by_cluster(records: list[dict], split_cfg: dict[str, float], seed: int) -> dict[str, list[dict]]:
    random.seed(seed)
    clusters: dict[int, list[dict]] = {}
    for row in records:
        clusters.setdefault(row["cluster_id"], []).append(row)

    cluster_items = list(clusters.items())
    random.shuffle(cluster_items)

    total = sum(len(items) for _, items in cluster_items)
    targets = {name: total * frac for name, frac in split_cfg.items()}
    assigned = {"train": [], "val": [], "test": []}
    counts = {name: 0 for name in assigned}

    for _, rows in cluster_items:
        bucket = min(counts, key=lambda name: counts[name] / max(targets[name], 1e-8))
        assigned[bucket].extend(rows)
        counts[bucket] += len(rows)
    return assigned


def preprocess_dataset(input_dir: str | Path, output_dir: str | Path, config: PreprocessConfig) -> None:
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    metadata_rows = list(read_jsonl(input_dir / "metadata.jsonl"))

    all_records: list[dict] = []
    skipped = {"empty_parse": 0, "missing_fraction": 0, "length": 0}

    for meta in metadata_rows:
        structure_path = Path(meta["structure_path"])
        try:
            chain_records = extract_chain_records(structure_path)
        except Exception:
            skipped["empty_parse"] += 1
            continue

        if not chain_records:
            skipped["empty_parse"] += 1
            continue

        for chain in chain_records:
            if chain["length"] < config.min_length or chain["length"] > config.max_length:
                skipped["length"] += 1
                continue
            if chain["missing_fraction"] > config.max_missing_fraction:
                skipped["missing_fraction"] += 1
                continue
            all_records.append(
                {
                    "sample_id": f"{meta['entry_id']}_{chain['chain_id']}",
                    "entry_id": meta["entry_id"],
                    "chain_id": chain["chain_id"],
                    "sequence": chain["sequence"],
                    "length": chain["length"],
                    "coords": chain["coords"],
                    "source_path": str(structure_path),
                    "keywords": meta["keywords"],
                }
            )

    cluster_ids = assign_clusters(
        all_records,
        kmer_size=int(config.homology["kmer_size"]),
        jaccard_threshold=float(config.homology["jaccard_threshold"]),
    )
    for row, cluster_id in zip(all_records, cluster_ids):
        row["cluster_id"] = cluster_id

    split_rows = split_by_cluster(all_records, config.split, seed=config.seed)

    write_jsonl(output_dir / "records.jsonl", all_records)
    for split_name, rows in split_rows.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    write_json(
        output_dir / "splits.json",
        {name: [row["sample_id"] for row in rows] for name, rows in split_rows.items()},
    )
    write_json(
        output_dir / "preprocess_report.json",
        {
            "num_records": len(all_records),
            "num_clusters": len(set(cluster_ids)),
            "skipped": skipped,
            "split_sizes": {name: len(rows) for name, rows in split_rows.items()},
            "config": {
                "min_length": config.min_length,
                "max_length": config.max_length,
                "max_missing_fraction": config.max_missing_fraction,
                "split": config.split,
                "homology": config.homology,
            },
        },
    )

