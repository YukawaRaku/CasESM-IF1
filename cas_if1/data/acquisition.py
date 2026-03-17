from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from cas_if1.utils.io import ensure_dir, write_json, write_jsonl


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
RCSB_FILE_URL = "https://files.rcsb.org/download/{entry_id}.{ext}"


@dataclass
class FetchConfig:
    keywords: list[str] = field(default_factory=lambda: ["CRISPR-associated", "Cas9", "Cas12", "Cas13"])
    download_format: str = "cif"
    max_results: int = 1000
    min_length: int = 80
    max_length: int = 4000
    deduplicate: bool = True
    request_timeout: int = 60

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FetchConfig":
        return cls(**data)

def build_keyword_query(keyword: str, rows: int) -> dict[str, Any]:
    return {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "operator": "contains_phrase",
                "attribute": "struct_keywords.text",
                "value": keyword,
            },
        },
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": rows,
            }
        },
        "return_type": "entry",
    }



def search_entries(session: requests.Session, keyword: str, timeout: int) -> list[str]:
    response = session.post(RCSB_SEARCH_URL, json=build_keyword_query(keyword, rows=1000), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [item["identifier"] for item in payload.get("result_set", [])]


def fetch_entry_metadata(session: requests.Session, entry_id: str, timeout: int) -> dict[str, Any]:
    response = session.get(RCSB_ENTRY_URL.format(entry_id=entry_id), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    info = payload.get("rcsb_entry_info", {})
    keywords = payload.get("struct_keywords", {}).get("pdbx_keywords", "")
    title = payload.get("struct", {}).get("title", "")
    polymer_entities = payload.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    return {
        "entry_id": entry_id,
        "title": title,
        "keywords_text": keywords,
        "polymer_entity_ids": polymer_entities,
        "deposited_polymer_monomer_count": info.get("deposited_polymer_monomer_count"),
        "resolution_combined": payload.get("rcsb_entry_info", {}).get("resolution_combined", []),
    }


def download_structure(session: requests.Session, entry_id: str, ext: str, out_dir: Path, timeout: int) -> Path:
    response = session.get(RCSB_FILE_URL.format(entry_id=entry_id, ext=ext), timeout=timeout)
    response.raise_for_status()
    path = out_dir / f"{entry_id}.{ext}"
    path.write_bytes(response.content)
    return path


def normalize_record(metadata: dict[str, Any], matched_keywords: list[str], structure_path: Path) -> dict[str, Any]:
    length = metadata.get("deposited_polymer_monomer_count") or 0
    sequence_hash = hashlib.sha1(f"{metadata['entry_id']}::{length}".encode("utf-8")).hexdigest()
    return {
        "entry_id": metadata["entry_id"],
        "title": metadata["title"],
        "keywords": matched_keywords,
        "keywords_text": metadata["keywords_text"],
        "polymer_entity_ids": metadata["polymer_entity_ids"],
        "deposited_polymer_monomer_count": length,
        "resolution_combined": metadata["resolution_combined"],
        "structure_path": str(structure_path),
        "sequence_hash_proxy": sequence_hash,
    }


def fetch_cas_dataset(config: FetchConfig, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    structures_dir = ensure_dir(output_dir / "structures")

    session = requests.Session()
    all_hits: dict[str, set[str]] = {}
    for keyword in config.keywords:
        for entry_id in search_entries(session, keyword=keyword, timeout=config.request_timeout):
            all_hits.setdefault(entry_id, set()).add(keyword)

    entry_ids = sorted(all_hits)[: config.max_results]
    metadata_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    skipped_length = 0
    skipped_dedup = 0

    for entry_id in tqdm(entry_ids, desc="Fetching entries"):
        try:
            metadata = fetch_entry_metadata(session, entry_id=entry_id, timeout=config.request_timeout)
            length = metadata.get("deposited_polymer_monomer_count") or 0
            if length < config.min_length or length > config.max_length:
                skipped_length += 1
                continue

            structure_path = download_structure(
                session=session,
                entry_id=entry_id,
                ext=config.download_format,
                out_dir=structures_dir,
                timeout=config.request_timeout,
            )
            record = normalize_record(metadata, matched_keywords=sorted(all_hits[entry_id]), structure_path=structure_path)
            if config.deduplicate and record["sequence_hash_proxy"] in seen_hashes:
                skipped_dedup += 1
                continue
            seen_hashes.add(record["sequence_hash_proxy"])
            metadata_rows.append(record)
        except requests.RequestException:
            continue

    write_jsonl(output_dir / "metadata.jsonl", metadata_rows)
    write_json(
        output_dir / "summary.json",
        {
            "num_keywords": len(config.keywords),
            "num_unique_entries": len(all_hits),
            "num_downloaded_records": len(metadata_rows),
            "skipped_length": skipped_length,
            "skipped_deduplicate": skipped_dedup,
            "download_format": config.download_format,
        },
    )
