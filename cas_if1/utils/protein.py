from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, Polypeptide
from Bio.Data.IUPACData import protein_letters_3to1


BACKBONE_ATOMS = ("N", "CA", "C")


def residue_to_aa(residue) -> str | None:
    resname = residue.get_resname().strip()
    if not Polypeptide.is_aa(resname, standard=True):
        return None
    return protein_letters_3to1.get(resname.capitalize())


def load_structure(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def extract_chain_records(path: str | Path) -> list[dict]:
    structure = load_structure(path)
    model = next(structure.get_models())
    records: list[dict] = []

    for chain in model.get_chains():
        residues = [res for res in chain.get_residues() if res.id[0] == " "]
        sequence = []
        coords = []
        total_seen = 0
        kept = 0

        for residue in residues:
            total_seen += 1
            aa = residue_to_aa(residue)
            if aa is None:
                continue
            atom_coords = []
            missing_atom = False
            for atom_name in BACKBONE_ATOMS:
                if atom_name not in residue:
                    missing_atom = True
                    break
                atom_coords.append(residue[atom_name].coord.astype(float).tolist())
            if missing_atom:
                continue
            kept += 1
            sequence.append(aa)
            coords.append(atom_coords)

        if not sequence:
            continue

        records.append(
            {
                "chain_id": chain.id,
                "sequence": "".join(sequence),
                "coords": coords,
                "length": len(sequence),
                "missing_fraction": 1.0 - (kept / max(total_seen, 1)),
            }
        )
    return records


def fasta_format(entries: Iterable[tuple[str, str]]) -> str:
    lines: list[str] = []
    for header, sequence in entries:
        lines.append(f">{header}")
        for start in range(0, len(sequence), 80):
            lines.append(sequence[start : start + 80])
    return "\n".join(lines) + "\n"


def sequence_kmers(sequence: str, k: int) -> set[str]:
    if len(sequence) < k:
        return {sequence}
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def length_bucket(length: int) -> str:
    if length < 256:
        return "<256"
    if length < 512:
        return "256-511"
    if length < 1024:
        return "512-1023"
    return ">=1024"


def to_numpy_coords(coords: list[list[list[float]]]) -> np.ndarray:
    return np.asarray(coords, dtype=np.float32)
