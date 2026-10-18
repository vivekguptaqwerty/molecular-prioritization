import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch

from infer_affinity import load_model
from protliggnn_train import build_protein_pocket_graph, mol_to_ligand_graph


DEFAULT_DECOYS_DIR = Path("data") / "CASF-2016" / "decoys_docking"
DEFAULT_CORESET_DIR = Path("data") / "CASF-2016" / "coreset"
DEFAULT_CHECKPOINT = Path("outputs") / "best_protliggnn.pt"
OUTPUT_DIR = Path("outputs") / "casf2016_benchmark"
RESULTS_CSV = OUTPUT_DIR / "docking_power_results.csv"
SUCCESS_CUTOFFS = (1.0, 2.0, 3.0)
TOP_N = (1, 2, 3)


def parse_args():
    parser = argparse.ArgumentParser(description="CASF-2016 docking power evaluation.")
    parser.add_argument("--decoys_dir", type=str, default=str(DEFAULT_DECOYS_DIR))
    parser.add_argument("--coreset_dir", type=str, default=str(DEFAULT_CORESET_DIR))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_rmsd_index(path: Path) -> Dict[str, float]:
    rows: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0].startswith("#"):
            continue
        try:
            rows[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return rows


def split_mol2_blocks(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split("@<TRIPOS>MOLECULE")[1:]
    return ["@<TRIPOS>MOLECULE" + chunk for chunk in chunks]


def pose_name_from_block(block: str) -> Optional[str]:
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and "Name:" in stripped:
            return stripped.split("Name:", 1)[1].strip()
    return None


def already_done_complexes(results_csv: Path) -> set:
    if not results_csv.exists():
        return set()
    done = set()
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add(row["pdb_id"])
    return done


def append_result_rows(results_csv: Path, rows: List[Dict[str, object]], write_header: bool):
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pdb_id", "success_rate_top1_2A", "success_rate_top2_2A", "success_rate_top3_2A", "n_poses", "best_pose_rmsd"]
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def evaluate_complex(
    pdb_id: str,
    decoys_dir: Path,
    coreset_dir: Path,
    model,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    mol2_path = decoys_dir / f"{pdb_id}_decoys.mol2"
    rmsd_path = decoys_dir / f"{pdb_id}_rmsd.dat"
    protein_pdb = coreset_dir / pdb_id / f"{pdb_id}_protein.pdb"
    if not (mol2_path.exists() and rmsd_path.exists() and protein_pdb.exists()):
        return None

    rmsd_index = load_rmsd_index(rmsd_path)
    blocks = split_mol2_blocks(mol2_path)

    scored_poses = []
    for block in blocks:
        name = pose_name_from_block(block)
        if name is None or name not in rmsd_index:
            continue
        mol = Chem.MolFromMol2Block(block, removeHs=False, sanitize=True)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        try:
            ligand_graph = mol_to_ligand_graph(mol)
            protein_graph = build_protein_pocket_graph(protein_pdb, ligand_graph.pos.numpy())
            ligand_batch = Batch.from_data_list([ligand_graph]).to(device)
            protein_batch = Batch.from_data_list([protein_graph]).to(device)
            with torch.no_grad():
                predicted = float(model(ligand_batch, protein_batch).item())
        except Exception:
            continue
        scored_poses.append((name, predicted, rmsd_index[name]))

    if not scored_poses:
        return None

    # Higher predicted affinity = better binding in this model's convention.
    ranked = sorted(scored_poses, key=lambda item: item[1], reverse=True)
    rmsds_in_rank_order = [item[2] for item in ranked]

    def success_at(top_n: int, cutoff: float) -> bool:
        return min(rmsds_in_rank_order[:top_n]) <= cutoff

    return {
        "pdb_id": pdb_id,
        "success_rate_top1_2A": int(success_at(1, 2.0)),
        "success_rate_top2_2A": int(success_at(2, 2.0)),
        "success_rate_top3_2A": int(success_at(3, 2.0)),
        "n_poses": len(scored_poses),
        "best_pose_rmsd": min(rmsds_in_rank_order),
    }


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")

    decoys_dir = Path(args.decoys_dir)
    coreset_dir = Path(args.coreset_dir)
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)

    pdb_ids = sorted({p.name.replace("_rmsd.dat", "") for p in decoys_dir.glob("*_rmsd.dat") if not p.name.startswith("._")})
    print(f"complexes with decoy pose data: {len(pdb_ids)}")

    done = already_done_complexes(RESULTS_CSV)
    todo = [pdb_id for pdb_id in pdb_ids if pdb_id not in done]
    print(f"already completed (resuming): {len(done)} | remaining: {len(todo)}")

    if not todo:
        print("Nothing left to do.")
        return

    model = load_model(checkpoint, device)
    write_header = not RESULTS_CSV.exists()

    start = time.time()
    for idx, pdb_id in enumerate(todo, start=1):
        result = evaluate_complex(pdb_id, decoys_dir, coreset_dir, model, device)
        if result is not None:
            append_result_rows(RESULTS_CSV, [result], write_header)
            write_header = False
        if idx % 10 == 0 or idx == len(todo):
            elapsed = time.time() - start
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - idx) / rate if rate > 0 else 0
            print(f"processed {idx}/{len(todo)} complexes | elapsed {elapsed:.0f}s | est. remaining {remaining:.0f}s")

    print(f"Done. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
