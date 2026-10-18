import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import Batch

from infer_affinity import build_graph_pair, load_model


COMPLEX_FEATURE_DIR_NAME = "coreset"
DEFAULT_CORESET_DAT = Path("data") / "casf2016_meta" / "CoreSet.dat"
DEFAULT_CASF_ROOT = Path("data") / "CASF-2016"
DEFAULT_CHECKPOINT = Path("outputs") / "best_protliggnn.pt"
OUTPUT_DIR = Path("outputs") / "casf2016_benchmark"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ProtLigGNN scoring power and ranking power on the official CASF-2016 core set."
    )
    parser.add_argument("--coreset_dat", type=str, default=str(DEFAULT_CORESET_DAT))
    parser.add_argument("--casf_root", type=str, default=str(DEFAULT_CASF_ROOT))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min_cluster_size", type=int, default=3)
    return parser.parse_args()


def load_coreset_index(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip() and not line.startswith("#code")]
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        rows.append(
            {
                "pdb_id": parts[0].lower(),
                "resolution": parts[1],
                "year": int(parts[2]),
                "logKa": float(parts[3]),
                "Ka": parts[4],
                "cluster": int(parts[5]),
            }
        )
    return rows


def find_complex_files(casf_root: Path, pdb_id: str) -> Optional[Tuple[Path, List[Path]]]:
    complex_dir = casf_root / COMPLEX_FEATURE_DIR_NAME / pdb_id
    protein_path = complex_dir / f"{pdb_id}_protein.pdb"
    if not protein_path.exists():
        return None
    # CASF-2016 ships both .sdf and .mol2 per ligand; some .sdf files fail RDKit
    # sanitization (unusual valences), so try .sdf first and fall back to .mol2.
    ligand_candidates = [
        complex_dir / f"{pdb_id}_ligand.sdf",
        complex_dir / f"{pdb_id}_ligand.mol2",
    ]
    ligand_candidates = [path for path in ligand_candidates if path.exists()]
    if not ligand_candidates:
        return None
    return protein_path, ligand_candidates


def predict_affinity(model, protein_pdb: Path, ligand_file: Path, device: torch.device) -> float:
    ligand_graph, protein_graph = build_graph_pair(protein_pdb, ligand_file)
    ligand_batch = Batch.from_data_list([ligand_graph]).to(device)
    protein_batch = Batch.from_data_list([protein_graph]).to(device)
    with torch.no_grad():
        return float(model(ligand_batch, protein_batch).item())


def compute_scoring_power(rows: List[Dict[str, object]]) -> Dict[str, float]:
    true_vals = [float(r["logKa"]) for r in rows]
    pred_vals = [float(r["predicted_affinity"]) for r in rows]
    pcc, _ = pearsonr(true_vals, pred_vals)
    spearman, _ = spearmanr(true_vals, pred_vals)
    rmse = (sum((t - p) ** 2 for t, p in zip(true_vals, pred_vals)) / len(true_vals)) ** 0.5
    mae = sum(abs(t - p) for t, p in zip(true_vals, pred_vals)) / len(true_vals)
    return {"pcc": pcc, "spearman": spearman, "rmse": rmse, "mae": mae, "n": len(rows)}


def compute_ranking_power(rows: List[Dict[str, object]], min_cluster_size: int) -> Dict[str, object]:
    clusters: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        clusters[int(row["cluster"])].append(row)

    cluster_spearmans = []
    cluster_details = []
    for cluster_id, members in sorted(clusters.items()):
        if len(members) < min_cluster_size:
            continue
        true_vals = [float(m["logKa"]) for m in members]
        pred_vals = [float(m["predicted_affinity"]) for m in members]
        if len(set(pred_vals)) < 2:
            continue
        rho, _ = spearmanr(true_vals, pred_vals)
        cluster_spearmans.append(rho)
        cluster_details.append({"cluster": cluster_id, "n": len(members), "spearman": rho})

    mean_spearman = statistics.mean(cluster_spearmans) if cluster_spearmans else None
    return {
        "mean_spearman": mean_spearman,
        "clusters_evaluated": len(cluster_spearmans),
        "clusters_total": len(clusters),
        "details": cluster_details,
    }


def main():
    args = parse_args()
    coreset_dat = Path(args.coreset_dat)
    casf_root = Path(args.casf_root)
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)

    if not coreset_dat.exists():
        raise FileNotFoundError(f"CASF-2016 CoreSet.dat not found: {coreset_dat}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = load_coreset_index(coreset_dat)
    print(f"official CASF-2016 core set size: {len(index_rows)}")

    model = load_model(checkpoint, device)

    scored_rows = []
    missing = []
    failed = []
    for idx, row in enumerate(index_rows, start=1):
        pdb_id = str(row["pdb_id"])
        files = find_complex_files(casf_root, pdb_id)
        if files is None:
            missing.append(pdb_id)
            continue
        protein_pdb, ligand_candidates = files
        predicted = None
        last_error = None
        used_ligand_file = None
        for ligand_file in ligand_candidates:
            try:
                predicted = predict_affinity(model, protein_pdb, ligand_file, device)
                used_ligand_file = ligand_file
                break
            except Exception as exc:
                last_error = exc
                continue
        if predicted is None:
            failed.append((pdb_id, str(last_error)))
            continue
        scored_row = dict(row)
        scored_row["predicted_affinity"] = predicted
        scored_row["ligand_file_used"] = used_ligand_file.name if used_ligand_file else ""
        scored_rows.append(scored_row)
        if idx % 25 == 0 or idx == len(index_rows):
            print(f"processed {idx}/{len(index_rows)} core set entries")

    print(f"scored: {len(scored_rows)} | missing locally: {len(missing)} | failed: {len(failed)}")

    fieldnames = ["pdb_id", "cluster", "year", "logKa", "predicted_affinity", "ligand_file_used"]
    with (OUTPUT_DIR / "casf2016_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored_rows)

    if missing:
        (OUTPUT_DIR / "missing_complexes.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    if failed:
        (OUTPUT_DIR / "failed_complexes.txt").write_text(
            "\n".join(f"{pdb_id}: {reason}" for pdb_id, reason in failed) + "\n", encoding="utf-8"
        )

    if not scored_rows:
        print("No complexes were scored; cannot compute scoring/ranking power.")
        return

    scoring = compute_scoring_power(scored_rows)
    ranking = compute_ranking_power(scored_rows, args.min_cluster_size)

    summary_lines = [
        "CASF-2016 Core Set Benchmark",
        "=============================",
        f"Checkpoint: {checkpoint}",
        f"Official core set size: {len(index_rows)}",
        f"Locally available and scored: {len(scored_rows)} ({100.0 * len(scored_rows) / len(index_rows):.1f}% coverage)",
        f"Missing locally: {len(missing)}",
        f"Failed to score: {len(failed)}",
        "",
        "Scoring Power (Pearson correlation on predicted vs. experimental logKa)",
        f"  PCC: {scoring['pcc']:.4f}",
        f"  Spearman: {scoring['spearman']:.4f}",
        f"  RMSE: {scoring['rmse']:.4f}",
        f"  MAE: {scoring['mae']:.4f}",
        f"  N: {scoring['n']}",
        "",
        "Ranking Power (mean within-cluster Spearman correlation)",
        f"  Mean Spearman: {ranking['mean_spearman']:.4f}" if ranking["mean_spearman"] is not None else "  Mean Spearman: N/A (no clusters with enough members)",
        f"  Clusters evaluated: {ranking['clusters_evaluated']} / {ranking['clusters_total']} (min cluster size: {args.min_cluster_size})",
        "",
        "Note: coverage is partial relative to the official 285-complex core set based on what",
        "was available in this environment's local PDBbind/CASF-2016 copy. Docking power and",
        "screening power (which require the separate decoy pose and screening library packages)",
        "are evaluated by docking_power_benchmark.py / screening_power_benchmark.py if present.",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    (OUTPUT_DIR / "casf2016_summary.txt").write_text(summary_text, encoding="utf-8")

    with (OUTPUT_DIR / "ranking_power_clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cluster", "n", "spearman"])
        writer.writeheader()
        writer.writerows(ranking["details"])

    print(summary_text)
    print(f"Saved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
