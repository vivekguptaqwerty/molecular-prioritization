"""Extend the ProtLigGNN-pKd-vs-Vina-score comparison beyond the original top-10
known ligands, by docking a larger, affinity-stratified sample of CASF-2016 core-set
complexes that are available locally. This gives a statistically meaningful sample
size for the pKd-vs-Vina correlation requested for the paper, instead of n=10.
"""
import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rdkit import RDLogger

from dock_top_candidates import dock_existing_ligand, find_complex_files
from vina_validation import require_tool, resolve_tool

CASF_PREDICTIONS_CSV = Path("outputs") / "casf2016_benchmark" / "casf2016_predictions.csv"
EXISTING_KNOWN_CSV = Path("outputs") / "vina_validation" / "known_top10" / "known_top10_docking_results.csv"
OUT_DIR = Path("outputs") / "vina_validation" / "extended_sample"
SAMPLE_SIZE = 30
SEED = 42


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stratified_sample(rows: List[Dict[str, str]], exclude: set, n: int, seed: int) -> List[Dict[str, str]]:
    """Sample across the affinity range (not just the top), for a representative correlation."""
    candidates = [r for r in rows if r["pdb_id"].lower() not in exclude]
    candidates.sort(key=lambda r: float(r["logKa"]))
    rng = random.Random(seed)
    bucket_count = min(n, len(candidates))
    bucket_size = max(1, len(candidates) // bucket_count)
    sampled = []
    for i in range(bucket_count):
        bucket = candidates[i * bucket_size : (i + 1) * bucket_size] or candidates[-bucket_size:]
        sampled.append(rng.choice(bucket))
    return sampled[:n]


def already_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {row["pdb_id"].lower() for row in read_csv(path)}


def main():
    RDLogger.DisableLog("rdApp.*")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = OUT_DIR / "extended_sample_docking_results.csv"
    failures_path = OUT_DIR / "extended_sample_failures.log"

    obabel_command = resolve_tool("obabel", [Path("C:/Program Files/OpenBabel-2.4.1/obabel.exe")])
    vina_command = resolve_tool(
        "vina",
        [Path("bin/vina"), Path("C:/Program Files (x86)/The Scripps Research Institute/Vina/vina.exe")],
    )
    require_tool(obabel_command, "Open Babel", version_flag="-V")
    require_tool(vina_command, "AutoDock Vina")

    casf_rows = read_csv(CASF_PREDICTIONS_CSV)
    already_known = {r["pdb_id"].lower() for r in read_csv(EXISTING_KNOWN_CSV)}
    already_sampled = already_done(results_csv)
    exclude = already_known | already_sampled

    sample = stratified_sample(casf_rows, exclude, SAMPLE_SIZE - len(already_sampled), SEED)
    print(f"docking {len(sample)} additional complexes (already have {len(already_sampled)} from a prior partial run)")

    results = []
    if results_csv.exists():
        results = read_csv(results_csv)

    failures = []
    for row in sample:
        pdb_id = row["pdb_id"].lower()
        try:
            protein_pdb, ligand_file = find_complex_files(pdb_id)
            vina_score = dock_existing_ligand(
                obabel_command=obabel_command,
                vina_command=vina_command,
                protein_pdb=protein_pdb,
                ligand_file=ligand_file,
                label=pdb_id,
                out_dir=OUT_DIR,
            )
            print(f"{pdb_id}: predicted_affinity={row['predicted_affinity']} logKa={row['logKa']} vina={vina_score:.3f}")
            results.append({
                "pdb_id": pdb_id,
                "predicted_affinity": row["predicted_affinity"],
                "true_affinity": row["logKa"],
                "vina_score": vina_score,
            })
        except Exception as exc:
            message = f"{pdb_id} FAILED: {exc}"
            failures.append(message)
            print(message)

        write_csv(results_csv, results, ["pdb_id", "predicted_affinity", "true_affinity", "vina_score"])

    failures_path.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"done. {len(results)} total docked, {len(failures)} failures this run.")
    print(f"results: {results_csv}")


if __name__ == "__main__":
    main()
