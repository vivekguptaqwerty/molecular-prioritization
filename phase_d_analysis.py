"""Phase D: ΔG reporting, synthetic accessibility, and GAPS prioritization precision/recall.

Three independent analyses, each reusing already-computed artifacts rather than
re-deriving them:

1. Convert existing pKd/pAffinity predictions (known PDBbind top-10 and the full
   CASF-2016 core set) into Gibbs free energy (ΔG, kcal/mol) via ΔG = -RT ln(10) * pKd.
2. Add RDKit's Ertl Synthetic Accessibility (SA) score to every generated molecule
   already scored by ProtLigGNN, and recompute GAPS across the full pool (not
   just the saved top 10) so ranking-based precision/recall is meaningful.
3. Define an objective, GAPS-independent "hit" criterion and measure precision@k /
   recall@k for GAPS ranking against it, honestly (no reverse-engineering GAPS to
   match the ground truth).
"""
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDKIT_CONTRIB = None
try:
    import rdkit as _rdkit
    RDKIT_CONTRIB = Path(_rdkit.__file__).parent / "Contrib" / "SA_Score"
except ImportError:
    pass
if RDKIT_CONTRIB and RDKIT_CONTRIB.exists():
    sys.path.append(str(RDKIT_CONTRIB))
import sascorer  # noqa: E402

GENAI_DIR = Path("outputs") / "genai"
SCORED_CSV = GENAI_DIR / "generated_molecules_scored.csv"
OUTPUT_DIR = Path("outputs") / "phase_d_analysis"

CASF_PREDICTIONS_CSV = Path("outputs") / "casf2016_benchmark" / "casf2016_predictions.csv"
KNOWN_TOP10_CSV = Path("outputs") / "vina_validation" / "known_top10" / "known_top10_docking_results.csv"

RT_LN10 = 1.3635  # kcal/mol, R=1.987e-3 kcal/mol/K, T=298.15K, RT*ln(10)

# Ranking cutoffs for precision@k / recall@k.
K_VALUES = [10, 25, 50, 100, 200, 500]


def pkd_to_dg(pkd: float) -> float:
    """pKd/pAffinity (-log10 Kd) -> Gibbs free energy of binding (kcal/mol)."""
    return -RT_LN10 * pkd


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom > 0 else 0.0


def rmse(xs: Sequence[float], ys: Sequence[float]) -> float:
    return (sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)) ** 0.5


def mae(xs: Sequence[float], ys: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


# --- 1. Delta-G conversion -------------------------------------------------

def delta_g_report():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full CASF-2016 core set (285 complexes): pKd-based scoring power, re-expressed in kcal/mol.
    casf_rows = list(csv.DictReader(CASF_PREDICTIONS_CSV.open("r", newline="", encoding="utf-8")))
    casf_out = []
    pred_dg, true_dg = [], []
    for row in casf_rows:
        pred_pkd = float(row["predicted_affinity"])
        true_pkd = float(row["logKa"])
        dg_pred = pkd_to_dg(pred_pkd)
        dg_true = pkd_to_dg(true_pkd)
        pred_dg.append(dg_pred)
        true_dg.append(dg_true)
        casf_out.append({**row, "predicted_dg_kcal_mol": dg_pred, "true_dg_kcal_mol": dg_true})

    with (OUTPUT_DIR / "casf2016_delta_g.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(casf_out[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(casf_out)

    casf_dg_metrics = {
        "n": len(casf_rows),
        "pcc": pearson(true_dg, pred_dg),
        "rmse_kcal_mol": rmse(true_dg, pred_dg),
        "mae_kcal_mol": mae(true_dg, pred_dg),
        "true_dg_range_kcal_mol": [min(true_dg), max(true_dg)],
        "pred_dg_range_kcal_mol": [min(pred_dg), max(pred_dg)],
    }

    # Known top-10 (already has Vina docking scores) -> add delta-G columns.
    known_rows = list(csv.DictReader(KNOWN_TOP10_CSV.open("r", newline="", encoding="utf-8")))
    known_out = []
    for row in known_rows:
        pred_pkd = float(row["predicted_affinity"])
        true_pkd = float(row["true_affinity"])
        known_out.append({
            **row,
            "predicted_dg_kcal_mol": round(pkd_to_dg(pred_pkd), 3),
            "true_dg_kcal_mol": round(pkd_to_dg(true_pkd), 3),
        })

    with (OUTPUT_DIR / "known_top10_delta_g.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(known_out[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(known_out)

    # pKd (ProtLigGNN) vs Vina score correlation on this same known set.
    vina_pairs = [(float(r["predicted_affinity"]), float(r["vina_score"])) for r in known_rows if r.get("vina_score")]
    pkd_vina_pcc = pearson([p for p, _ in vina_pairs], [v for _, v in vina_pairs]) if len(vina_pairs) > 1 else None

    summary = {
        "casf2016_delta_g_metrics": casf_dg_metrics,
        "rt_ln10_kcal_mol": RT_LN10,
        "conversion_formula": "delta_G = -RT*ln(10) * pKd  (T=298.15K, R=1.987e-3 kcal/mol/K)",
        "known_top10_pkd_vs_vina_pcc": pkd_vina_pcc,
        "known_top10_pkd_vs_vina_n": len(vina_pairs),
        "note": (
            "pKd-vs-Vina correlation sample size is small (top-10 only); treat as a "
            "qualitative consistency check, not a robust correlation estimate."
        ),
    }
    (OUTPUT_DIR / "delta_g_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Scatter plot: predicted vs true dG across the full CASF-2016 core set.
    plt.figure(figsize=(6, 6))
    plt.scatter(true_dg, pred_dg, alpha=0.4, s=16)
    lo, hi = min(true_dg + pred_dg), max(true_dg + pred_dg)
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("True ΔG (kcal/mol)")
    plt.ylabel("Predicted ΔG (kcal/mol)")
    plt.title(f"CASF-2016 Core Set: ΔG Scoring Power (n={len(casf_rows)}, PCC={casf_dg_metrics['pcc']:.3f})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "casf2016_delta_g_scatter.png", dpi=150)
    plt.close()

    print(f"CASF-2016 ΔG metrics: PCC={casf_dg_metrics['pcc']:.4f} RMSE={casf_dg_metrics['rmse_kcal_mol']:.3f} kcal/mol "
          f"MAE={casf_dg_metrics['mae_kcal_mol']:.3f} kcal/mol (n={casf_dg_metrics['n']})")
    print(f"known top-10 pKd vs Vina PCC: {pkd_vina_pcc}")
    return summary


# --- 2 & 3. SA score, full-pool GAPS, and prioritization precision/recall --

def load_scored_pool() -> List[Dict[str, object]]:
    with SCORED_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_sa_scores(rows: List[Dict[str, object]]):
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        row["sa_score"] = sascorer.calculateScore(mol) if mol is not None else None


def min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def compute_diversity_scores(rows: Sequence[Dict[str, object]]) -> List[float]:
    fingerprints = []
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol else None)

    scores = []
    for idx, fp in enumerate(fingerprints):
        if fp is None:
            scores.append(0.0)
            continue
        others = [f for j, f in enumerate(fingerprints) if j != idx and f is not None]
        if not others:
            scores.append(1.0)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, others)
        scores.append(1.0 - max(sims))
    return scores


def add_gaps_scores(rows: List[Dict[str, object]]):
    affinities = [float(r["predicted_affinity"]) for r in rows]
    qeds = [float(r["qed"]) for r in rows]
    norm_affinity = min_max_normalize(affinities)
    norm_qed = min_max_normalize(qeds)
    diversity = compute_diversity_scores(rows)
    for idx, row in enumerate(rows):
        novelty = 1.0 if str(row["is_novel"]).lower() == "true" else 0.0
        row["gaps"] = (
            0.50 * norm_affinity[idx] + 0.20 * novelty + 0.15 * norm_qed[idx] + 0.15 * diversity[idx]
        )


def define_hits(rows: List[Dict[str, object]]) -> List[bool]:
    """Objective, GAPS-independent 'drug-like high-affinity hit' criterion.

    high affinity: top quartile of predicted ProtLigGNN affinity in this pool
    + zero Lipinski violations + QED >= 0.5 + SA score <= 6 (easier end of the
    1-10 Ertl scale). None of these inputs come from the GAPS formula itself,
    so measuring GAPS's precision/recall against them is not circular.
    """
    affinities = sorted(float(r["predicted_affinity"]) for r in rows)
    affinity_threshold = affinities[int(0.75 * (len(affinities) - 1))]
    hits = []
    for row in rows:
        is_hit = (
            float(row["predicted_affinity"]) >= affinity_threshold
            and int(float(row["lipinski_violations"])) == 0
            and float(row["qed"]) >= 0.5
            and row["sa_score"] is not None
            and row["sa_score"] <= 6.0
        )
        hits.append(is_hit)
    return hits, affinity_threshold


def precision_recall_at_k(ranked_hits: List[bool], total_hits: int) -> Dict[int, Dict[str, float]]:
    results = {}
    cumulative = 0
    for k in K_VALUES:
        if k > len(ranked_hits):
            continue
        cumulative = sum(ranked_hits[:k])
        precision = cumulative / k
        recall = cumulative / total_hits if total_hits else 0.0
        results[k] = {"precision": precision, "recall": recall, "hits_in_top_k": cumulative}
    return results


def gaps_prioritization_report():
    rows = load_scored_pool()
    add_sa_scores(rows)
    rows = [r for r in rows if r["sa_score"] is not None]
    add_gaps_scores(rows)

    hits, affinity_threshold = define_hits(rows)
    total_hits = sum(hits)
    hit_rate = total_hits / len(rows)

    for row, is_hit in zip(rows, hits):
        row["is_hit"] = is_hit

    with (OUTPUT_DIR / "generated_molecules_full_scored.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["smiles", "predicted_affinity", "qed", "sa_score", "mol_weight", "logp", "tpsa",
                      "hbd", "hba", "lipinski_violations", "is_novel", "gaps", "is_hit"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Rank by GAPS, and as a baseline, rank by raw affinity alone.
    by_gaps = sorted(zip(rows, hits), key=lambda pair: float(pair[0]["gaps"]), reverse=True)
    by_affinity = sorted(zip(rows, hits), key=lambda pair: float(pair[0]["predicted_affinity"]), reverse=True)

    gaps_hits_ranked = [h for _, h in by_gaps]
    affinity_hits_ranked = [h for _, h in by_affinity]

    gaps_pr = precision_recall_at_k(gaps_hits_ranked, total_hits)
    affinity_pr = precision_recall_at_k(affinity_hits_ranked, total_hits)

    summary = {
        "pool_size": len(rows),
        "hit_definition": (
            f"predicted_affinity >= {affinity_threshold:.4f} (top quartile of pool) "
            "AND lipinski_violations == 0 AND qed >= 0.5 AND sa_score <= 6.0"
        ),
        "total_hits": total_hits,
        "hit_rate_pct": round(100 * hit_rate, 2),
        "gaps_ranking_precision_recall": gaps_pr,
        "affinity_only_ranking_precision_recall": affinity_pr,
    }
    (OUTPUT_DIR / "gaps_prioritization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (OUTPUT_DIR / "gaps_prioritization_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# GAPS Drug-Prioritization Precision/Recall\n\n")
        handle.write(f"Pool size: {len(rows)} generated, RDKit-valid molecules, all scored by ProtLigGNN.\n\n")
        handle.write(f"**Hit definition (independent of GAPS):** {summary['hit_definition']}\n\n")
        handle.write(f"Total hits in pool: {total_hits} ({summary['hit_rate_pct']}%)\n\n")
        handle.write("| k | GAPS precision@k | GAPS recall@k | Affinity-only precision@k | Affinity-only recall@k |\n")
        handle.write("|---|---|---|---|---|\n")
        for k in K_VALUES:
            if k not in gaps_pr:
                continue
            g, a = gaps_pr[k], affinity_pr[k]
            handle.write(f"| {k} | {g['precision']:.3f} | {g['recall']:.3f} | {a['precision']:.3f} | {a['recall']:.3f} |\n")

    # Precision-recall curve plot.
    plt.figure(figsize=(6, 5))
    plt.plot([gaps_pr[k]["recall"] for k in gaps_pr], [gaps_pr[k]["precision"] for k in gaps_pr],
              marker="o", label="GAPS ranking")
    plt.plot([affinity_pr[k]["recall"] for k in affinity_pr], [affinity_pr[k]["precision"] for k in affinity_pr],
              marker="s", label="Affinity-only ranking")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Drug Prioritization: Precision/Recall vs. Objective Hit Criterion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gaps_precision_recall_curve.png", dpi=150)
    plt.close()

    print(json.dumps(summary, indent=2))
    return summary


def main():
    RDLogger.DisableLog("rdApp.*")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    delta_g_report()
    gaps_prioritization_report()
    print(f"saved all Phase D outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
