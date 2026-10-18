import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence


OUTPUT_ROOT = Path("outputs")
FINAL_DIR = OUTPUT_ROOT / "final_research_package"

COMPARISON_5000 = OUTPUT_ROOT / "comparison_all_models_5000_e40.txt"
REPORT_RESULTS = OUTPUT_ROOT / "report_assets" / "results_table.csv"
TOP_KNOWN = OUTPUT_ROOT / "report_assets" / "top10_predicted_affinities.csv"
GENAI_SUMMARY = OUTPUT_ROOT / "genai" / "generated_molecule_quality_summary.json"
TOP_GENERATED_GAPS = OUTPUT_ROOT / "genai" / "top10_generated_candidates_by_gaps.csv"
SCORED_GENERATED = OUTPUT_ROOT / "genai" / "generated_molecules_scored.csv"
KNOWN_DOCKING = OUTPUT_ROOT / "vina_validation" / "known_top10" / "known_top10_docking_results.csv"
GENERATED_DOCKING = OUTPUT_ROOT / "vina_validation" / "generated_top10" / "generated_top10_docking_results.csv"
FINAL_DOCKING_MD = OUTPUT_ROOT / "vina_validation" / "final_docking_comparison.md"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_5000_comparison(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n(?=Run: )", text)
    rows = []
    for block in blocks:
        if not block.startswith("Run: "):
            continue
        row = {}
        patterns = {
            "run_name": r"Run:\s*(.+)",
            "best_epoch": r"Best epoch:\s*(.+)",
            "pcc": r"PCC:\s*(.+)",
            "spearman": r"Spearman:\s*(.+)",
            "rmse": r"RMSE:\s*(.+)",
            "mae": r"MAE:\s*(.+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, block)
            if match:
                row[key] = match.group(1).strip()
        if row:
            rows.append(row)
    return rows


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def best_model_row(rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    if not rows:
        return {}
    return max(rows, key=lambda row: safe_float(row.get("pcc"), float("-inf")))


def markdown_table(rows: Sequence[Dict[str, str]], columns: Sequence[str]) -> List[str]:
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(column, "")) for column in columns) + "|")
    return lines


def build_key_metrics(best: Dict[str, str], genai: Dict, generated_docking: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    best_generated = best_generated_by_vina(generated_docking)
    rows = []
    if best:
        rows.extend(
            [
                {"category": "ProtLigGNN", "metric": "best_run", "value": best.get("run_name", "")},
                {"category": "ProtLigGNN", "metric": "PCC", "value": best.get("pcc", "")},
                {"category": "ProtLigGNN", "metric": "Spearman", "value": best.get("spearman", "")},
                {"category": "ProtLigGNN", "metric": "RMSE", "value": best.get("rmse", "")},
                {"category": "ProtLigGNN", "metric": "MAE", "value": best.get("mae", "")},
            ]
        )
    for metric in [
        "generated_count",
        "valid_count",
        "unique_valid_count",
        "novel_unique_count",
        "validity_percentage",
        "uniqueness_percentage",
        "novelty_percentage",
    ]:
        if metric in genai:
            rows.append({"category": "GenAI", "metric": metric, "value": str(genai[metric])})
    if best_generated:
        rows.extend(
            [
                {"category": "Docking", "metric": "best_generated_vina_score", "value": best_generated.get("vina_score", "")},
                {"category": "Docking", "metric": "best_generated_smiles", "value": best_generated.get("smiles", "")},
            ]
        )
    return rows


def best_generated_by_vina(rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    scored = [row for row in rows if safe_float(row.get("vina_score")) is not None]
    if not scored:
        return {}
    return min(scored, key=lambda row: safe_float(row.get("vina_score"), 0.0))


def build_final_summary_md(
    comparison_rows: Sequence[Dict[str, str]],
    report_rows: Sequence[Dict[str, str]],
    known_rows: Sequence[Dict[str, str]],
    genai: Dict,
    generated_rows: Sequence[Dict[str, str]],
    known_docking: Sequence[Dict[str, str]],
    generated_docking: Sequence[Dict[str, str]],
) -> str:
    best = best_model_row(comparison_rows)
    best_generated = best_generated_by_vina(generated_docking)
    lines = [
        "# Final Results Summary",
        "",
        "## Project Scope",
        "",
        "ProtLigGNN predicts protein-ligand binding affinity from paired ligand molecular graphs and protein pocket graphs. The GenAI extension adds a closed-loop workflow: generated ligands are chemically validated with RDKit, prioritized by ProtLigGNN affinity and GAPS, then secondarily checked with AutoDock Vina docking.",
        "",
        "Novelty statement: the central novelty is the closed-loop GenAI + ProtLigGNN affinity-guided ligand prioritization framework, not RDKit or AutoDock Vina themselves.",
        "",
        "RDKit and Vina are validation and filtering tools in this project. They support chemical validity and secondary docking checks, but they are not claimed as the novel contribution.",
        "",
        "## ProtLigGNN Best Model Metrics",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Best run: {best.get('run_name')}",
                f"- PCC: {best.get('pcc')}",
                f"- Spearman: {best.get('spearman')}",
                f"- RMSE: {best.get('rmse')}",
                f"- MAE: {best.get('mae')}",
            ]
        )
    else:
        lines.append("- No ProtLigGNN comparison metrics found.")

    if comparison_rows:
        lines.extend(["", "## Ablation Comparison", ""])
        lines.extend(markdown_table(comparison_rows, ["run_name", "best_epoch", "pcc", "spearman", "rmse", "mae"]))

    if report_rows:
        lines.extend(["", "## Earlier Report-Asset Experiment Table", ""])
        lines.extend(markdown_table(report_rows, ["run_name", "best_epoch", "test_pcc", "test_spearman", "test_rmse", "test_mae"]))

    if known_rows:
        lines.extend(["", "## Top Known Binder Results", ""])
        lines.extend(markdown_table(known_rows[:10], ["rank", "pdb_id", "predicted_affinity", "true_affinity", "absolute_error"]))

    lines.extend(["", "## GenAI Generation Metrics", ""])
    if genai:
        for key in [
            "generated_count",
            "valid_count",
            "unique_valid_count",
            "novel_unique_count",
            "validity_percentage",
            "uniqueness_percentage",
            "novelty_percentage",
        ]:
            if key in genai:
                lines.append(f"- {key}: {genai[key]}")
    else:
        lines.append("- No GenAI quality summary found.")

    lines.extend(
        [
            "",
            "## GAPS Explanation",
            "",
            "GAPS means Generative Affinity Prioritization Score. In this project it combines normalized ProtLigGNN affinity, novelty, normalized QED, and diversity to rank generated molecules before docking validation.",
        ]
    )

    if generated_rows:
        lines.extend(["", "## Top Generated Candidates", ""])
        lines.extend(
            markdown_table(
                generated_rows[:10],
                ["smiles", "predicted_affinity", "gaps", "qed", "lipinski_violations"],
            )
        )

    if known_docking or generated_docking:
        lines.extend(["", "## Vina Docking Validation Results", ""])
        if known_docking:
            lines.extend(["### Known Top 10", ""])
            lines.extend(markdown_table(known_docking, ["rank", "pdb_id", "predicted_affinity", "true_affinity", "abs_error", "vina_score"]))
        if generated_docking:
            lines.extend(["", "### Generated Top 10", ""])
            lines.extend(
                markdown_table(
                    generated_docking,
                    ["rank", "smiles", "predicted_affinity", "gaps_score", "qed", "lipinski_violations", "vina_score"],
                )
            )

    if best_generated:
        lines.extend(
            [
                "",
                "## Best Generated Molecule Result",
                "",
                f"- SMILES: {best_generated.get('smiles')}",
                f"- ProtLigGNN predicted affinity: {best_generated.get('predicted_affinity')}",
                f"- GAPS: {best_generated.get('gaps_score')}",
                f"- QED: {best_generated.get('qed')}",
                f"- Lipinski violations: {best_generated.get('lipinski_violations')}",
                f"- Vina score: {best_generated.get('vina_score')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Output Evidence",
            "",
            "- ProtLigGNN comparison: outputs/comparison_all_models_5000_e40.txt",
            "- GenAI quality summary: outputs/genai/generated_molecule_quality_summary.json",
            "- Generated candidate ranking: outputs/genai/top10_generated_candidates_by_gaps.csv",
            "- Docking comparison: outputs/vina_validation/final_docking_comparison.md",
        ]
    )
    return "\n".join(lines) + "\n"


def strip_markdown(md: str) -> str:
    text = re.sub(r"^#+\s*", "", md, flags=re.MULTILINE)
    text = text.replace("|", " ")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text


def build_top_generated_summary(generated_rows: Sequence[Dict[str, str]], generated_docking: Sequence[Dict[str, str]]) -> str:
    docking_by_smiles = {row.get("smiles"): row for row in generated_docking}
    best = best_generated_by_vina(generated_docking)
    lines = [
        "# Top Generated Candidate Summary",
        "",
        "Generated candidates were ranked with GAPS before docking. GAPS combines normalized ProtLigGNN affinity, novelty, QED, and diversity.",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Best Docked Generated Candidate",
                "",
                f"- SMILES: {best.get('smiles')}",
                f"- Predicted affinity: {best.get('predicted_affinity')}",
                f"- GAPS: {best.get('gaps_score')}",
                f"- QED: {best.get('qed')}",
                f"- Lipinski violations: {best.get('lipinski_violations')}",
                f"- Vina score: {best.get('vina_score')}",
                "",
            ]
        )
    lines.extend(["## Top Candidates", ""])
    rows = []
    for row in generated_rows[:10]:
        docking = docking_by_smiles.get(row.get("smiles"), {})
        rows.append(
            {
                "smiles": row.get("smiles", ""),
                "predicted_affinity": row.get("predicted_affinity", ""),
                "gaps": row.get("gaps", ""),
                "qed": row.get("qed", ""),
                "lipinski_violations": row.get("lipinski_violations", ""),
                "vina_score": docking.get("vina_score", ""),
            }
        )
    lines.extend(markdown_table(rows, ["smiles", "predicted_affinity", "gaps", "qed", "lipinski_violations", "vina_score"]))
    return "\n".join(lines) + "\n"


def build_limitations() -> str:
    return "\n".join(
        [
            "# Limitations and Future Work",
            "",
            "## Limitations",
            "",
            "- ProtLigGNN performance is evaluated computationally; experimental binding validation is not included.",
            "- The SMILES LSTM is lightweight and character-level, so chemical syntax and scaffold quality depend heavily on RDKit filtering after generation.",
            "- AutoDock Vina is used as secondary computational validation, not as ground truth.",
            "- RDKit descriptors and validity checks are filtering tools and are not the novelty of the project.",
            "- Docking boxes are computational approximations and may not capture all induced-fit or solvent effects.",
            "",
            "## Future Work",
            "",
            "- Train stronger molecular generators such as graph-based or transformer-based models.",
            "- Add protein-conditioned generation rather than target-agnostic SMILES generation.",
            "- Add more rigorous docking box selection around known binding pockets.",
            "- Validate top generated candidates with molecular dynamics or experimental assays.",
            "- Expand the closed-loop GenAI + ProtLigGNN system with active learning from scored candidates.",
            "- Extend beyond single protein-ligand binding affinity (pKd) to drug-response prediction: a cell-line-conditioned GNN over drug and cell-line features predicting IC50, alongside a reinforcement-learning-based molecule generator (optimizing directly against the affinity/response predictor rather than supervised SMILES sampling) and a combined drug-pair sensitivity (CDPS) scoring scheme. This was part of the original project scope but was not implemented in the current codebase; this paper reports the completed ProtLigGNN + GenAI + docking pipeline only.",
        ]
    ) + "\n"


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = parse_5000_comparison(COMPARISON_5000)
    report_rows = read_csv(REPORT_RESULTS)
    known_rows = read_csv(TOP_KNOWN)
    genai_summary = read_json(GENAI_SUMMARY)
    generated_rows = read_csv(TOP_GENERATED_GAPS)
    known_docking = read_csv(KNOWN_DOCKING)
    generated_docking = read_csv(GENERATED_DOCKING)

    best = best_model_row(comparison_rows)
    key_metrics = build_key_metrics(best, genai_summary, generated_docking)
    write_csv(FINAL_DIR / "key_metrics_table.csv", key_metrics, ["category", "metric", "value"])
    key_metrics_md = "\n".join(markdown_table(key_metrics, ["category", "metric", "value"])) + "\n"
    write_text(FINAL_DIR / "key_metrics_table.md", key_metrics_md)

    final_md = build_final_summary_md(
        comparison_rows=comparison_rows,
        report_rows=report_rows,
        known_rows=known_rows,
        genai=genai_summary,
        generated_rows=generated_rows,
        known_docking=known_docking,
        generated_docking=generated_docking,
    )
    write_text(FINAL_DIR / "final_results_summary.md", final_md)
    write_text(FINAL_DIR / "final_results_summary.txt", strip_markdown(final_md))
    write_text(FINAL_DIR / "top_generated_candidate_summary.md", build_top_generated_summary(generated_rows, generated_docking))
    write_text(FINAL_DIR / "limitations_and_future_work.md", build_limitations())

    print(f"Created final research package: {FINAL_DIR}")
    for path in [
        "final_results_summary.md",
        "final_results_summary.txt",
        "key_metrics_table.csv",
        "key_metrics_table.md",
        "top_generated_candidate_summary.md",
        "limitations_and_future_work.md",
    ]:
        print(FINAL_DIR / path)


if __name__ == "__main__":
    main()
