import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from train_gene_autoencoder import StackedAutoencoder

EXPRESSION_CSV = Path("data") / "depmap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
MODEL_CSV = Path("data") / "depmap" / "Model.csv"
DRUGCOMB_CSV = Path("data") / "drugcomb" / "summary_table_v1.4.csv"

AE_DIR = Path("outputs") / "gene_autoencoder"
EXPR_DIR = Path("outputs") / "gene_expression"
OUTPUT_DIR = Path("outputs") / "cdps_gene_features"


def norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def load_model_to_sanger() -> dict:
    """ModelID -> SangerModelID, for every DepMap model that has one."""
    mapping = {}
    with MODEL_CSV.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("SangerModelID"):
                mapping[row["ModelID"]] = row["SangerModelID"]
    return mapping


def load_drugcomb_name_to_sanger(model_to_sanger: dict) -> dict:
    """norm(StrippedCellLineName) -> SangerModelID, to join DrugComb's cell_line_name."""
    stripped_to_sanger = {}
    with MODEL_CSV.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sanger_id = model_to_sanger.get(row["ModelID"])
            if sanger_id:
                stripped_to_sanger[norm(row.get("StrippedCellLineName", ""))] = sanger_id
    return stripped_to_sanger


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the EXACT gene selection + normalization statistics fit in Phase A
    # (prepare_gene_expression.py) so this is a frozen feature extractor applied
    # to a wider cell-line universe, not a re-fit (no new leakage, no retraining).
    selected_gene_names = json.loads((EXPR_DIR / "selected_gene_names.json").read_text())
    gene_mean = np.load(EXPR_DIR / "gene_mean.npy")
    gene_std = np.load(EXPR_DIR / "gene_std.npy")

    ae_checkpoint = torch.load(AE_DIR / "gene_autoencoder.pt", map_location="cpu", weights_only=False)
    autoencoder = StackedAutoencoder(input_dim=ae_checkpoint["input_dim"], latent_dim=ae_checkpoint["latent_dim"])
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
    autoencoder.eval()

    model_to_sanger = load_model_to_sanger()
    drugcomb_stripped_to_sanger = load_drugcomb_name_to_sanger(model_to_sanger)
    print(f"DepMap models with SangerModelID: {len(model_to_sanger)}")

    with EXPRESSION_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        gene_col_idx = {name: idx for idx, name in enumerate(header[1:])}
        missing = [g for g in selected_gene_names if g not in gene_col_idx]
        if missing:
            raise RuntimeError(f"{len(missing)} selected genes not found in expression header, e.g. {missing[:5]}")
        selected_idx = [gene_col_idx[g] for g in selected_gene_names]

        sanger_ids, feature_rows = [], []
        for row in reader:
            model_id = row[0]
            sanger_id = model_to_sanger.get(model_id)
            if sanger_id is None:
                continue
            values = row[1:]
            try:
                feature_rows.append([float(values[i]) for i in selected_idx])
            except (ValueError, IndexError):
                continue
            sanger_ids.append(sanger_id)

    matrix = np.array(feature_rows, dtype=np.float32)
    print(f"DepMap cell lines with expression + SangerModelID: {matrix.shape}")

    normalized = (matrix - gene_mean) / gene_std
    with torch.no_grad():
        latents = autoencoder.encoder(torch.tensor(normalized, dtype=torch.float32)).numpy()

    sanger_to_embedding = {sid: latents[i].tolist() for i, sid in enumerate(sanger_ids)}
    (OUTPUT_DIR / "sanger_to_embedding.json").write_text(json.dumps(sanger_to_embedding), encoding="utf-8")
    (OUTPUT_DIR / "drugcomb_stripped_to_sanger.json").write_text(json.dumps(drugcomb_stripped_to_sanger), encoding="utf-8")

    # Coverage check against the actual DrugComb file, for an honest number in the summary.
    total_rows, covered_rows, dc_names = 0, 0, set()
    with DRUGCOMB_CSV.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            dc_names.add(row.get("cell_line_name"))
            sanger_id = drugcomb_stripped_to_sanger.get(norm(row.get("cell_line_name")))
            if sanger_id and sanger_id in sanger_to_embedding:
                covered_rows += 1

    summary = {
        "num_cell_lines_with_embedding": len(sanger_to_embedding),
        "latent_dim": ae_checkpoint["latent_dim"],
        "drugcomb_total_rows": total_rows,
        "drugcomb_rows_with_real_transcriptomics": covered_rows,
        "drugcomb_coverage_pct": 100.0 * covered_rows / max(total_rows, 1),
        "drugcomb_unique_cell_lines": len(dc_names),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
