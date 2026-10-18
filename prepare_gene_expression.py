import csv
import json
from pathlib import Path

import numpy as np

EXPRESSION_CSV = Path("data") / "depmap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
MODEL_CSV = Path("data") / "depmap" / "Model.csv"
GDSC_CSV = Path("data") / "gdsc" / "GDSC2_fitted_dose_response.csv"
OUTPUT_DIR = Path("outputs") / "gene_expression"

TOP_N_VARIABLE_GENES = 2000


def load_model_to_sanger_id() -> dict:
    mapping = {}
    with MODEL_CSV.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("SangerModelID"):
                mapping[row["ModelID"]] = row["SangerModelID"]
    return mapping


def load_gdsc_sanger_ids() -> set:
    ids = set()
    with GDSC_CSV.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("SANGER_MODEL_ID"):
                ids.add(row["SANGER_MODEL_ID"])
    return ids


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_to_sanger = load_model_to_sanger_id()
    gdsc_ids = load_gdsc_sanger_ids()
    print(f"GDSC cell lines (SangerModelID): {len(gdsc_ids)}")
    print(f"DepMap models with SangerModelID: {len(model_to_sanger)}")

    with EXPRESSION_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        gene_cols = header[1:]
        print(f"genes in expression matrix: {len(gene_cols)}")

        model_ids = []
        rows = []
        for row in reader:
            model_id = row[0]
            sanger_id = model_to_sanger.get(model_id)
            if sanger_id is None or sanger_id not in gdsc_ids:
                continue
            model_ids.append(sanger_id)
            rows.append(row[1:])

    print(f"cell lines kept (overlap with GDSC): {len(model_ids)}")

    matrix = np.array(rows, dtype=np.float32)
    print(f"raw matrix shape: {matrix.shape}")

    # Variance filtering: keep the most variable genes. The scale (thousands of
    # genes vs. tens of thousands) is a standard dimensionality-reduction step
    # before an autoencoder on bulk expression data.
    gene_variance = matrix.var(axis=0)
    top_gene_idx = np.argsort(gene_variance)[::-1][:TOP_N_VARIABLE_GENES]
    top_gene_idx = np.sort(top_gene_idx)
    filtered = matrix[:, top_gene_idx]
    filtered_gene_names = [gene_cols[i] for i in top_gene_idx]
    print(f"after variance filtering: {filtered.shape}")

    # Z-score normalization per gene across cell lines.
    mean = filtered.mean(axis=0, keepdims=True)
    std = filtered.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    normalized = (filtered - mean) / std

    np.save(OUTPUT_DIR / "preprocessed_expression.npy", normalized.astype(np.float32))
    (OUTPUT_DIR / "cell_line_sanger_ids.json").write_text(json.dumps(model_ids), encoding="utf-8")
    (OUTPUT_DIR / "selected_gene_names.json").write_text(json.dumps(filtered_gene_names), encoding="utf-8")
    np.save(OUTPUT_DIR / "gene_mean.npy", mean.astype(np.float32))
    np.save(OUTPUT_DIR / "gene_std.npy", std.astype(np.float32))

    summary = {
        "raw_genes": len(gene_cols),
        "selected_genes": TOP_N_VARIABLE_GENES,
        "cell_lines": len(model_ids),
        "gdsc_cell_lines_total": len(gdsc_ids),
        "coverage_pct": 100.0 * len(model_ids) / len(gdsc_ids),
    }
    (OUTPUT_DIR / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
