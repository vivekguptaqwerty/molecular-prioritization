import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

EXPR_DIR = Path("outputs") / "gene_expression"
OUTPUT_DIR = Path("outputs") / "gene_modules"

NUM_MODULES = 32


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = np.load(EXPR_DIR / "preprocessed_expression.npy")  # [cell_lines, 2000 genes], z-scored
    sanger_ids = json.loads((EXPR_DIR / "cell_line_sanger_ids.json").read_text())
    print(f"expression matrix: {matrix.shape}")

    # Cluster genes (not cell lines) by their expression profile across cell lines --
    # genes that co-vary together form a "module". Each module's mean expression per
    # cell line becomes one real, biologically-grounded attention token, instead of an
    # arbitrary reshape of the autoencoder's abstract bottleneck vector.
    gene_profiles = matrix.T  # [2000 genes, cell_lines]
    kmeans = KMeans(n_clusters=NUM_MODULES, random_state=42, n_init=10)
    gene_cluster_labels = kmeans.fit_predict(gene_profiles)

    module_scores = np.zeros((matrix.shape[0], NUM_MODULES), dtype=np.float32)
    module_sizes = []
    for module_id in range(NUM_MODULES):
        gene_mask = gene_cluster_labels == module_id
        module_sizes.append(int(gene_mask.sum()))
        if gene_mask.sum() > 0:
            module_scores[:, module_id] = matrix[:, gene_mask].mean(axis=1)

    print(f"module sizes (genes per module): min={min(module_sizes)} max={max(module_sizes)} mean={np.mean(module_sizes):.1f}")

    cell_line_to_modules = {sid: module_scores[i].tolist() for i, sid in enumerate(sanger_ids)}
    (OUTPUT_DIR / "cell_line_gene_modules.json").write_text(json.dumps(cell_line_to_modules), encoding="utf-8")
    (OUTPUT_DIR / "module_sizes.json").write_text(json.dumps(module_sizes), encoding="utf-8")

    summary = {"num_modules": NUM_MODULES, "num_cell_lines": matrix.shape[0],
               "min_module_size": min(module_sizes), "max_module_size": max(module_sizes)}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
