import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import RDLogger
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch.utils.data import DataLoader, Dataset

from protliggnn_train import LIGAND_FEATURE_DIM, LigandEncoder, BidirectionalCrossAttention
from pubchem_lookup import load_cache as load_pubchem_cache
from train_ic50_model import embed_drug_graph


DRUGCOMB_CSV = Path("data") / "drugcomb" / "summary_table_v1.4.csv"
OUTPUT_DIR = Path("outputs") / "cdps_model"
SYNERGY_COLUMN = "synergy_zip"

GENE_EMBEDDINGS = Path("outputs") / "cdps_gene_features" / "sanger_to_embedding.json"
DRUGCOMB_NAME_TO_SANGER = Path("outputs") / "cdps_gene_features" / "drugcomb_stripped_to_sanger.json"
GENE_TOKEN_DIM = 16
NUM_GENE_TOKENS = 128 // GENE_TOKEN_DIM  # 8 tokens of 16 dims each, same scheme as the fused IC50 model


def norm_cell_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def reshape_gene_tokens(latent: List[float]) -> torch.Tensor:
    arr = torch.tensor(latent, dtype=torch.float32)
    return arr.view(NUM_GENE_TOKENS, GENE_TOKEN_DIM)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Drug+Drug+CellLine GNN to predict synergy (DrugComb).")
    parser.add_argument("--drugcomb_csv", type=str, default=str(DRUGCOMB_CSV))
    parser.add_argument("--synergy_column", type=str, default=SYNERGY_COLUMN)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--run_name", type=str, default="cdps_gnn")
    parser.add_argument("--no_crossgraph", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


# DrugComb uses these as placeholders for "no second agent" (single-drug dose-
# response rows embedded in the combo table), not real compounds. A naive name
# resolver can accidentally match them to an unrelated real compound (e.g. "NA"
# resolving to the element sodium), silently turning single-agent rows into
# fake zero-synergy pairs.
_NON_DRUG_PLACEHOLDERS = {"", "NA", "N/A", "NONE", "DMSO", "CONTROL", "VEHICLE"}


def load_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            drug_row = (row.get("drug_row") or "").strip().upper()
            drug_col = (row.get("drug_col") or "").strip().upper()
            if drug_row in _NON_DRUG_PLACEHOLDERS or drug_col in _NON_DRUG_PLACEHOLDERS:
                continue
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def build_drug_graph_cache(drug_names: List[str], smiles_cache: Dict[str, Optional[str]]) -> Dict[str, Data]:
    graphs: Dict[str, Data] = {}
    skipped = 0
    for idx, name in enumerate(drug_names, start=1):
        smiles = smiles_cache.get(name)
        if not smiles:
            skipped += 1
            continue
        graph = embed_drug_graph(smiles)
        if graph is None:
            skipped += 1
            continue
        graphs[name] = graph
        if idx % 50 == 0 or idx == len(drug_names):
            print(f"embedded {idx}/{len(drug_names)} drugs ({skipped} skipped so far)")
    return graphs


class CDPSDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, str, str, float]], drug_graphs: Dict[str, Data],
                 gene_tokens: Dict[str, torch.Tensor]):
        self.samples = []
        for drug_row, drug_col, sanger_id, label in rows:
            graph_a = drug_graphs.get(drug_row)
            graph_b = drug_graphs.get(drug_col)
            tokens = gene_tokens.get(sanger_id)
            if graph_a is None or graph_b is None or tokens is None:
                continue
            self.samples.append((graph_a, graph_b, tokens, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_cdps(batch):
    graphs_a = [item[0] for item in batch]
    graphs_b = [item[1] for item in batch]
    gene_tokens = torch.stack([item[2] for item in batch])
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    return Batch.from_data_list(graphs_a), Batch.from_data_list(graphs_b), gene_tokens, labels


class CDPSModel(nn.Module):
    """Drug-row + Drug-col + real cell-line transcriptomics -> synergy.

    The cell line is represented by the frozen gene-expression autoencoder
    embedding (same foundation as the fused IC50 model), not an arbitrary
    learned per-cell-line index lookup.
    """

    def __init__(self, hidden_dim: int = 128, no_crossgraph: bool = False):
        super().__init__()
        self.no_crossgraph = no_crossgraph
        # Shared drug encoder: both drug_row and drug_col are generic small molecules.
        self.drug_encoder = LigandEncoder(LIGAND_FEATURE_DIM, hidden_dim)
        self.gene_token_proj = nn.Linear(GENE_TOKEN_DIM, hidden_dim)
        self.cross_attention = BidirectionalCrossAttention(hidden_dim=hidden_dim, num_heads=4)
        self.gene_pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def pool_drug(self, drug_batch: Batch) -> torch.Tensor:
        drug_x = self.drug_encoder(drug_batch.x, drug_batch.edge_index)
        return global_mean_pool(drug_x, drug_batch.batch)

    def forward(self, drug_batch_a: Batch, drug_batch_b: Batch, gene_tokens_batch: torch.Tensor):
        drug_x_a = self.drug_encoder(drug_batch_a.x, drug_batch_a.edge_index)
        drug_x_b = self.drug_encoder(drug_batch_b.x, drug_batch_b.edge_index)
        pool_a = global_mean_pool(drug_x_a, drug_batch_a.batch)
        pool_b = global_mean_pool(drug_x_b, drug_batch_b.batch)

        gene_x = self.gene_token_proj(gene_tokens_batch)  # [B, NUM_GENE_TOKENS, hidden_dim]
        batch_size = drug_batch_a.num_graphs
        gene_contexts = []
        for b in range(batch_size):
            mask_a = drug_batch_a.batch == b
            mask_b = drug_batch_b.batch == b
            # Both drugs in the pair attend jointly against this cell line's
            # transcriptome, so the synergy context reflects the pair, not either
            # drug alone.
            pair_nodes = torch.cat([drug_x_a[mask_a], drug_x_b[mask_b]], dim=0)
            gene_nodes = gene_x[b]
            if self.no_crossgraph:
                gene_ctx = gene_nodes
            else:
                _, gene_ctx = self.cross_attention(pair_nodes, gene_nodes)
            gene_contexts.append(gene_ctx.mean(dim=0))

        gene_context = self.gene_pool_proj(torch.stack(gene_contexts))
        # Symmetric drug-pair interaction term, kept from the original design.
        drug_interaction = pool_a * pool_b
        joint = self.dropout(torch.cat([pool_a, pool_b, drug_interaction, gene_context], dim=-1))
        return self.regressor(joint).squeeze(-1)


def split_indices(num_samples: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_size = max(1, int(0.8 * num_samples))
    val_size = max(1, int(0.1 * num_samples))
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]
    return train_idx, val_idx, test_idx


def evaluate(model, loader, device) -> Dict[str, object]:
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for drug_a, drug_b, gene_tokens_b, labels in loader:
            drug_a, drug_b = drug_a.to(device), drug_b.to(device)
            gene_tokens_b = gene_tokens_b.to(device)
            labels = labels.to(device)
            output = model(drug_a, drug_b, gene_tokens_b)
            loss = F.mse_loss(output, labels)
            total_loss += float(loss.item()) * len(labels)
            count += len(labels)
            preds.extend(output.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    pcc = pearsonr(trues, preds)[0] if len(set(preds)) > 1 else 0.0
    spearman = spearmanr(trues, preds)[0] if len(set(preds)) > 1 else 0.0
    rmse = (sum((t - p) ** 2 for t, p in zip(trues, preds)) / max(len(trues), 1)) ** 0.5
    mae = sum(abs(t - p) for t, p in zip(trues, preds)) / max(len(trues), 1)
    return {"loss": total_loss / max(count, 1), "pcc": pcc, "spearman": spearman, "rmse": rmse, "mae": mae, "preds": preds, "trues": trues}


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    set_seed(args.seed)
    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_path) if args.save_path else OUTPUT_DIR / "best_cdps_model.pt"

    print(f"loading DrugComb data from {args.drugcomb_csv}")
    # Read the full available file before subsampling: DrugComb rows are grouped
    # by experiment/study, so a short prefix read yields very poor drug diversity.
    rows = load_rows(Path(args.drugcomb_csv), max_rows=None)
    print(f"total rows available: {len(rows)}")

    smiles_cache = load_pubchem_cache()
    gene_embeddings_raw = json.loads(GENE_EMBEDDINGS.read_text())
    name_to_sanger = json.loads(DRUGCOMB_NAME_TO_SANGER.read_text())

    def resolve_sanger(cell_line_name: str) -> Optional[str]:
        sanger_id = name_to_sanger.get(norm_cell_name(cell_line_name))
        return sanger_id if sanger_id in gene_embeddings_raw else None

    # Filter to rows where both drugs already have a resolved SMILES AND the
    # cell line has a real transcriptomic embedding, BEFORE subsampling. The
    # drug-name and cell-line-name distributions both have long tails;
    # subsampling first and filtering after wastes almost the whole sample on
    # drugs/cell-lines nobody ever resolved.
    rows = [r for r in rows if smiles_cache.get(r["drug_row"]) and smiles_cache.get(r["drug_col"])
            and resolve_sanger(r["cell_line_name"])]
    print(f"rows with both drugs resolvable and cell line having real transcriptomics: {len(rows)}")

    random.Random(args.seed).shuffle(rows)
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"using {len(rows)} rows")

    unique_drugs = sorted({row["drug_row"] for row in rows} | {row["drug_col"] for row in rows})
    print(f"unique drugs in sample: {len(unique_drugs)}")
    drug_graphs = build_drug_graph_cache(unique_drugs, smiles_cache)
    print(f"successfully embedded drug graphs: {len(drug_graphs)}/{len(unique_drugs)}")

    cell_names = sorted({row["cell_line_name"] for row in rows})
    print(f"unique cell lines (all with real transcriptomics): {len(cell_names)}")
    gene_tokens = {sanger_id: reshape_gene_tokens(vec) for sanger_id, vec in gene_embeddings_raw.items()}

    prepared_rows = []
    skipped = 0
    for row in rows:
        drug_row, drug_col = row["drug_row"], row["drug_col"]
        if drug_row not in drug_graphs or drug_col not in drug_graphs:
            skipped += 1
            continue
        try:
            label = float(row[args.synergy_column])
        except (TypeError, ValueError, KeyError):
            skipped += 1
            continue
        prepared_rows.append((drug_row, drug_col, resolve_sanger(row["cell_line_name"]), label))
    print(f"usable rows: {len(prepared_rows)} (skipped {skipped})")

    train_idx, val_idx, test_idx = split_indices(len(prepared_rows), args.seed)
    train_rows = [prepared_rows[i] for i in train_idx]
    val_rows = [prepared_rows[i] for i in val_idx]
    test_rows = [prepared_rows[i] for i in test_idx]

    train_ds = CDPSDataset(train_rows, drug_graphs, gene_tokens)
    val_ds = CDPSDataset(val_rows, drug_graphs, gene_tokens)
    test_ds = CDPSDataset(test_rows, drug_graphs, gene_tokens)
    print(f"train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_cdps)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cdps)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cdps)

    model = CDPSModel(hidden_dim=args.hidden_dim, no_crossgraph=args.no_crossgraph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for drug_a, drug_b, gene_tokens_b, labels in train_loader:
            drug_a, drug_b = drug_a.to(device), drug_b.to(device)
            gene_tokens_b = gene_tokens_b.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(drug_a, drug_b, gene_tokens_b)
            loss = F.mse_loss(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(labels)
            train_count += len(labels)

        train_loss = train_loss_total / max(train_count, 1)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_PCC={val_metrics['pcc']:.4f} val_Spearman={val_metrics['spearman']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], "val_pcc": val_metrics["pcc"], "val_spearman": val_metrics["spearman"]})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "stats": {"epoch": epoch, **{k: v for k, v in val_metrics.items() if k not in ("preds", "trues")}},
                },
                save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Loaded best model from epoch {best_epoch}\n"
        f"Test loss={test_metrics['loss']:.4f} PCC={test_metrics['pcc']:.4f} "
        f"Spearman={test_metrics['spearman']:.4f} RMSE={test_metrics['rmse']:.4f} MAE={test_metrics['mae']:.4f}"
    )

    with (OUTPUT_DIR / f"epoch_history_{args.run_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_pcc", "val_spearman"])
        writer.writeheader()
        writer.writerows(history)

    with (OUTPUT_DIR / f"test_predictions_{args.run_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_synergy", "predicted_synergy"])
        for t, p in zip(test_metrics["trues"], test_metrics["preds"]):
            writer.writerow([t, p])

    plt.figure(figsize=(6, 6))
    plt.scatter(test_metrics["trues"], test_metrics["preds"], alpha=0.4, s=10)
    min_v = min(min(test_metrics["trues"]), min(test_metrics["preds"]))
    max_v = max(max(test_metrics["trues"]), max(test_metrics["preds"]))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel(f"True {args.synergy_column}")
    plt.ylabel(f"Predicted {args.synergy_column}")
    plt.title(f"CDPS Model ({args.run_name}): Predicted vs True")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"scatter_{args.run_name}.png", dpi=150)
    plt.close()

    summary = {
        "run_name": args.run_name,
        "synergy_column": args.synergy_column,
        "best_epoch": best_epoch,
        "test_loss": test_metrics["loss"],
        "test_pcc": test_metrics["pcc"],
        "test_spearman": test_metrics["spearman"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "num_samples_used": len(prepared_rows),
        "unique_drugs_embedded": len(drug_graphs),
        "unique_cell_lines": len(cell_names),
        "no_crossgraph": args.no_crossgraph,
    }
    (OUTPUT_DIR / f"summary_{args.run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
