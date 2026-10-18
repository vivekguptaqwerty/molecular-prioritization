import argparse
import csv
import json
import random
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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import DataLoader, Dataset

from protliggnn_train import LIGAND_FEATURE_DIM, LigandEncoder, BidirectionalCrossAttention
from pubchem_lookup import load_cache as load_pubchem_cache
from train_ic50_model import embed_drug_graph

GDSC_CSV = Path("data") / "gdsc" / "GDSC2_fitted_dose_response.csv"
GENE_EMBEDDINGS = Path("outputs") / "gene_autoencoder" / "cell_line_embeddings.json"
GENE_MODULES = Path("outputs") / "gene_modules" / "cell_line_gene_modules.json"
OUTPUT_DIR = Path("outputs") / "ic50_fused_model"

GENE_TOKEN_DIM = 16
NUM_GENE_TOKENS = 128 // GENE_TOKEN_DIM  # 8 tokens of 16 dims each, reshaped from the 128-dim latent
NUM_GENE_MODULES = 32  # must match build_gene_modules.py


def parse_args():
    parser = argparse.ArgumentParser(description="Fused drug + compressed gene-expression IC50 model.")
    parser.add_argument("--split_protocol", type=str, default="random",
                         choices=["random", "drug_blind", "cell_blind", "both_blind"])
    parser.add_argument("--no_crossgraph", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--encoder", type=str, default="gat", choices=["gat", "gcn", "grover"])
    parser.add_argument("--gene_tokens", type=str, default="autoencoder_chunks",
                         choices=["autoencoder_chunks", "gene_modules"],
                         help="autoencoder_chunks: arbitrary reshape of the 128-dim AE bottleneck into 8x16 tokens. "
                              "gene_modules: 32 real co-expression gene-module scores, one scalar token each.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_random(rows: List[Dict], seed: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n = len(indices)
    train_size = max(1, int(0.8 * n))
    val_size = max(1, int(0.1 * n))
    return indices[:train_size], indices[train_size:train_size + val_size], indices[train_size + val_size:]


def split_by_group(rows: List[Dict], group_key: str, seed: int) -> Tuple[List[int], List[int], List[int]]:
    # Entire groups (drugs, or cell lines) go to one split only -- the group never
    # appears in both train and test, so the model cannot memorize a per-group effect.
    groups = sorted({row[group_key] for row in rows})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    train_groups = set(groups[: max(1, int(0.8 * n))])
    val_groups = set(groups[max(1, int(0.8 * n)): max(1, int(0.8 * n)) + max(1, int(0.1 * n))])
    test_groups = set(groups) - train_groups - val_groups
    train_idx = [i for i, r in enumerate(rows) if r[group_key] in train_groups]
    val_idx = [i for i, r in enumerate(rows) if r[group_key] in val_groups]
    test_idx = [i for i, r in enumerate(rows) if r[group_key] in test_groups]
    return train_idx, val_idx, test_idx


def split_both_blind(rows: List[Dict], seed: int) -> Tuple[List[int], List[int], List[int]]:
    drugs = sorted({row["DRUG_NAME"] for row in rows})
    cells = sorted({row["SANGER_MODEL_ID"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(drugs)
    rng.shuffle(cells)

    def partition(items):
        n = len(items)
        tr = set(items[: max(1, int(0.8 * n))])
        va = set(items[max(1, int(0.8 * n)): max(1, int(0.8 * n)) + max(1, int(0.1 * n))])
        te = set(items) - tr - va
        return tr, va, te

    drug_tr, drug_va, drug_te = partition(drugs)
    cell_tr, cell_va, cell_te = partition(cells)

    train_idx, val_idx, test_idx = [], [], []
    for i, r in enumerate(rows):
        if r["DRUG_NAME"] in drug_tr and r["SANGER_MODEL_ID"] in cell_tr:
            train_idx.append(i)
        elif r["DRUG_NAME"] in drug_va and r["SANGER_MODEL_ID"] in cell_va:
            val_idx.append(i)
        elif r["DRUG_NAME"] in drug_te and r["SANGER_MODEL_ID"] in cell_te:
            test_idx.append(i)
        # rows that straddle a train-group drug with a test-group cell (etc.) are
        # dropped entirely -- they would leak partial identity information either way.
    return train_idx, val_idx, test_idx


def reshape_gene_tokens(latent: List[float]) -> torch.Tensor:
    arr = torch.tensor(latent, dtype=torch.float32)
    return arr.view(NUM_GENE_TOKENS, GENE_TOKEN_DIM)


def reshape_gene_modules(module_scores: List[float]) -> torch.Tensor:
    arr = torch.tensor(module_scores, dtype=torch.float32)
    return arr.view(NUM_GENE_MODULES, 1)


class FusedDataset(Dataset):
    def __init__(self, rows: List[Dict], indices: List[int], drug_graphs: Dict[str, Data], gene_tokens: Dict[str, torch.Tensor]):
        self.samples = []
        for i in indices:
            row = rows[i]
            graph = drug_graphs.get(row["DRUG_NAME"])
            tokens = gene_tokens.get(row["SANGER_MODEL_ID"])
            if graph is None or tokens is None:
                continue
            try:
                label = float(row["LN_IC50"])
            except (TypeError, ValueError):
                continue
            self.samples.append((graph, tokens, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fused(batch):
    graphs = [item[0] for item in batch]
    tokens = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return Batch.from_data_list(graphs), tokens, labels


class LigandEncoderGCN(nn.Module):
    # Same depth/width as the GAT-based LigandEncoder in protliggnn_train.py, swapping
    # GATConv for GCNConv -- isolates the encoder choice as the only architectural
    # difference, for a fair comparison.
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        x = self.norm1(F.relu(self.conv1(x, edge_index)))
        x = self.norm2(F.relu(self.conv2(x, edge_index)))
        x = self.norm3(F.relu(self.conv3(x, edge_index)))
        return x


class LigandEncoderGrover(nn.Module):
    """GROVER-style ligand encoder: global transformer self-attention over each
    molecule's atoms, instead of the local message passing GAT/GCN restrict to
    bonded/graph-adjacent neighbors. This is a from-scratch, randomly-initialized
    transformer -- the original GROVER paper's pretrained self-supervised weights
    are not available in this environment, so this isolates the *architectural*
    difference (global attention vs. local convolution), not GROVER's pretraining
    benefit. That distinction is reported honestly alongside any results.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, batch_index):
        x = self.input_proj(x)
        batch_size = int(batch_index.max().item()) + 1
        outputs = torch.zeros_like(x)
        for b in range(batch_size):
            mask = batch_index == b
            tokens = x[mask].unsqueeze(0)  # [1, num_atoms_in_molecule, hidden_dim]
            encoded = self.transformer(tokens).squeeze(0)
            outputs[mask] = encoded
        return self.norm(outputs)


ENCODER_REGISTRY = {
    "gat": lambda hidden_dim: LigandEncoder(LIGAND_FEATURE_DIM, hidden_dim),
    "gcn": lambda hidden_dim: LigandEncoderGCN(LIGAND_FEATURE_DIM, hidden_dim),
    "grover": lambda hidden_dim: LigandEncoderGrover(LIGAND_FEATURE_DIM, hidden_dim),
}


class FusedIC50Model(nn.Module):
    def __init__(self, hidden_dim: int = 128, no_crossgraph: bool = False, encoder: str = "gat",
                 gene_tokens: str = "autoencoder_chunks"):
        super().__init__()
        self.no_crossgraph = no_crossgraph
        self.hidden_dim = hidden_dim
        self.gene_tokens_mode = gene_tokens
        self.encoder_name = encoder
        self.drug_encoder = ENCODER_REGISTRY[encoder](hidden_dim)
        if gene_tokens == "gene_modules":
            # Each token is a single real co-expression-module score; a shared linear
            # lift plus a learned per-module identity embedding (since all 32 scalars
            # would otherwise look alike to a single shared Linear(1, hidden_dim)).
            self.gene_token_proj = nn.Linear(1, hidden_dim)
            self.gene_module_id_embedding = nn.Embedding(NUM_GENE_MODULES, hidden_dim)
        else:
            self.gene_token_proj = nn.Linear(GENE_TOKEN_DIM, hidden_dim)
            self.gene_module_id_embedding = None
        self.cross_attention = BidirectionalCrossAttention(hidden_dim=hidden_dim, num_heads=4)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, drug_batch: Batch, gene_tokens_batch: torch.Tensor):
        if self.encoder_name == "grover":
            drug_x = self.drug_encoder(drug_batch.x, drug_batch.edge_index, drug_batch.batch)
        else:
            drug_x = self.drug_encoder(drug_batch.x, drug_batch.edge_index)
        gene_x = self.gene_token_proj(gene_tokens_batch)  # [B, num_gene_tokens, hidden_dim]
        if self.gene_module_id_embedding is not None:
            num_tokens = gene_tokens_batch.shape[1]
            token_ids = torch.arange(num_tokens, device=gene_tokens_batch.device)
            gene_x = gene_x + self.gene_module_id_embedding(token_ids).unsqueeze(0)

        drug_pools, gene_pools = [], []
        batch_size = drug_batch.num_graphs
        for b in range(batch_size):
            mask = drug_batch.batch == b
            drug_nodes = drug_x[mask]
            gene_nodes = gene_x[b]

            if self.no_crossgraph:
                drug_ctx = drug_nodes
                gene_ctx = gene_nodes
            else:
                drug_ctx, gene_ctx = self.cross_attention(drug_nodes, gene_nodes)

            drug_pools.append(drug_ctx.mean(dim=0))
            gene_pools.append(gene_ctx.mean(dim=0))

        drug_pool = torch.stack(drug_pools)
        gene_pool = torch.stack(gene_pools)
        joint = self.dropout(torch.cat([drug_pool, gene_pool], dim=-1))
        return self.regressor(joint).squeeze(-1)


def evaluate(model, loader, device) -> Dict[str, object]:
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for drug_batch, gene_tokens, labels in loader:
            drug_batch = drug_batch.to(device)
            gene_tokens = gene_tokens.to(device)
            labels = labels.to(device)
            output = model(drug_batch, gene_tokens)
            loss = F.mse_loss(output, labels)
            total_loss += float(loss.item()) * len(labels)
            count += len(labels)
            preds.extend(output.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    pcc = pearsonr(trues, preds)[0] if len(set(preds)) > 1 and len(trues) > 1 else 0.0
    spearman = spearmanr(trues, preds)[0] if len(set(preds)) > 1 and len(trues) > 1 else 0.0
    rmse = (sum((t - p) ** 2 for t, p in zip(trues, preds)) / max(len(trues), 1)) ** 0.5
    mae = sum(abs(t - p) for t, p in zip(trues, preds)) / max(len(trues), 1)
    ss_res = sum((t - p) ** 2 for t, p in zip(trues, preds))
    mean_t = sum(trues) / max(len(trues), 1)
    ss_tot = sum((t - mean_t) ** 2 for t in trues) or 1.0
    r2 = 1 - ss_res / ss_tot
    dispersion = (np.std(preds) / np.std(trues)) if len(trues) > 1 and np.std(trues) > 0 else 0.0
    return {"loss": total_loss / max(count, 1), "pcc": pcc, "spearman": spearman, "rmse": rmse,
            "mae": mae, "r2": r2, "dispersion_ratio": float(dispersion), "preds": preds, "trues": trues}


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    set_seed(args.seed)
    device = torch.device(args.device)
    run_name = args.run_name or f"{'no_crossgraph' if args.no_crossgraph else 'crossattn'}_{args.split_protocol}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_path) if args.save_path else OUTPUT_DIR / f"best_{run_name}.pt"

    gene_source_path = GENE_MODULES if args.gene_tokens == "gene_modules" else GENE_EMBEDDINGS
    print(f"loading GDSC2 data; gene tokens ({args.gene_tokens}) from {gene_source_path}")
    rows = load_rows(GDSC_CSV)
    gene_raw = json.loads(gene_source_path.read_text())
    smiles_cache = load_pubchem_cache()

    rows = [r for r in rows if r.get("SANGER_MODEL_ID") in gene_raw and smiles_cache.get(r.get("DRUG_NAME"))]
    print(f"rows with both drug SMILES and gene embedding available: {len(rows)}")
    random.Random(args.seed).shuffle(rows)
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"using {len(rows)} rows")

    unique_drugs = sorted({r["DRUG_NAME"] for r in rows})
    drug_graphs = {}
    skipped = 0
    for idx, name in enumerate(unique_drugs, start=1):
        smiles = smiles_cache.get(name)
        graph = embed_drug_graph(smiles) if smiles else None
        if graph is not None:
            drug_graphs[name] = graph
        else:
            skipped += 1
        if idx % 50 == 0 or idx == len(unique_drugs):
            print(f"embedded {idx}/{len(unique_drugs)} drugs ({skipped} skipped so far)")
    print(f"successfully embedded drug graphs: {len(drug_graphs)}/{len(unique_drugs)}")

    reshape_fn = reshape_gene_modules if args.gene_tokens == "gene_modules" else reshape_gene_tokens
    gene_tokens_map = {sid: reshape_fn(vec) for sid, vec in gene_raw.items()}

    if args.split_protocol == "random":
        train_idx, val_idx, test_idx = split_random(rows, args.seed)
    elif args.split_protocol == "drug_blind":
        train_idx, val_idx, test_idx = split_by_group(rows, "DRUG_NAME", args.seed)
    elif args.split_protocol == "cell_blind":
        train_idx, val_idx, test_idx = split_by_group(rows, "SANGER_MODEL_ID", args.seed)
    else:
        train_idx, val_idx, test_idx = split_both_blind(rows, args.seed)

    train_ds = FusedDataset(rows, train_idx, drug_graphs, gene_tokens_map)
    val_ds = FusedDataset(rows, val_idx, drug_graphs, gene_tokens_map)
    test_ds = FusedDataset(rows, test_idx, drug_graphs, gene_tokens_map)
    print(f"[{run_name}] train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    if len(train_ds) < 10 or len(test_ds) < 5:
        print(f"WARNING: split '{args.split_protocol}' produced too few usable samples; aborting this run.")
        summary = {"run_name": run_name, "split_protocol": args.split_protocol, "no_crossgraph": args.no_crossgraph,
                   "status": "insufficient_data", "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds)}
        (OUTPUT_DIR / f"summary_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fused)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fused)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fused)

    model = FusedIC50Model(hidden_dim=args.hidden_dim, no_crossgraph=args.no_crossgraph, encoder=args.encoder,
                           gene_tokens=args.gene_tokens).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for drug_batch, gene_tokens_b, labels in train_loader:
            drug_batch = drug_batch.to(device)
            gene_tokens_b = gene_tokens_b.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(drug_batch, gene_tokens_b)
            loss = F.mse_loss(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(labels)
            train_count += len(labels)

        train_loss = train_loss_total / max(train_count, 1)
        val_metrics = evaluate(model, val_loader, device)
        print(f"[{run_name}] Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
              f"val_PCC={val_metrics['pcc']:.4f} val_Spearman={val_metrics['spearman']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"],
                         "val_pcc": val_metrics["pcc"], "val_spearman": val_metrics["spearman"]})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[{run_name}] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[{run_name}] Loaded best model from epoch {best_epoch}\n"
          f"Test loss={test_metrics['loss']:.4f} PCC={test_metrics['pcc']:.4f} Spearman={test_metrics['spearman']:.4f} "
          f"RMSE={test_metrics['rmse']:.4f} MAE={test_metrics['mae']:.4f} R2={test_metrics['r2']:.4f} "
          f"Dispersion={test_metrics['dispersion_ratio']:.4f}")

    with (OUTPUT_DIR / f"epoch_history_{run_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_pcc", "val_spearman"])
        writer.writeheader()
        writer.writerows(history)

    with (OUTPUT_DIR / f"test_predictions_{run_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_ln_ic50", "predicted_ln_ic50"])
        for t, p in zip(test_metrics["trues"], test_metrics["preds"]):
            writer.writerow([t, p])

    if test_metrics["trues"]:
        plt.figure(figsize=(6, 6))
        plt.scatter(test_metrics["trues"], test_metrics["preds"], alpha=0.4, s=10)
        min_v = min(min(test_metrics["trues"]), min(test_metrics["preds"]))
        max_v = max(max(test_metrics["trues"]), max(test_metrics["preds"]))
        plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
        plt.xlabel("True LN(IC50)")
        plt.ylabel("Predicted LN(IC50)")
        plt.title(f"Fused IC50 Model ({run_name})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"scatter_{run_name}.png", dpi=150)
        plt.close()

    summary = {
        "run_name": run_name, "split_protocol": args.split_protocol, "no_crossgraph": args.no_crossgraph,
        "encoder": args.encoder, "gene_tokens": args.gene_tokens, "status": "ok", "best_epoch": best_epoch,
        "test_loss": test_metrics["loss"], "test_pcc": test_metrics["pcc"], "test_spearman": test_metrics["spearman"],
        "test_rmse": test_metrics["rmse"], "test_mae": test_metrics["mae"], "test_r2": test_metrics["r2"],
        "test_dispersion_ratio": test_metrics["dispersion_ratio"],
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
        "unique_drugs_embedded": len(drug_graphs),
    }
    (OUTPUT_DIR / f"summary_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
