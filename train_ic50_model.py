import argparse
import csv
import json
import random
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch.utils.data import DataLoader, Dataset

from protliggnn_train import LIGAND_FEATURE_DIM, LigandEncoder, mol_to_ligand_graph
from pubchem_lookup import load_cache as load_pubchem_cache


GDSC_CSV = Path("data") / "gdsc" / "GDSC2_fitted_dose_response.csv"
OUTPUT_DIR = Path("outputs") / "ic50_model"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Drug+CellLine GNN to predict IC50 (GDSC2).")
    parser.add_argument("--gdsc_csv", type=str, default=str(GDSC_CSV))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--run_name", type=str, default="ic50_gnn")
    parser.add_argument("--no_crossgraph", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


class _EmbedTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _EmbedTimeout()


def embed_drug_graph(smiles: str, timeout_seconds: int = 10) -> Optional[Data]:
    # RDKit's ETKDG embedder can occasionally spin for a very long time on
    # difficult/strained real-drug structures; hard-cap it with SIGALRM so one
    # bad molecule can't stall an entire training run.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_seconds)
    try:
        params = AllChem.ETKDGv3()
        params.maxIterations = 200
        status = AllChem.EmbedMolecule(mol, params)
    except _EmbedTimeout:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    if status != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    try:
        return mol_to_ligand_graph(mol)
    except Exception:
        return None


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


class IC50Dataset(Dataset):
    def __init__(self, rows: List[Tuple[str, int, float]], drug_graphs: Dict[str, Data]):
        self.samples = []
        for drug_name, cell_idx, label in rows:
            graph = drug_graphs.get(drug_name)
            if graph is None:
                continue
            self.samples.append((graph, cell_idx, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ic50(batch):
    graphs = [item[0] for item in batch]
    cell_idx = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return Batch.from_data_list(graphs), cell_idx, labels


class DrugCellIC50Model(nn.Module):
    def __init__(self, num_cell_lines: int, hidden_dim: int = 128, no_crossgraph: bool = False):
        super().__init__()
        self.no_crossgraph = no_crossgraph
        self.drug_encoder = LigandEncoder(LIGAND_FEATURE_DIM, hidden_dim)
        self.cell_embedding = nn.Embedding(num_cell_lines, hidden_dim)
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, drug_batch: Batch, cell_idx: torch.Tensor):
        drug_x = self.drug_encoder(drug_batch.x, drug_batch.edge_index)
        drug_pool = global_mean_pool(drug_x, drug_batch.batch)
        cell_emb = self.cell_embedding(cell_idx)

        if self.no_crossgraph:
            context = torch.zeros_like(drug_pool)
        else:
            gamma, beta = self.film(cell_emb).chunk(2, dim=-1)
            context = drug_pool * (1.0 + gamma) + beta

        joint = torch.cat([drug_pool, context, cell_emb], dim=-1)
        return self.regressor(joint).squeeze(-1)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        for drug_batch, cell_idx, labels in loader:
            drug_batch = drug_batch.to(device)
            cell_idx = cell_idx.to(device)
            labels = labels.to(device)
            output = model(drug_batch, cell_idx)
            loss = F.mse_loss(output, labels)
            total_loss += float(loss.item()) * len(labels)
            count += len(labels)
            preds.extend(output.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    pcc = pearsonr(trues, preds)[0] if len(set(preds)) > 1 else 0.0
    spearman = spearmanr(trues, preds)[0] if len(set(preds)) > 1 else 0.0
    rmse = (sum((t - p) ** 2 for t, p in zip(trues, preds)) / max(len(trues), 1)) ** 0.5
    mae = sum(abs(t - p) for t, p in zip(trues, preds)) / max(len(trues), 1)
    return {
        "loss": total_loss / max(count, 1),
        "pcc": pcc,
        "spearman": spearman,
        "rmse": rmse,
        "mae": mae,
        "preds": preds,
        "trues": trues,
    }


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    set_seed(args.seed)
    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_path = Path(args.save_path) if args.save_path else OUTPUT_DIR / "best_ic50_model.pt"

    print(f"loading GDSC2 data from {args.gdsc_csv}")
    rows = load_rows(Path(args.gdsc_csv))
    random.Random(args.seed).shuffle(rows)
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"using {len(rows)} rows")

    smiles_cache = load_pubchem_cache()
    unique_drugs = sorted({row["DRUG_NAME"] for row in rows})
    print(f"unique drugs in sample: {len(unique_drugs)}")
    drug_graphs = build_drug_graph_cache(unique_drugs, smiles_cache)
    print(f"successfully embedded drug graphs: {len(drug_graphs)}/{len(unique_drugs)}")

    cell_names = sorted({row["CELL_LINE_NAME"] for row in rows})
    cell_to_idx = {name: idx for idx, name in enumerate(cell_names)}
    print(f"unique cell lines: {len(cell_names)}")

    prepared_rows = []
    skipped = 0
    for row in rows:
        drug_name = row["DRUG_NAME"]
        if drug_name not in drug_graphs:
            skipped += 1
            continue
        try:
            label = float(row["LN_IC50"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        prepared_rows.append((drug_name, cell_to_idx[row["CELL_LINE_NAME"]], label))
    print(f"usable rows: {len(prepared_rows)} (skipped {skipped})")

    train_idx, val_idx, test_idx = split_indices(len(prepared_rows), args.seed)
    train_rows = [prepared_rows[i] for i in train_idx]
    val_rows = [prepared_rows[i] for i in val_idx]
    test_rows = [prepared_rows[i] for i in test_idx]

    train_ds = IC50Dataset(train_rows, drug_graphs)
    val_ds = IC50Dataset(val_rows, drug_graphs)
    test_ds = IC50Dataset(test_rows, drug_graphs)
    print(f"train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ic50)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ic50)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ic50)

    model = DrugCellIC50Model(
        num_cell_lines=len(cell_names), hidden_dim=args.hidden_dim, no_crossgraph=args.no_crossgraph
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for drug_batch, cell_idx, labels in train_loader:
            drug_batch = drug_batch.to(device)
            cell_idx = cell_idx.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(drug_batch, cell_idx)
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
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_pcc": val_metrics["pcc"],
                "val_spearman": val_metrics["spearman"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "cell_to_idx": cell_to_idx,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / f"epoch_history_{args.run_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_pcc", "val_spearman"])
        writer.writeheader()
        writer.writerows(history)

    with (OUTPUT_DIR / f"test_predictions_{args.run_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_ln_ic50", "predicted_ln_ic50"])
        for t, p in zip(test_metrics["trues"], test_metrics["preds"]):
            writer.writerow([t, p])

    plt.figure(figsize=(6, 6))
    plt.scatter(test_metrics["trues"], test_metrics["preds"], alpha=0.4, s=10)
    min_v = min(min(test_metrics["trues"]), min(test_metrics["preds"]))
    max_v = max(max(test_metrics["trues"]), max(test_metrics["preds"]))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel("True LN(IC50)")
    plt.ylabel("Predicted LN(IC50)")
    plt.title(f"IC50 Model ({args.run_name}): Predicted vs True")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"scatter_{args.run_name}.png", dpi=150)
    plt.close()

    summary = {
        "run_name": args.run_name,
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
