import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EXPR_DIR = Path("outputs") / "gene_expression"
OUTPUT_DIR = Path("outputs") / "gene_autoencoder"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a stacked autoencoder on preprocessed gene expression.")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class StackedAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = np.load(EXPR_DIR / "preprocessed_expression.npy")
    sanger_ids = json.loads((EXPR_DIR / "cell_line_sanger_ids.json").read_text())
    print(f"expression matrix: {matrix.shape}")

    num_cell_lines = matrix.shape[0]
    indices = np.arange(num_cell_lines)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    val_size = max(1, int(0.1 * num_cell_lines))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_tensor = torch.tensor(matrix[train_idx], dtype=torch.float32)
    val_tensor = torch.tensor(matrix[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu")
    model = StackedAutoencoder(input_dim=matrix.shape[1], latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    history = []
    patience, patience_counter = 15, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * batch.size(0)
        train_loss = train_loss_total / len(train_idx)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss_total += float(loss.item()) * batch.size(0)
        val_loss = val_loss_total / len(val_idx)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val_loss {best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()

    full_tensor = torch.tensor(matrix, dtype=torch.float32)
    with torch.no_grad():
        recon_full, embeddings = model(full_tensor)
        final_recon_mse = float(criterion(recon_full, full_tensor).item())

    embeddings_np = embeddings.numpy()
    cell_line_to_embedding = {sid: embeddings_np[i].tolist() for i, sid in enumerate(sanger_ids)}

    torch.save(
        {"model_state_dict": model.state_dict(), "input_dim": matrix.shape[1], "latent_dim": args.latent_dim},
        OUTPUT_DIR / "gene_autoencoder.pt",
    )
    (OUTPUT_DIR / "cell_line_embeddings.json").write_text(json.dumps(cell_line_to_embedding), encoding="utf-8")

    with (OUTPUT_DIR / "training_history.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,val_loss\n")
        for row in history:
            handle.write(f"{row['epoch']},{row['train_loss']},{row['val_loss']}\n")

    plt.figure(figsize=(7, 5))
    epochs_list = [h["epoch"] for h in history]
    plt.plot(epochs_list, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs_list, [h["val_loss"] for h in history], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title("Gene Expression Autoencoder: Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "autoencoder_loss_curve.png", dpi=150)
    plt.close()

    summary = {
        "input_dim": matrix.shape[1],
        "latent_dim": args.latent_dim,
        "num_cell_lines": num_cell_lines,
        "best_val_recon_mse": best_val_loss,
        "final_full_recon_mse": final_recon_mse,
        "best_epoch": epochs_list[-1] - patience_counter if patience_counter >= patience else epochs_list[-1],
    }
    (OUTPUT_DIR / "autoencoder_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
