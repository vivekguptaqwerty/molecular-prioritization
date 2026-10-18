import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Batch

from infer_affinity import build_graph_pair, load_model as load_protliggnn_model


GENAI_DIR = Path("outputs") / "genai"
VOCAB_PATH = GENAI_DIR / "smiles_vocab.json"
BASE_GENERATOR_PATH = GENAI_DIR / "smiles_lstm_generator.pt"
OUTPUT_DIR = Path("outputs") / "rl_genai"
RL_MODEL_PATH = OUTPUT_DIR / "rl_smiles_generator.pt"
HISTORY_PATH = OUTPUT_DIR / "rl_training_history.csv"
SAMPLE_PATH = OUTPUT_DIR / "rl_generated_molecules.csv"
SUMMARY_PATH = OUTPUT_DIR / "rl_vs_base_summary.md"

DEFAULT_PROTEIN_PDB = Path("data") / "pdbbind2020" / "pdbbind_subset" / "2011-2019" / "2l3r" / "2l3r_protein.pdb"
INVALID_PENALTY = -2.0


class SmilesLSTMGenerator(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, hidden=None):
        embedded = self.embedding(input_ids)
        outputs, hidden = self.lstm(embedded, hidden)
        return self.output(outputs), hidden


def parse_args():
    parser = argparse.ArgumentParser(description="RL fine-tune the SMILES LSTM against ProtLigGNN predicted affinity.")
    parser.add_argument("--protein_pdb", type=str, default=str(DEFAULT_PROTEIN_PDB))
    parser.add_argument("--checkpoint", type=str, default="outputs/best_protliggnn.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--final_sample_size", type=int, default=200)
    return parser.parse_args()


def load_vocab(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    char_to_idx = {str(char): int(idx) for char, idx in payload["char_to_idx"].items()}
    idx_to_char = {int(idx): str(char) for idx, char in payload["idx_to_char"].items()}
    return char_to_idx, idx_to_char


def load_base_generator(path: Path, vocab_size: int) -> SmilesLSTMGenerator:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["model_config"]
    model = SmilesLSTMGenerator(
        vocab_size=vocab_size,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        embedding_dim=int(config.get("embedding_dim", 64)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def sample_batch_with_logprobs(
    model: SmilesLSTMGenerator,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    batch_size: int,
    max_length: int,
    temperature: float,
) -> Tuple[List[str], torch.Tensor]:
    bos_idx = char_to_idx["<BOS>"]
    eos_idx = char_to_idx["<EOS>"]
    pad_idx = char_to_idx["<PAD>"]

    current = torch.full((batch_size, 1), bos_idx, dtype=torch.long)
    hidden = None
    finished = torch.zeros(batch_size, dtype=torch.bool)
    sequence_logprobs = torch.zeros(batch_size)
    char_lists: List[List[str]] = [[] for _ in range(batch_size)]

    for _ in range(max_length):
        logits, hidden = model(current, hidden)
        next_logits = logits[:, -1, :] / temperature
        next_logits = next_logits.clone()
        next_logits[:, pad_idx] = -float("inf")
        next_logits[:, bos_idx] = -float("inf")
        log_probs = F.log_softmax(next_logits, dim=-1)
        probs = log_probs.exp()
        next_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        chosen_logprob = log_probs.gather(1, next_idx.unsqueeze(-1)).squeeze(-1)

        active = ~finished
        sequence_logprobs = sequence_logprobs + chosen_logprob * active.float()

        for b in range(batch_size):
            if finished[b]:
                continue
            token_id = int(next_idx[b].item())
            if token_id == eos_idx:
                finished[b] = True
            else:
                char_lists[b].append(idx_to_char[token_id])

        current = next_idx.unsqueeze(-1)
        if bool(finished.all()):
            break

    smiles_list = ["".join(chars) for chars in char_lists]
    return smiles_list, sequence_logprobs


def embed_smiles_to_sdf(smiles: str, output_path: Path) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status != 0:
        return False
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    mol.SetProp("_Name", smiles)
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    return output_path.exists() and output_path.stat().st_size > 0


def score_smiles(
    smiles: str,
    reward_model,
    protein_pdb: Path,
    device: torch.device,
    work_dir: Path,
    tag: str,
) -> float:
    if not smiles:
        return INVALID_PENALTY
    sdf_path = work_dir / f"{tag}.sdf"
    if not embed_smiles_to_sdf(smiles, sdf_path):
        return INVALID_PENALTY
    try:
        ligand_graph, protein_graph = build_graph_pair(protein_pdb, sdf_path)
        ligand_batch = Batch.from_data_list([ligand_graph]).to(device)
        protein_batch = Batch.from_data_list([protein_graph]).to(device)
        with torch.no_grad():
            return float(reward_model(ligand_batch, protein_batch).item())
    except Exception:
        return INVALID_PENALTY


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    protein_pdb = Path(args.protein_pdb)
    checkpoint = Path(args.checkpoint)
    if not protein_pdb.exists():
        raise FileNotFoundError(f"Protein PDB not found: {protein_pdb}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"ProtLigGNN checkpoint not found: {checkpoint}")

    char_to_idx, idx_to_char = load_vocab(VOCAB_PATH)
    policy = load_base_generator(BASE_GENERATOR_PATH, vocab_size=len(char_to_idx))
    policy.train()

    reward_model = load_protliggnn_model(checkpoint, device)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"RL target protein: {protein_pdb}")
    print(f"iterations: {args.iterations} | batch_size: {args.batch_size}")

    history = []
    with tempfile.TemporaryDirectory(prefix="rl_genai_") as temp_dir:
        work_dir = Path(temp_dir)
        for iteration in range(1, args.iterations + 1):
            smiles_list, sequence_logprobs = sample_batch_with_logprobs(
                policy, char_to_idx, idx_to_char, args.batch_size, args.max_length, args.temperature
            )

            rewards = []
            valid_count = 0
            for idx, smiles in enumerate(smiles_list):
                reward = score_smiles(smiles, reward_model, protein_pdb, device, work_dir, tag=f"it{iteration}_{idx}")
                if reward != INVALID_PENALTY:
                    valid_count += 1
                rewards.append(reward)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            baseline = reward_tensor.mean()
            advantage = reward_tensor - baseline

            loss = -(sequence_logprobs * advantage.detach()).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            optimizer.step()

            mean_reward = float(reward_tensor.mean().item())
            valid_fraction = valid_count / len(smiles_list)
            valid_rewards = [r for r in rewards if r != INVALID_PENALTY]
            mean_valid_affinity = sum(valid_rewards) / len(valid_rewards) if valid_rewards else float("nan")

            history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.item()),
                    "mean_reward": mean_reward,
                    "valid_fraction": valid_fraction,
                    "mean_valid_affinity": mean_valid_affinity,
                }
            )
            print(
                f"iter {iteration}/{args.iterations} | loss {loss.item():.4f} | "
                f"mean_reward {mean_reward:.4f} | valid {valid_fraction:.2f} | "
                f"mean_valid_affinity {mean_valid_affinity:.4f}"
            )

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "loss", "mean_reward", "valid_fraction", "mean_valid_affinity"])
        writer.writeheader()
        writer.writerows(history)

    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "model_config": {
                "vocab_size": len(char_to_idx),
                "hidden_size": policy.lstm.hidden_size,
                "num_layers": policy.lstm.num_layers,
                "embedding_dim": policy.embedding.embedding_dim,
            },
            "rl_args": vars(args),
            "base_generator": str(BASE_GENERATOR_PATH),
        },
        RL_MODEL_PATH,
    )

    policy.eval()
    final_rows = []
    seen = set()
    with tempfile.TemporaryDirectory(prefix="rl_genai_final_") as temp_dir:
        work_dir = Path(temp_dir)
        attempts = 0
        while len(final_rows) < args.final_sample_size and attempts < args.final_sample_size * 4:
            attempts += 1
            with torch.no_grad():
                smiles_list, _ = sample_batch_with_logprobs(
                    policy, char_to_idx, idx_to_char, 1, args.max_length, args.temperature
                )
            smiles = smiles_list[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical in seen:
                continue
            seen.add(canonical)
            reward = score_smiles(canonical, reward_model, protein_pdb, device, work_dir, tag=f"final_{attempts}")
            if reward == INVALID_PENALTY:
                continue
            final_rows.append({"smiles": canonical, "predicted_affinity": reward})

    final_rows.sort(key=lambda row: row["predicted_affinity"], reverse=True)
    with SAMPLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles", "predicted_affinity"])
        writer.writeheader()
        writer.writerows(final_rows)

    base_scored_path = GENAI_DIR / "generated_molecules_scored.csv"
    base_mean_affinity = None
    if base_scored_path.exists():
        with base_scored_path.open("r", newline="", encoding="utf-8") as handle:
            base_rows = list(csv.DictReader(handle))
        if base_rows:
            base_mean_affinity = sum(float(r["predicted_affinity"]) for r in base_rows) / len(base_rows)

    rl_mean_affinity = (
        sum(row["predicted_affinity"] for row in final_rows) / len(final_rows) if final_rows else None
    )

    summary_lines = [
        "# R.L Generator vs Base Supervised Generator",
        "",
        "RL fine-tunes the pretrained character-level SMILES LSTM with REINFORCE, using the",
        "trained ProtLigGNN model's predicted affinity against the 2L3R target as the reward",
        "signal (invalid/unembeddable molecules receive a fixed penalty).",
        "",
        f"- Target protein: {protein_pdb}",
        f"- RL iterations: {args.iterations} | batch size: {args.batch_size}",
        f"- Base supervised generator mean predicted affinity (first {len(base_rows) if base_scored_path.exists() else 0} scored molecules): "
        + (f"{base_mean_affinity:.4f}" if base_mean_affinity is not None else "N/A"),
        f"- RL generator mean predicted affinity ({len(final_rows)} valid unique molecules sampled post-RL): "
        + (f"{rl_mean_affinity:.4f}" if rl_mean_affinity is not None else "N/A"),
        "",
        "## Top RL-Generated Candidates",
        "",
    ]
    summary_lines.append("|rank|smiles|predicted_affinity|")
    summary_lines.append("|---|---|---|")
    for rank, row in enumerate(final_rows[:10], start=1):
        summary_lines.append(f"|{rank}|{row['smiles']}|{row['predicted_affinity']:.4f}|")

    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved RL model: {RL_MODEL_PATH}")
    print(f"saved training history: {HISTORY_PATH}")
    print(f"saved final RL sample: {SAMPLE_PATH}")
    print(f"saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
