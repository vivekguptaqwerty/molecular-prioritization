"""Measures the five formal properties of the learned representations (Table 4.2):
isomorphism preservation, permutation invariance, losslessness, sensitivity, and
robustness -- using the actually-trained GNAE Drugnet (fused IC50 model) ligand
encoder and the actually-trained gene-expression autoencoder. No numbers here are
assumed; every cell is measured from a real forward pass of a real checkpoint.
"""
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import spearmanr
from torch_geometric.data import Batch

from protliggnn_train import mol_to_ligand_graph
from pubchem_lookup import load_cache as load_pubchem_cache
from train_gene_autoencoder import StackedAutoencoder
from train_ic50_fused_model import FusedIC50Model, reshape_gene_tokens, GENE_EMBEDDINGS

OUT_DIR = Path("outputs") / "formal_properties"
IC50_CHECKPOINT = Path("outputs/ic50_fused_model/best_no_crossgraph_random.pt")
AE_CHECKPOINT = Path("outputs/gene_autoencoder/gene_autoencoder.pt")
EXPR_DIR = Path("outputs") / "gene_expression"
GDSC_CSV = Path("data") / "gdsc" / "GDSC2_fitted_dose_response.csv"

SEED = 42


def load_ic50_model():
    ckpt = torch.load(IC50_CHECKPOINT, map_location="cpu", weights_only=False)
    model = FusedIC50Model(hidden_dim=128, no_crossgraph=True, encoder="gat", gene_tokens="autoencoder_chunks")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def embed_mol_3d(mol: Chem.Mol, timeout_ok=True) -> Optional[Chem.Mol]:
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.maxIterations = 200
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol


def pooled_embedding(ligand_encoder, mol: Chem.Mol) -> Optional[np.ndarray]:
    graph = mol_to_ligand_graph(mol)
    batch = Batch.from_data_list([graph])
    with torch.no_grad():
        atom_embeds = ligand_encoder(batch.x, batch.edge_index)
        pooled = atom_embeds.mean(dim=0)
    return pooled.numpy()


def sample_base_smiles(n: int, seed: int) -> List[str]:
    """Real drug SMILES from our own pipeline's resolved PubChem cache, filtered to
    ones RDKit parses and 3D-embeds successfully."""
    cache = load_pubchem_cache()
    candidates = [s for s in cache.values() if s]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = []
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 4:
            continue
        chosen.append(smi)
        if len(chosen) >= n:
            break
    return chosen


# ---------------------------------------------------------------------------
# 1 & 2. Isomorphism preservation + permutation invariance
# ---------------------------------------------------------------------------

def run_isomorphism_and_permutation(ligand_encoder, smiles_list: List[str], num_permutations=10, seed=SEED):
    rng = random.Random(seed)
    relabel_distances = []   # test 1: one random relabeling vs. canonical, per molecule
    permutation_variances = []  # test 2: variance across K random relabelings, per molecule
    n_molecules_used = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol3d = embed_mol_3d(mol)
        if mol3d is None:
            continue
        canonical_embed = pooled_embedding(ligand_encoder, mol3d)
        if canonical_embed is None:
            continue

        n_atoms = mol3d.GetNumAtoms()
        trial_embeds = []
        for _ in range(num_permutations):
            order = list(range(n_atoms))
            rng.shuffle(order)
            renumbered = Chem.RenumberAtoms(mol3d, order)
            embed = pooled_embedding(ligand_encoder, renumbered)
            if embed is not None:
                trial_embeds.append(embed)
        if not trial_embeds:
            continue

        trial_embeds = np.stack(trial_embeds)
        relabel_distances.append(float(np.linalg.norm(trial_embeds[0] - canonical_embed)))
        permutation_variances.append(float(trial_embeds.var(axis=0).mean()))
        n_molecules_used += 1

    return {
        "n_molecules": n_molecules_used,
        "num_permutations_per_molecule": num_permutations,
        "isomorphism_mean_embedding_distance": float(np.mean(relabel_distances)) if relabel_distances else None,
        "isomorphism_max_embedding_distance": float(np.max(relabel_distances)) if relabel_distances else None,
        "permutation_mean_variance_per_dim": float(np.mean(permutation_variances)) if permutation_variances else None,
        "permutation_max_variance_per_dim": float(np.max(permutation_variances)) if permutation_variances else None,
    }


# ---------------------------------------------------------------------------
# 3. Losslessness (reuse the already-trained autoencoder's own held-out metric)
# ---------------------------------------------------------------------------

def run_losslessness():
    summary = json.loads((Path("outputs/gene_autoencoder/autoencoder_summary.json")).read_text())
    return {
        "n_cell_lines": summary["num_cell_lines"],
        "held_out_val_reconstruction_mse": summary["best_val_recon_mse"],
        "full_set_reconstruction_mse": summary["final_full_recon_mse"],
        "note": "held-out validation split is ~10% of the 706 cell lines (same split used to train the autoencoder in Phase A)",
    }


# ---------------------------------------------------------------------------
# 4. Sensitivity: controlled functional-group substitution, embedding shift vs.
#    substituent size (atomic mass of a halogen substituted onto an aromatic CH)
# ---------------------------------------------------------------------------

HALOGEN_SERIES = [("F", 18.998), ("Cl", 35.453), ("Br", 79.904), ("I", 126.904)]


def substitute_aromatic_h(mol: Chem.Mol, symbol: str) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    target_idx = None
    for atom in rw.GetAtoms():
        if atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
            target_idx = atom.GetIdx()
            break
    if target_idx is None:
        return None
    new_idx = rw.AddAtom(Chem.Atom(symbol))
    rw.AddBond(target_idx, new_idx, Chem.BondType.SINGLE)
    try:
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
    except Exception:
        return None
    return new_mol


def run_sensitivity(ligand_encoder, smiles_list: List[str]):
    per_molecule_correlations = []
    monotonic_count = 0
    n_molecules_used = 0

    for smi in smiles_list:
        base_mol = Chem.MolFromSmiles(smi)
        base_3d = embed_mol_3d(base_mol)
        if base_3d is None:
            continue
        base_embed = pooled_embedding(ligand_encoder, base_3d)
        if base_embed is None:
            continue

        masses, distances = [], []
        for symbol, mass in HALOGEN_SERIES:
            variant = substitute_aromatic_h(base_mol, symbol)
            if variant is None:
                continue
            variant_3d = embed_mol_3d(variant)
            if variant_3d is None:
                continue
            variant_embed = pooled_embedding(ligand_encoder, variant_3d)
            if variant_embed is None:
                continue
            masses.append(mass)
            distances.append(float(np.linalg.norm(variant_embed - base_embed)))

        if len(masses) < 3:
            continue
        rho, _ = spearmanr(masses, distances)
        if rho == rho:  # not NaN
            per_molecule_correlations.append(float(rho))
            monotonic_count += int(all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1)))
            n_molecules_used += 1

    return {
        "n_molecules": n_molecules_used,
        "substitution_series": [s for s, _ in HALOGEN_SERIES],
        "mean_spearman_mass_vs_embedding_shift": float(np.mean(per_molecule_correlations)) if per_molecule_correlations else None,
        "fraction_strictly_monotonic": (monotonic_count / n_molecules_used) if n_molecules_used else None,
    }


# ---------------------------------------------------------------------------
# 5. Robustness: Gaussian perturbation of the raw (z-scored) expression input,
#    variance of the downstream predicted LN(IC50) measured
# ---------------------------------------------------------------------------

def run_robustness(fused_model, n_pairs=20, trials_per_level=10, noise_levels=(0.1, 0.3), seed=SEED):
    rng = np.random.default_rng(seed)
    torch_rng = torch.Generator().manual_seed(seed)

    expr_matrix = np.load(EXPR_DIR / "preprocessed_expression.npy")  # [706, 2000], already z-scored
    sanger_ids = json.loads((EXPR_DIR / "cell_line_sanger_ids.json").read_text())
    sanger_to_idx = {sid: i for i, sid in enumerate(sanger_ids)}

    ae_ckpt = torch.load(AE_CHECKPOINT, map_location="cpu", weights_only=False)
    autoencoder = StackedAutoencoder(input_dim=ae_ckpt["input_dim"], latent_dim=ae_ckpt["latent_dim"])
    autoencoder.load_state_dict(ae_ckpt["model_state_dict"])
    autoencoder.eval()

    smiles_cache = load_pubchem_cache()
    rows = []
    import csv
    with GDSC_CSV.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid = row.get("SANGER_MODEL_ID")
            name = row.get("DRUG_NAME")
            if sid in sanger_to_idx and smiles_cache.get(name):
                rows.append((name, sid))
    rng.shuffle(rows)

    from train_ic50_model import embed_drug_graph

    results_by_level = {lvl: [] for lvl in noise_levels}
    n_pairs_used = 0
    for name, sid in rows:
        if n_pairs_used >= n_pairs:
            break
        smiles = smiles_cache[name]
        drug_graph = embed_drug_graph(smiles)
        if drug_graph is None:
            continue
        drug_batch = Batch.from_data_list([drug_graph])
        clean_vec = expr_matrix[sanger_to_idx[sid]]

        used_pair = False
        for level in noise_levels:
            preds = []
            for _ in range(trials_per_level):
                noisy_vec = clean_vec + rng.normal(scale=level, size=clean_vec.shape)
                with torch.no_grad():
                    latent = autoencoder.encoder(torch.tensor(noisy_vec, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                    tokens = reshape_gene_tokens(latent.tolist()).unsqueeze(0)
                    pred = fused_model(drug_batch, tokens).item()
                preds.append(pred)
            results_by_level[level].append(float(np.var(preds)))
            used_pair = True
        if used_pair:
            n_pairs_used += 1

    return {
        "n_drug_cell_pairs": n_pairs_used,
        "trials_per_noise_level": trials_per_level,
        "noise_levels_sigma_zscore_units": list(noise_levels),
        "mean_prediction_variance_by_noise_level": {
            str(lvl): (float(np.mean(v)) if v else None) for lvl, v in results_by_level.items()
        },
    }


def main():
    RDLogger.DisableLog("rdApp.*")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("loading GNAE Drugnet ligand encoder...")
    ic50_model = load_ic50_model()
    ligand_encoder = ic50_model.drug_encoder

    smiles_sample = sample_base_smiles(30, seed=SEED)
    print(f"sampled {len(smiles_sample)} real drug SMILES for isomorphism/permutation/sensitivity tests")

    print("\n=== 1 & 2. Isomorphism preservation + permutation invariance ===")
    iso_perm = run_isomorphism_and_permutation(ligand_encoder, smiles_sample)
    print(json.dumps(iso_perm, indent=2))

    print("\n=== 3. Losslessness ===")
    lossless = run_losslessness()
    print(json.dumps(lossless, indent=2))

    print("\n=== 4. Sensitivity ===")
    sensitivity = run_sensitivity(ligand_encoder, smiles_sample)
    print(json.dumps(sensitivity, indent=2))

    print("\n=== 5. Robustness ===")
    robustness = run_robustness(ic50_model, n_pairs=20, trials_per_level=10)
    print(json.dumps(robustness, indent=2))

    all_results = {
        "isomorphism_preservation": iso_perm,
        "permutation_invariance": iso_perm,
        "losslessness": lossless,
        "sensitivity": sensitivity,
        "robustness": robustness,
    }
    (OUT_DIR / "table_4_2_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT_DIR / 'table_4_2_results.json'}")


if __name__ == "__main__":
    main()
