"""Build real candidates.csv and library.csv for the mentor's cdps_algorithm.py,
using our actually-trained GNAE Drugnet (fused IC50) and ProtLigGNN checkpoints,
plus real measured GDSC2 LN(IC50) as the observed_response ground truth.

Target cell line: SIDM00237 (PC-14, lung adenocarcinoma) -- chosen because it has
both the most GDSC2 drug-response rows (295) among cell lines with a real DepMap
transcriptomic embedding, and real gene expression (not an arbitrary filler).

Target protein: 2L3R, the same target already used for GenAI/R.L scoring
elsewhere in this project, so predicted_affinity is on a consistent, already-
validated target rather than a new one picked just for this experiment.
"""
import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Batch

from pubchem_lookup import load_cache as load_pubchem_cache
from train_ic50_model import embed_drug_graph
from train_ic50_fused_model import FusedIC50Model, reshape_gene_tokens, GENE_EMBEDDINGS
from infer_affinity import load_model as load_protliggnn, build_graph_pair

GDSC_CSV = Path("data") / "gdsc" / "GDSC2_fitted_dose_response.csv"
TARGET_SANGER_ID = "SIDM00237"
TARGET_CELL_LINE_NAME = "PC-14"
TARGET_PROTEIN_PDB = Path("data/pdbbind2020/pdbbind_subset/2011-2019/2l3r/2l3r_protein.pdb")

IC50_CHECKPOINT = Path("outputs/ic50_fused_model/best_crossattn_random.pt")
PROTLIGGNN_CHECKPOINT = Path("outputs/best_protliggnn.pt")

OUT_DIR = Path("outputs") / "cdps_prioritization"
CANDIDATES_CSV = OUT_DIR / "candidates.csv"
LIBRARY_CSV = OUT_DIR / "library.csv"
LIBRARY_SOURCE = Path("outputs") / "genai" / "pdbbind_ligands_smiles.csv"


def smiles_to_temp_sdf(smiles: str, output_path: Path) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
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


def load_gdsc_rows_for_cell_line(sanger_id: str) -> List[Dict[str, str]]:
    rows = []
    with GDSC_CSV.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("SANGER_MODEL_ID") == sanger_id:
                rows.append(row)
    return rows


def main():
    RDLogger.DisableLog("rdApp.*")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"target cell line: {TARGET_CELL_LINE_NAME} ({TARGET_SANGER_ID})")
    print(f"target protein: {TARGET_PROTEIN_PDB}")

    rows = load_gdsc_rows_for_cell_line(TARGET_SANGER_ID)
    print(f"GDSC2 rows for this cell line: {len(rows)}")

    smiles_cache = load_pubchem_cache()
    rows = [r for r in rows if smiles_cache.get(r["DRUG_NAME"])]
    print(f"rows with resolved SMILES: {len(rows)}")

    # Dedup by drug name (GDSC2 sometimes has repeat screens for the same drug).
    seen = {}
    for r in rows:
        seen.setdefault(r["DRUG_NAME"], r)
    rows = list(seen.values())
    print(f"unique drugs for this cell line: {len(rows)}")

    # --- load gene tokens for the target cell line (GNAE Drugnet input) ---
    gene_embeddings_raw = json.loads(GENE_EMBEDDINGS.read_text())
    if TARGET_SANGER_ID not in gene_embeddings_raw:
        raise RuntimeError(f"{TARGET_SANGER_ID} has no real gene-expression embedding; pick a different cell line.")
    gene_tokens = reshape_gene_tokens(gene_embeddings_raw[TARGET_SANGER_ID]).unsqueeze(0)  # [1, 8, 16]

    # --- load GNAE Drugnet (fused IC50 model) ---
    ic50_ckpt = torch.load(IC50_CHECKPOINT, map_location=device, weights_only=False)
    ic50_model = FusedIC50Model(hidden_dim=128, no_crossgraph=False, encoder="gat",
                                 gene_tokens="autoencoder_chunks").to(device)
    ic50_model.load_state_dict(ic50_ckpt["model_state_dict"])
    ic50_model.eval()
    print(f"loaded GNAE Drugnet checkpoint: {IC50_CHECKPOINT}")

    # --- load ProtLigGNN ---
    protliggnn_model = load_protliggnn(PROTLIGGNN_CHECKPOINT, device)
    print(f"loaded ProtLigGNN checkpoint: {PROTLIGGNN_CHECKPOINT}")

    out_rows = []
    skipped = {"ic50_graph": 0, "protliggnn_embed": 0, "ic50_score": 0, "protliggnn_score": 0}

    with tempfile.TemporaryDirectory(prefix="cdps_candidates_") as temp_dir:
        temp_root = Path(temp_dir)
        for idx, row in enumerate(rows, start=1):
            drug_name = row["DRUG_NAME"]
            smiles = smiles_cache[drug_name]
            observed_ln_ic50 = row.get("LN_IC50")
            if not observed_ln_ic50:
                continue

            # --- predicted_ic50 via GNAE Drugnet ---
            drug_graph = embed_drug_graph(smiles)
            if drug_graph is None:
                skipped["ic50_graph"] += 1
                continue
            try:
                with torch.no_grad():
                    drug_batch = Batch.from_data_list([drug_graph]).to(device)
                    predicted_ic50 = float(ic50_model(drug_batch, gene_tokens).item())
            except Exception as exc:
                skipped["ic50_score"] += 1
                print(f"  skip {drug_name}: GNAE Drugnet scoring failed: {exc}")
                continue

            # --- predicted_affinity via ProtLigGNN (against 2L3R) ---
            ligand_sdf = temp_root / f"{idx:04d}.sdf"
            if not smiles_to_temp_sdf(smiles, ligand_sdf):
                skipped["protliggnn_embed"] += 1
                continue
            try:
                ligand_graph, protein_graph = build_graph_pair(TARGET_PROTEIN_PDB, ligand_sdf)
                with torch.no_grad():
                    ligand_batch = Batch.from_data_list([ligand_graph]).to(device)
                    protein_batch = Batch.from_data_list([protein_graph]).to(device)
                    predicted_affinity = float(protliggnn_model(ligand_batch, protein_batch).item())
            except Exception as exc:
                skipped["protliggnn_score"] += 1
                print(f"  skip {drug_name}: ProtLigGNN scoring failed: {exc}")
                continue

            out_rows.append({
                "candidate_id": drug_name,
                "smiles": smiles,
                "predicted_ic50": predicted_ic50,
                "predicted_affinity": predicted_affinity,
                "observed_response": float(observed_ln_ic50),
            })

            if idx % 25 == 0 or idx == len(rows):
                print(f"scored {idx}/{len(rows)} drugs ({len(out_rows)} usable so far)")

    print(f"\nfinal usable candidates: {len(out_rows)}")
    print(f"skipped: {skipped}")

    with CANDIDATES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "smiles", "predicted_ic50",
                                                      "predicted_affinity", "observed_response"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"saved {CANDIDATES_CSV}")

    # --- library.csv: the same PDBbind ligand SMILES used for the GenAI novelty check ---
    lib_rows = list(csv.DictReader(LIBRARY_SOURCE.open("r", newline="", encoding="utf-8")))
    smiles_col = "smiles" if "smiles" in (lib_rows[0].keys() if lib_rows else []) else None
    if smiles_col is None:
        raise RuntimeError(f"Could not find a 'smiles' column in {LIBRARY_SOURCE}")
    with LIBRARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles"])
        for r in lib_rows:
            writer.writerow([r[smiles_col]])
    print(f"saved {LIBRARY_CSV} ({len(lib_rows)} known compounds)")


if __name__ == "__main__":
    main()
