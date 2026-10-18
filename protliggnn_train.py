import argparse
import csv
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser, is_aa
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdchem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from tqdm import tqdm


LIGAND_FEATURE_DIM = 78
PROTEIN_FEATURE_DIM = 30
AFFINITY_INDEX = "INDEX_general_PL.2020R1.lst"
AA_CODES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_CODES)}
AA_GROUPS = {
    "aromatic": {"PHE", "TYR", "TRP", "HIS"},
    "hydrophobic": {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"},
    "polar": {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS"},
    "positive": {"LYS", "ARG", "HIS"},
    "negative": {"ASP", "GLU"},
}
AA_MASS = {
    "ALA": 89.09,
    "ARG": 174.20,
    "ASN": 132.12,
    "ASP": 133.10,
    "CYS": 121.16,
    "GLN": 146.15,
    "GLU": 147.13,
    "GLY": 75.07,
    "HIS": 155.16,
    "ILE": 131.18,
    "LEU": 131.18,
    "LYS": 146.19,
    "MET": 149.21,
    "PHE": 165.19,
    "PRO": 115.13,
    "SER": 105.09,
    "THR": 119.12,
    "TRP": 204.23,
    "TYR": 181.19,
    "VAL": 117.15,
}
METALS = {
    3,
    4,
    11,
    12,
    13,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    55,
    56,
    57,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
}


@dataclass
class ComplexRecord:
    pdb_id: str
    affinity: float
    protein_path: Path
    ligand_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot_with_unknown(value, choices: Sequence) -> List[float]:
    encoded = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    encoded[idx] = 1.0
    return encoded


def safe_div(value: float, denom: float) -> float:
    return float(value) / float(denom) if denom else 0.0


def parse_affinity_to_pkd(binding_token: str) -> Optional[float]:
    token = binding_token.strip()
    match = re.match(r"([A-Za-z]+)([<>=]+)([0-9]*\.?[0-9]+)([fpnum]?M)", token)
    if not match:
        return None
    _, _, value_str, unit = match.groups()
    value = float(value_str)
    scale = {
        "fM": 1e-15,
        "pM": 1e-12,
        "nM": 1e-9,
        "uM": 1e-6,
        "mM": 1e-3,
        "M": 1.0,
    }.get(unit)
    if scale is None or value <= 0:
        return None
    molar = value * scale
    return -math.log10(molar)


def load_affinity_index(index_dir: Path) -> Dict[str, float]:
    index_path = index_dir / AFFINITY_INDEX
    if not index_path.exists():
        raise FileNotFoundError(f"Missing affinity index file: {index_path}")

    affinities: Dict[str, float] = {}
    with index_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb_id = parts[0].lower()
            affinity = parse_affinity_to_pkd(parts[3])
            if affinity is not None:
                affinities[pdb_id] = affinity
    return affinities


def find_complex_directories(data_dir: Path) -> Dict[str, Path]:
    root = data_dir / "pdbbind_subset"
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset folder: {root}")

    complex_dirs: Dict[str, Path] = {}
    for protein_path in root.rglob("*_protein.pdb"):
        pdb_id = protein_path.stem.replace("_protein", "").lower()
        complex_dirs[pdb_id] = protein_path.parent
    return complex_dirs


def select_ligand_file(complex_dir: Path, pdb_id: str) -> Optional[Path]:
    preferred = [
        complex_dir / f"{pdb_id}_ligand.sdf",
        complex_dir / f"{pdb_id}_ligand.mol2",
    ]
    for path in preferred:
        if path.exists():
            return path

    fallback_patterns = ["*_ligand.sdf", "*_ligand.mol2", "*.sdf", "*.mol2"]
    for pattern in fallback_patterns:
        matches = sorted(complex_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_complex_records(
    data_dir: Path,
    max_samples: Optional[int],
    exclude_ids: Optional[set] = None,
) -> List[ComplexRecord]:
    affinities = load_affinity_index(data_dir / "index")
    complex_dirs = find_complex_directories(data_dir)
    exclude_ids = exclude_ids or set()

    records: List[ComplexRecord] = []
    for pdb_id in sorted(complex_dirs):
        if pdb_id not in affinities:
            continue
        if pdb_id.lower() in exclude_ids:
            continue
        complex_dir = complex_dirs[pdb_id]
        protein_path = complex_dir / f"{pdb_id}_protein.pdb"
        ligand_path = select_ligand_file(complex_dir, pdb_id)
        if not protein_path.exists() or ligand_path is None:
            continue
        records.append(
            ComplexRecord(
                pdb_id=pdb_id,
                affinity=affinities[pdb_id],
                protein_path=protein_path,
                ligand_path=ligand_path,
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    return records


def load_ligand_molecule(ligand_path: Path) -> Optional[Chem.Mol]:
    suffix = ligand_path.suffix.lower()
    mol: Optional[Chem.Mol] = None
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=True)
        if len(supplier) > 0:
            mol = supplier[0]
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(ligand_path), removeHs=False, sanitize=True)
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        return None
    return mol


def atom_is_donor(atom: rdchem.Atom) -> float:
    if atom.GetAtomicNum() not in {7, 8, 15, 16}:
        return 0.0
    return 1.0 if atom.GetTotalNumHs(includeNeighbors=True) > 0 else 0.0


def atom_is_acceptor(atom: rdchem.Atom) -> float:
    atomic_num = atom.GetAtomicNum()
    if atomic_num == 7:
        return 1.0 if atom.GetFormalCharge() <= 0 else 0.0
    if atomic_num == 8:
        return 1.0 if atom.GetFormalCharge() <= 0 else 0.0
    if atomic_num in {9, 15, 16, 17, 35, 53}:
        return 1.0
    return 0.0


def atom_is_acidic(atom: rdchem.Atom) -> float:
    return 1.0 if atom.GetAtomicNum() == 8 and atom.GetFormalCharge() < 0 else 0.0


def atom_is_basic(atom: rdchem.Atom) -> float:
    return 1.0 if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0 else 0.0


def atom_features(atom: rdchem.Atom) -> List[float]:
    atomic_choices = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    degree_choices = list(range(0, 7))
    charge_choices = [-2, -1, 0, 1, 2]
    hybrid_choices = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    h_choices = [0, 1, 2, 3]
    valence_choices = list(range(0, 7))
    chirality_choices = [
        rdchem.ChiralType.CHI_UNSPECIFIED,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    ring_size_features = [
        1.0 if atom.IsInRingSize(size) else 0.0 for size in [3, 4, 5, 6, 7, 8]
    ]

    features: List[float] = []
    features.extend(one_hot_with_unknown(atom.GetAtomicNum(), atomic_choices))
    features.extend(one_hot_with_unknown(atom.GetDegree(), degree_choices))
    features.extend(one_hot_with_unknown(atom.GetFormalCharge(), charge_choices))
    features.extend(one_hot_with_unknown(atom.GetHybridization(), hybrid_choices))
    features.extend(one_hot_with_unknown(atom.GetTotalNumHs(includeNeighbors=True), h_choices))
    features.extend(one_hot_with_unknown(atom.GetImplicitValence(), valence_choices))
    features.extend(one_hot_with_unknown(atom.GetTotalValence(), valence_choices))
    features.extend(one_hot_with_unknown(atom.GetChiralTag(), chirality_choices))
    features.extend(ring_size_features)
    features.extend(
        [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            safe_div(atom.GetIsotope(), 100.0),
            safe_div(atom.GetMass(), 250.0),
            safe_div(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()), 3.0),
            safe_div(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()), 2.5),
            safe_div(atom.GetNumRadicalElectrons(), 4.0),
            float(atom.HasProp("_ChiralityPossible")),
            1.0 if atom.HasProp("_CIPCode") else 0.0,
            atom_is_donor(atom),
            atom_is_acceptor(atom),
            atom_is_acidic(atom),
            atom_is_basic(atom),
            1.0 if atom.GetAtomicNum() in {9, 17, 35, 53} else 0.0,
            1.0 if atom.GetAtomicNum() in METALS else 0.0,
            safe_div(atom.GetDegree(), 6.0),
        ]
    )
    if len(features) != LIGAND_FEATURE_DIM:
        raise ValueError(f"Unexpected ligand feature size: {len(features)}")
    return features


def mol_to_ligand_graph(mol: Chem.Mol) -> Data:
    conf = mol.GetConformer()
    node_features = []
    positions = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        node_features.append(atom_features(atom))
        pos = conf.GetAtomPosition(idx)
        positions.append([pos.x, pos.y, pos.z])

    edge_pairs: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_pairs.append((begin, end))
        edge_pairs.append((end, begin))

    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        pos=torch.tensor(positions, dtype=torch.float32),
    )


def residue_features(residue, residue_center: np.ndarray) -> List[float]:
    resname = residue.get_resname().strip().upper()
    one_hot = [0.0] * len(AA_CODES)
    if resname in AA_TO_INDEX:
        one_hot[AA_TO_INDEX[resname]] = 1.0

    atom_names = {atom.get_name().strip().upper() for atom in residue.get_atoms()}
    features = list(one_hot)
    features.extend(
        [
            1.0 if resname in AA_GROUPS["aromatic"] else 0.0,
            1.0 if resname in AA_GROUPS["hydrophobic"] else 0.0,
            1.0 if resname in AA_GROUPS["polar"] else 0.0,
            1.0 if resname in AA_GROUPS["positive"] else 0.0,
            1.0 if resname in AA_GROUPS["negative"] else 0.0,
            1.0 if resname == "GLY" else 0.0,
            1.0 if resname == "PRO" else 0.0,
            1.0 if "CA" in atom_names else 0.0,
            safe_div(AA_MASS.get(resname, 110.0), 250.0),
            safe_div(max(len(atom_names) - 4, 0), 10.0),
        ]
    )
    if len(features) != PROTEIN_FEATURE_DIM:
        raise ValueError(f"Unexpected protein feature size: {len(features)}")
    return features


def get_residue_atom_coords(residue) -> np.ndarray:
    coords = [atom.get_coord() for atom in residue.get_atoms() if atom.element != "H"]
    if not coords:
        coords = [atom.get_coord() for atom in residue.get_atoms()]
    if not coords:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def get_residue_center(residue) -> Optional[np.ndarray]:
    if "CA" in residue:
        return np.asarray(residue["CA"].get_coord(), dtype=np.float32)
    coords = get_residue_atom_coords(residue)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def build_protein_pocket_graph(protein_path: Path, ligand_positions: np.ndarray) -> Data:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_path.stem, str(protein_path))

    pocket_residues = []
    residue_centers = []
    node_features = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                atom_coords = get_residue_atom_coords(residue)
                if atom_coords.size == 0:
                    continue
                distances = np.linalg.norm(
                    atom_coords[:, None, :] - ligand_positions[None, :, :], axis=-1
                )
                if float(distances.min()) > 6.0:
                    continue
                center = get_residue_center(residue)
                if center is None:
                    continue
                pocket_residues.append(residue)
                residue_centers.append(center)
                node_features.append(residue_features(residue, center))

    if not pocket_residues:
        raise ValueError("No pocket residues found within 6A of ligand")

    coords = np.asarray(residue_centers, dtype=np.float32)
    edge_pairs: List[Tuple[int, int]] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance <= 8.0:
                edge_pairs.append((i, j))
                edge_pairs.append((j, i))
    if not edge_pairs and len(coords) > 1:
        for i in range(len(coords) - 1):
            edge_pairs.append((i, i + 1))
            edge_pairs.append((i + 1, i))

    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        pos=torch.tensor(coords, dtype=torch.float32),
    )


def process_complex(record: ComplexRecord) -> Tuple[Optional[Tuple[Data, Data, float, str]], Optional[str]]:
    try:
        ligand_mol = load_ligand_molecule(record.ligand_path)
        if ligand_mol is None:
            return None, "ligand unreadable or missing 3D conformer"
        ligand_graph = mol_to_ligand_graph(ligand_mol)
        protein_graph = build_protein_pocket_graph(
            record.protein_path,
            ligand_graph.pos.numpy(),
        )
        return (ligand_graph, protein_graph, record.affinity, record.pdb_id), None
    except Exception as exc:  # pragma: no cover - robustness path
        return None, str(exc)


class PDBbindPairDataset(Dataset):
    def __init__(self, records: Sequence[ComplexRecord]):
        self.records = list(records)
        self.samples: List[Tuple[Data, Data, float, str]] = []
        self.skipped: List[Tuple[str, str]] = []

        for record in tqdm(self.records, desc="Processing complexes"):
            sample, error = process_complex(record)
            if sample is None:
                self.skipped.append((record.pdb_id, error or "unknown error"))
            else:
                self.samples.append(sample)

        print(
            f"Successfully processed {len(self.samples)} complexes; "
            f"skipped {len(self.skipped)} complexes."
        )
        if self.skipped:
            preview = ", ".join(f"{pdb_id} ({reason})" for pdb_id, reason in self.skipped[:5])
            print(f"Skip examples: {preview}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_pairs(batch):
    ligand_graphs = [item[0] for item in batch]
    protein_graphs = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    pdb_ids = [item[3] for item in batch]
    return (
        Batch.from_data_list(ligand_graphs),
        Batch.from_data_list(protein_graphs),
        labels,
        pdb_ids,
    )


class LigandEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // 4, heads=4, dropout=0.1)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=0.1)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        x = self.norm1(F.elu(self.conv1(x, edge_index)))
        x = self.norm2(F.elu(self.conv2(x, edge_index)))
        x = self.norm3(F.elu(self.conv3(x, edge_index)))
        return x


class ProteinEncoder(nn.Module):
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


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        # Ligand atoms attend over protein residues to pull pocket context.
        self.ligand_to_protein_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        # Protein residues also attend back over ligand atoms so the interaction is symmetric.
        self.protein_to_ligand_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.ligand_norm = nn.LayerNorm(hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)

    def forward(self, ligand_x: torch.Tensor, protein_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # MultiheadAttention expects [batch, seq, dim]. Each protein-ligand pair is handled
        # as a single sequence batch, so we temporarily add a batch dimension.
        ligand_seq = ligand_x.unsqueeze(0)
        protein_seq = protein_x.unsqueeze(0)

        ligand_context, _ = self.ligand_to_protein_attn(
            query=ligand_seq,
            key=protein_seq,
            value=protein_seq,
            need_weights=False,
        )
        protein_context, _ = self.protein_to_ligand_attn(
            query=protein_seq,
            key=ligand_seq,
            value=ligand_seq,
            need_weights=False,
        )

        # Residual connections keep the original node features available while the attention
        # layers add cross-graph context from the opposite molecular partner.
        ligand_context = self.ligand_norm((ligand_context + ligand_seq).squeeze(0))
        protein_context = self.protein_norm((protein_context + protein_seq).squeeze(0))
        return ligand_context, protein_context


class ProtLigGNN(nn.Module):
    def __init__(self, ligand_dim: int, protein_dim: int, hidden_dim: int = 256, no_crossgraph: bool = False):
        super().__init__()
        self.no_crossgraph = no_crossgraph
        self.hidden_dim = hidden_dim
        self.ligand_encoder = LigandEncoder(ligand_dim, hidden_dim)
        self.protein_encoder = ProteinEncoder(protein_dim, hidden_dim)
        self.cross_attention = BidirectionalCrossAttention(hidden_dim=hidden_dim, num_heads=4)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def cross_graph_interaction(self, ligand_x, protein_x):
        if self.no_crossgraph:
            zero_l = torch.zeros_like(ligand_x)
            zero_p = torch.zeros_like(protein_x)
            return torch.cat([ligand_x, zero_l], dim=-1), torch.cat([protein_x, zero_p], dim=-1)

        ligand_context, protein_context = self.cross_attention(ligand_x, protein_x)
        return (
            torch.cat([ligand_x, ligand_context], dim=-1),
            torch.cat([protein_x, protein_context], dim=-1),
        )

    def forward(self, ligand_batch: Batch, protein_batch: Batch):
        ligand_x = self.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = self.protein_encoder(protein_batch.x, protein_batch.edge_index)

        ligand_chunks = []
        protein_chunks = []
        batch_size = ligand_batch.num_graphs
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            protein_mask = protein_batch.batch == idx
            ligand_chunk, protein_chunk = self.cross_graph_interaction(
                ligand_x[ligand_mask],
                protein_x[protein_mask],
            )
            ligand_chunks.append(ligand_chunk)
            protein_chunks.append(protein_chunk)

        ligand_joint = torch.cat(ligand_chunks, dim=0)
        protein_joint = torch.cat(protein_chunks, dim=0)
        ligand_pool = global_mean_pool(ligand_joint, ligand_batch.batch)
        protein_pool = global_mean_pool(protein_joint, protein_batch.batch)
        joint = self.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
        return self.regressor(joint).squeeze(-1)


def split_indices(num_samples: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if num_samples == 1:
        return indices, indices, indices
    if num_samples == 2:
        return [indices[0]], [indices[0]], [indices[1]]
    if num_samples < 10:
        return indices[:-2], [indices[-2]], [indices[-1]]

    train_size = max(1, int(0.8 * num_samples))
    val_size = max(1, int(0.1 * num_samples))
    test_size = num_samples - train_size - val_size
    if test_size < 1:
        test_size = 1
        if train_size > val_size:
            train_size -= 1
        else:
            val_size -= 1

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]
    return train_idx, val_idx, test_idx


def make_loader(samples, indices, batch_size: int, shuffle: bool) -> Optional[DataLoader]:
    if not indices:
        return None
    subset = [samples[idx] for idx in indices]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pairs)


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    predictions = []
    targets = []
    pdb_ids = []

    for ligand_batch, protein_batch, labels, batch_pdb_ids in loader:
        ligand_batch = ligand_batch.to(device)
        protein_batch = protein_batch.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            preds = model(ligand_batch, protein_batch)
            loss = F.mse_loss(preds, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        predictions.append(preds.detach().cpu())
        targets.append(labels.detach().cpu())
        pdb_ids.extend(batch_pdb_ids)

    if total_count == 0:
        return {"loss": float("nan"), "y_true": np.array([]), "y_pred": np.array([]), "pdb_ids": []}

    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(predictions).numpy()
    return {
        "loss": total_loss / total_count,
        "y_true": y_true,
        "y_pred": y_pred,
        "pdb_ids": pdb_ids,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"pcc": float("nan"), "spearman": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    if len(y_true) < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        pcc = float("nan")
        spearman = float("nan")
    else:
        pcc = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)

    return {"pcc": pcc, "spearman": spearman, "rmse": rmse, "mae": mae}


def save_checkpoint(model: nn.Module, output_path: Path, args, stats: Dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "stats": stats,
        },
        output_path,
    )


def format_metrics(prefix: str, loss: float, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix} loss={loss:.4f} "
        f"PCC={metrics['pcc']:.4f} "
        f"Spearman={metrics['spearman']:.4f} "
        f"RMSE={metrics['rmse']:.4f} "
        f"MAE={metrics['mae']:.4f}"
    )


def save_epoch_history(history: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_pcc",
        "train_spearman",
        "train_rmse",
        "train_mae",
        "val_loss",
        "val_pcc",
        "val_spearman",
        "val_rmse",
        "val_mae",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_test_predictions(pdb_ids: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdb_id", "true_affinity", "predicted_affinity", "error"])
        for pdb_id, true_value, pred_value in zip(pdb_ids, y_true, y_pred):
            writer.writerow([pdb_id, float(true_value), float(pred_value), float(pred_value - true_value)])


def save_scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, run_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="none")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.0, color="tab:red")
    plt.xlabel("True affinity")
    plt.ylabel("Predicted affinity")
    plt.title(f"Test Predictions: {run_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_training_curves(history: List[Dict[str, float]], output_path: Path, run_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_rmse = [row["train_rmse"] for row in history]
    val_rmse = [row["val_rmse"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(epochs, train_rmse, label="train")
    axes[1].plot(epochs, val_rmse, label="val")
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()

    fig.suptitle(f"Training Curves: {run_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ProtLigGNN on PDBbind v2020.")
    parser.add_argument("--data_dir", type=str, default="data/pdbbind2020")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--run_name", type=str, default="protliggnn_run")
    parser.add_argument("--no_crossgraph", action="store_true")
    parser.add_argument("--save_path", type=str, default="outputs/best_protliggnn.pt")
    parser.add_argument(
        "--exclude_ids_file",
        type=str,
        default=None,
        help="Path to a text file of PDB IDs (one per line) to exclude from training, e.g. an external benchmark core set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_ids = None
    if args.exclude_ids_file:
        exclude_path = Path(args.exclude_ids_file)
        exclude_ids = {line.strip().lower() for line in exclude_path.read_text().splitlines() if line.strip()}
        print(f"Excluding {len(exclude_ids)} PDB IDs listed in {exclude_path.resolve()}")

    print(f"Loading dataset from {data_dir.resolve()}")
    records = discover_complex_records(data_dir, args.max_samples, exclude_ids=exclude_ids)
    print(f"Found {len(records)} candidate complexes with labels and matching files before validation.")
    if not records:
        raise RuntimeError("No candidate complexes were found. Check --data_dir and index files.")

    dataset = PDBbindPairDataset(records)
    if len(dataset) == 0:
        raise RuntimeError("No complexes could be processed successfully.")

    train_idx, val_idx, test_idx = split_indices(len(dataset), args.seed)
    print(
        f"Dataset split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )

    train_loader = make_loader(dataset.samples, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(dataset.samples, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(dataset.samples, test_idx, args.batch_size, shuffle=False)

    model = ProtLigGNN(
        ligand_dim=LIGAND_FEATURE_DIM,
        protein_dim=PROTEIN_FEATURE_DIM,
        no_crossgraph=args.no_crossgraph,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("inf")
    best_stats = {}
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_out = run_epoch(model, train_loader, device, optimizer=optimizer)
        train_metrics = compute_metrics(train_out["y_true"], train_out["y_pred"])

        eval_loader = val_loader if val_loader is not None else train_loader
        val_out = run_epoch(model, eval_loader, device, optimizer=None)
        val_metrics = compute_metrics(val_out["y_true"], val_out["y_pred"])

        print(format_metrics(f"Epoch {epoch:03d} train", train_out["loss"], train_metrics))
        print(format_metrics(f"Epoch {epoch:03d} val  ", val_out["loss"], val_metrics))
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_out["loss"],
                "train_pcc": train_metrics["pcc"],
                "train_spearman": train_metrics["spearman"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "val_loss": val_out["loss"],
                "val_pcc": val_metrics["pcc"],
                "val_spearman": val_metrics["spearman"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
            }
        )

        score = val_metrics["rmse"] if not math.isnan(val_metrics["rmse"]) else val_out["loss"]
        if score < best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_stats = {
                "epoch": epoch,
                "train_loss": train_out["loss"],
                "val_loss": val_out["loss"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            save_checkpoint(model, Path(args.save_path), args, best_stats)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['stats']['epoch']} at {Path(args.save_path).resolve()}")

    test_out = run_epoch(model, test_loader, device, optimizer=None)
    test_metrics = compute_metrics(test_out["y_true"], test_out["y_pred"])
    print(format_metrics("Test", test_out["loss"], test_metrics))

    save_epoch_history(epoch_history, output_dir / f"epoch_history_{args.run_name}.csv")
    save_test_predictions(
        test_out["pdb_ids"],
        test_out["y_true"],
        test_out["y_pred"],
        output_dir / f"test_predictions_{args.run_name}.csv",
    )
    save_scatter_plot(
        test_out["y_true"],
        test_out["y_pred"],
        output_dir / f"scatter_{args.run_name}.png",
        args.run_name,
    )
    save_training_curves(
        epoch_history,
        output_dir / f"training_curves_{args.run_name}.png",
        args.run_name,
    )


if __name__ == "__main__":
    main()
