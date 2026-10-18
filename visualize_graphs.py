import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from protliggnn_train import (
    AA_TO_INDEX,
    build_protein_pocket_graph,
    load_ligand_molecule,
    mol_to_ligand_graph,
)


OUTPUT_DIR = Path("outputs") / "graph_visualizations"
IDX_TO_AA = {idx: name for name, idx in AA_TO_INDEX.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the ligand graph and protein pocket graph for one complex.")
    parser.add_argument("--protein_pdb", type=str, required=True)
    parser.add_argument("--ligand_file", type=str, required=True)
    parser.add_argument("--tag", type=str, default="example")
    return parser.parse_args()


def residue_name_from_features(feature_row: np.ndarray) -> str:
    one_hot = feature_row[: len(AA_TO_INDEX)]
    idx = int(np.argmax(one_hot))
    return IDX_TO_AA.get(idx, "UNK")


def draw_ligand_2d(mol: Chem.Mol, out_path: Path):
    # A clean skeletal depiction for the paper: strip explicit Hs and indices.
    flat_mol = Chem.RemoveHs(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(700, 600)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, flat_mol)
    drawer.FinishDrawing()
    out_path.write_bytes(drawer.GetDrawingText())


def plot_ligand_graph_3d(ligand_graph, mol: Chem.Mol, out_path: Path):
    pos = ligand_graph.pos.numpy()
    edge_index = ligand_graph.edge_index.numpy()
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]

    element_colors = {"C": "#4c4c4c", "N": "#3050f8", "O": "#ff0d0d", "S": "#ffff30", "F": "#90e050", "Cl": "#1ff01f"}

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    drawn_pairs = set()
    for start, end in zip(edge_index[0], edge_index[1]):
        pair = tuple(sorted((int(start), int(end))))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)
        ax.plot(
            [pos[start, 0], pos[end, 0]],
            [pos[start, 1], pos[end, 1]],
            [pos[start, 2], pos[end, 2]],
            color="#888888",
            linewidth=1.5,
        )

    for i, elem in enumerate(elements):
        color = element_colors.get(elem, "#cc44cc")
        ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], color=color, s=120, edgecolors="black", linewidths=0.5)

    ax.set_title(f"Ligand Molecular Graph\n({len(elements)} atom nodes, {len(drawn_pairs)} bond edges)")
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (A)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pocket_graph_3d(protein_graph, ligand_pos: np.ndarray, out_path: Path):
    pos = protein_graph.pos.numpy()
    edge_index = protein_graph.edge_index.numpy()
    features = protein_graph.x.numpy()
    res_names = [residue_name_from_features(row) for row in features]

    group_colors = {
        "aromatic": "#e69f00",
        "hydrophobic": "#999999",
        "polar": "#56b4e9",
        "positive": "#0072b2",
        "negative": "#d55e00",
    }
    aromatic = {"PHE", "TRP", "TYR", "HIS"}
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "GLY"}
    polar = {"SER", "THR", "ASN", "GLN", "CYS"}
    positive = {"LYS", "ARG", "HIS"}
    negative = {"ASP", "GLU"}

    def color_for(resname: str) -> str:
        if resname in aromatic:
            return group_colors["aromatic"]
        if resname in positive:
            return group_colors["positive"]
        if resname in negative:
            return group_colors["negative"]
        if resname in polar:
            return group_colors["polar"]
        return group_colors["hydrophobic"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    drawn_pairs = set()
    for start, end in zip(edge_index[0], edge_index[1]):
        pair = tuple(sorted((int(start), int(end))))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)
        ax.plot(
            [pos[start, 0], pos[end, 0]],
            [pos[start, 1], pos[end, 1]],
            [pos[start, 2], pos[end, 2]],
            color="#cccccc",
            linewidth=0.8,
            alpha=0.6,
        )

    for i, resname in enumerate(res_names):
        ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], color=color_for(resname), s=140, edgecolors="black", linewidths=0.5)
        ax.text(pos[i, 0], pos[i, 1], pos[i, 2], resname, fontsize=6)

    ax.scatter(
        ligand_pos[:, 0], ligand_pos[:, 1], ligand_pos[:, 2],
        color="red", marker="*", s=60, label="ligand atoms", alpha=0.8,
    )

    ax.set_title(
        f"Protein Pocket Graph\n({len(res_names)} residue nodes within 6A of ligand, {len(drawn_pairs)} edges within 8A)"
    )
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (A)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    protein_pdb = Path(args.protein_pdb)
    ligand_file = Path(args.ligand_file)

    ligand_mol = load_ligand_molecule(ligand_file)
    if ligand_mol is None:
        raise ValueError(f"Unable to load ligand: {ligand_file}")
    ligand_graph = mol_to_ligand_graph(ligand_mol)
    protein_graph = build_protein_pocket_graph(protein_pdb, ligand_graph.pos.numpy())

    draw_ligand_2d(ligand_mol, OUTPUT_DIR / f"{args.tag}_ligand_2d.png")
    plot_ligand_graph_3d(ligand_graph, ligand_mol, OUTPUT_DIR / f"{args.tag}_ligand_graph_3d.png")
    plot_pocket_graph_3d(protein_graph, ligand_graph.pos.numpy(), OUTPUT_DIR / f"{args.tag}_pocket_graph_3d.png")

    print(f"ligand atoms (nodes): {ligand_graph.x.shape[0]}")
    print(f"ligand bonds (edges, undirected): {ligand_graph.edge_index.shape[1] // 2}")
    print(f"pocket residues (nodes): {protein_graph.x.shape[0]}")
    print(f"pocket edges (undirected): {protein_graph.edge_index.shape[1] // 2}")
    print(f"saved visualizations to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
