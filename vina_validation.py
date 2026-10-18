import argparse
import csv
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


DEFAULT_GENERATED_CSV = Path("outputs") / "genai" / "top10_generated_candidates_by_gaps.csv"
DEFAULT_CHECKPOINT = Path("outputs") / "best_protliggnn.pt"
LOCAL_DEPS_DIR = Path("python_deps")


def parse_args():
    parser = argparse.ArgumentParser(description="Dock ProtLigGNN and GenAI ligands with AutoDock Vina.")
    parser.add_argument("--protein_pdb", type=str, required=True)
    parser.add_argument("--ligand_smiles", type=str, default=None)
    parser.add_argument("--generated_csv", type=str, default=str(DEFAULT_GENERATED_CSV))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="outputs/vina_validation")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def run_command(command: Sequence[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def resolve_tool(tool_name: str, fallback_paths: Sequence[Path]) -> str:
    path = shutil.which(tool_name)
    if path:
        return path
    for fallback in fallback_paths:
        if fallback.exists():
            return str(fallback)
    raise FileNotFoundError(f"{tool_name} was not found on PATH or known Windows install locations.")


def require_tool(tool_command: str, display_name: str, version_flag: str = "--version"):
    try:
        result = run_command([tool_command, version_flag], timeout=20)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{display_name} was not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{display_name} did not respond to {version_flag} within the timeout.") from exc

    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{display_name} check failed: {message}")


def load_protliggnn_helpers():
    if LOCAL_DEPS_DIR.exists():
        sys.path.append(str(LOCAL_DEPS_DIR.resolve()))

    from torch_geometric.data import Batch
    from infer_affinity import build_graph_pair, load_model

    return Batch, build_graph_pair, load_model


def smiles_to_3d_sdf(smiles: str, sdf_path: Path) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"RDKit 3D embedding failed for SMILES: {smiles}")

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    mol.SetProp("_Name", smiles)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return mol


def convert_with_obabel(
    obabel_command: str,
    input_path: Path,
    output_path: Path,
    extra_args: Optional[Sequence[str]] = None,
):
    command = [obabel_command, str(input_path), "-O", str(output_path)]
    if extra_args:
        command.extend(extra_args)

    result = run_command(command, timeout=300)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Open Babel conversion failed for {input_path}: {message}")


def parse_pdb_coordinates(protein_pdb: Path) -> List[Tuple[float, float, float]]:
    coords = []
    with protein_pdb.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    if not coords:
        raise ValueError(f"No atom coordinates found in protein PDB: {protein_pdb}")
    return coords


def docking_box_from_protein(protein_pdb: Path) -> Dict[str, float]:
    # Without an explicit co-crystal ligand or box arguments, use a conservative
    # blind-docking box around the protein coordinates.
    coords = parse_pdb_coordinates(protein_pdb)
    xs, ys, zs = zip(*coords)
    center = {
        "center_x": (min(xs) + max(xs)) / 2.0,
        "center_y": (min(ys) + max(ys)) / 2.0,
        "center_z": (min(zs) + max(zs)) / 2.0,
    }
    size = {
        "size_x": min(max(max(xs) - min(xs) + 8.0, 18.0), 40.0),
        "size_y": min(max(max(ys) - min(ys) + 8.0, 18.0), 40.0),
        "size_z": min(max(max(zs) - min(zs) + 8.0, 18.0), 40.0),
    }
    return {**center, **size}


def run_vina(
    vina_command: str,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    pose_path: Path,
    log_path: Path,
    box: Dict[str, float],
) -> float:
    command = [
        vina_command,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        f"{box['center_x']:.3f}",
        "--center_y",
        f"{box['center_y']:.3f}",
        "--center_z",
        f"{box['center_z']:.3f}",
        "--size_x",
        f"{box['size_x']:.3f}",
        "--size_y",
        f"{box['size_y']:.3f}",
        "--size_z",
        f"{box['size_z']:.3f}",
        "--out",
        str(pose_path),
    ]
    result = run_command(command, timeout=600)
    log_text = "\n".join(part for part in [result.stdout, result.stderr] if part)
    if log_text:
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")

    if result.returncode != 0:
        raise RuntimeError(f"Vina failed: {(result.stderr or result.stdout).strip()}")

    score = parse_vina_score(log_text)
    if score is None and pose_path.exists():
        score = parse_vina_score(pose_path.read_text(encoding="utf-8", errors="ignore"))
    if score is None and log_path.exists():
        score = parse_vina_score(log_path.read_text(encoding="utf-8", errors="ignore"))
    if score is None:
        raise RuntimeError("Vina completed but no docking score could be parsed.")
    return score


def parse_vina_score(text: str) -> Optional[float]:
    remark = re.search(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", text)
    if remark:
        return float(remark.group(1))

    table_row = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", text, flags=re.MULTILINE)
    if table_row:
        return float(table_row.group(1))

    return None


def run_protliggnn_affinity(protein_pdb: Path, ligand_sdf: Path, checkpoint: Path, device: torch.device) -> float:
    batch_cls, build_graph_pair, load_model = load_protliggnn_helpers()
    model = load_model(checkpoint, device)
    ligand_graph, protein_graph = build_graph_pair(protein_pdb, ligand_sdf)
    ligand_batch = batch_cls.from_data_list([ligand_graph]).to(device)
    protein_batch = batch_cls.from_data_list([protein_graph]).to(device)
    with torch.no_grad():
        return float(model(ligand_batch, protein_batch).item())


PDBQT_ALLOWED_PREFIXES = (
    "ROOT",
    "ENDROOT",
    "BRANCH",
    "ENDBRANCH",
    "TORSDOF",
    "ATOM",
    "HETATM",
    "REMARK",
    "TER",
    "MODEL",
    "ENDMDL",
)


def strip_non_pdbqt_records(pdbqt_path: Path):
    # Open Babel carries PDB metadata (HEADER/COMPND/SEQRES/...) into the PDBQT
    # output; modern Vina's stricter parser rejects those unknown record types.
    lines = pdbqt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    kept = [line for line in lines if line.startswith(PDBQT_ALLOWED_PREFIXES)]
    pdbqt_path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def prepare_receptor(obabel_command: str, protein_pdb: Path, out_dir: Path) -> Path:
    receptor_pdbqt = out_dir / f"{protein_pdb.stem}_receptor.pdbqt"
    if not receptor_pdbqt.exists() or receptor_pdbqt.stat().st_size == 0:
        convert_with_obabel(obabel_command, protein_pdb, receptor_pdbqt, extra_args=["-xr"])
        strip_non_pdbqt_records(receptor_pdbqt)
    return receptor_pdbqt


def dock_smiles(
    smiles: str,
    label: str,
    protein_pdb: Path,
    receptor_pdbqt: Path,
    obabel_command: str,
    vina_command: str,
    box: Dict[str, float],
    work_dir: Path,
    poses_dir: Path,
) -> Dict[str, object]:
    sdf_path = work_dir / f"{label}.sdf"
    ligand_pdbqt = work_dir / f"{label}.pdbqt"
    pose_path = poses_dir / f"{label}_vina_pose.pdbqt"
    log_path = work_dir / f"{label}_vina.log"

    # Workflow: RDKit builds a 3D ligand, Open Babel writes PDBQT, and Vina docks
    # that ligand into the prepared receptor using the inferred protein box.
    smiles_to_3d_sdf(smiles, sdf_path)
    convert_with_obabel(obabel_command, sdf_path, ligand_pdbqt)
    vina_score = run_vina(vina_command, receptor_pdbqt, ligand_pdbqt, pose_path, log_path, box)

    return {
        "sdf_path": str(sdf_path),
        "ligand_pdbqt": str(ligand_pdbqt),
        "pose_path": str(pose_path),
        "vina_log": str(log_path),
        "vina_score": vina_score,
    }


def save_single_outputs(out_dir: Path, payload: Dict[str, object], summary_text: str):
    (out_dir / "vina_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "docking_summary.txt").write_text(summary_text, encoding="utf-8")


def validate_single_ligand(
    args,
    protein_pdb: Path,
    out_dir: Path,
    receptor_pdbqt: Path,
    obabel_command: str,
    vina_command: str,
    box: Dict[str, float],
):
    if not args.ligand_smiles:
        return

    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    poses_dir = out_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vina_single_", dir=str(out_dir)) as temp_dir:
        work_dir = Path(temp_dir)
        docking = dock_smiles(
            args.ligand_smiles,
            "single_ligand",
            protein_pdb,
            receptor_pdbqt,
            obabel_command,
            vina_command,
            box,
            work_dir,
            poses_dir,
        )
        protliggnn_affinity = run_protliggnn_affinity(
            protein_pdb=protein_pdb,
            ligand_sdf=Path(docking["sdf_path"]),
            checkpoint=checkpoint,
            device=device,
        )

    payload = {
        "mode": "single_ligand",
        "protein_pdb": str(protein_pdb),
        "ligand_smiles": args.ligand_smiles,
        "protliggnn_predicted_affinity": protliggnn_affinity,
        "vina_score": docking["vina_score"],
        "pose_path": docking["pose_path"],
        "docking_box": box,
    }
    summary = "\n".join(
        [
            "Single Ligand Docking Validation",
            "================================",
            f"Protein PDB: {protein_pdb}",
            f"Ligand SMILES: {args.ligand_smiles}",
            f"ProtLigGNN affinity: {protliggnn_affinity:.4f}",
            f"Vina docking score: {float(docking['vina_score']):.4f} kcal/mol",
            f"Pose file: {docking['pose_path']}",
            "",
            "Interpretation note: ProtLigGNN affinity and Vina score are different model outputs.",
            "Vina scores are binding-energy-like values where more negative is usually better.",
        ]
    )
    save_single_outputs(out_dir, payload, summary + "\n")
    print(summary)


def read_generated_candidates(path: Path, top_k: int) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Generated candidate CSV not found: {path}")

    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("smiles"):
                rows.append(row)
            if len(rows) >= top_k:
                break
    return rows


def save_generated_results(rows: Sequence[Dict[str, object]], path: Path):
    fieldnames = ["smiles", "predicted_affinity", "gaps_score", "vina_score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_generated_batch(
    args,
    protein_pdb: Path,
    out_dir: Path,
    receptor_pdbqt: Path,
    obabel_command: str,
    vina_command: str,
    box: Dict[str, float],
):
    generated_csv = Path(args.generated_csv)
    candidates = read_generated_candidates(generated_csv, args.top_k)
    if not candidates:
        raise ValueError(f"No generated candidates found in: {generated_csv}")

    poses_dir = out_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []

    with tempfile.TemporaryDirectory(prefix="vina_generated_", dir=str(out_dir)) as temp_dir:
        work_dir = Path(temp_dir)
        for index, candidate in enumerate(candidates, start=1):
            smiles = candidate["smiles"]
            label = f"generated_{index:03d}"
            try:
                docking = dock_smiles(
                    smiles,
                    label,
                    protein_pdb,
                    receptor_pdbqt,
                    obabel_command,
                    vina_command,
                    box,
                    work_dir,
                    poses_dir,
                )
                results.append(
                    {
                        "smiles": smiles,
                        "predicted_affinity": candidate.get("predicted_affinity", ""),
                        "gaps_score": candidate.get("gaps", candidate.get("gaps_score", "")),
                        "vina_score": docking["vina_score"],
                    }
                )
                print(f"docked {index}/{len(candidates)}: Vina score {float(docking['vina_score']):.4f}")
            except Exception as exc:
                message = f"FAILED {index}/{len(candidates)} {smiles}: {exc}"
                failures.append(message)
                print(message)

    results_path = out_dir / "generated_molecule_docking_results.csv"
    save_generated_results(results, results_path)
    save_scatter_plot(results, out_dir / "vina_vs_affinity_scatter.png")
    save_generated_summary(results, failures, out_dir / "generated_docking_summary.txt", generated_csv, args.top_k)


def save_generated_summary(
    results: Sequence[Dict[str, object]],
    failures: Sequence[str],
    summary_path: Path,
    generated_csv: Path,
    top_k: int,
):
    by_affinity = sorted(results, key=lambda row: safe_float(row["predicted_affinity"]), reverse=True)
    by_gaps = sorted(results, key=lambda row: safe_float(row["gaps_score"]), reverse=True)
    by_vina = sorted(results, key=lambda row: safe_float(row["vina_score"]))

    lines = [
        "Generated Molecule Docking Summary",
        "==================================",
        f"Input CSV: {generated_csv}",
        f"Requested top_k: {top_k}",
        f"Docked successfully: {len(results)}",
        f"Failures: {len(failures)}",
        "",
        "Rank by ProtLigGNN affinity:",
    ]
    lines.extend(format_rank_lines(by_affinity, "predicted_affinity", reverse_label=True))
    lines.append("")
    lines.append("Rank by GAPS:")
    lines.extend(format_rank_lines(by_gaps, "gaps_score", reverse_label=True))
    lines.append("")
    lines.append("Rank by Vina docking score:")
    lines.extend(format_rank_lines(by_vina, "vina_score", reverse_label=False))

    if failures:
        lines.append("")
        lines.append("Failures:")
        lines.extend(failures)

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_rank_lines(rows: Sequence[Dict[str, object]], key: str, reverse_label: bool) -> List[str]:
    direction = "higher is better" if reverse_label else "more negative is better"
    lines = [f"({direction})"]
    for idx, row in enumerate(rows, start=1):
        lines.append(f"{idx}. {row['smiles']} | {key}={safe_float(row[key]):.4f}")
    return lines


def save_scatter_plot(rows: Sequence[Dict[str, object]], path: Path):
    width, height = 800, 520
    margin_left, margin_right = 75, 35
    margin_top, margin_bottom = 35, 70
    image = bytearray([255, 255, 255] * width * height)

    def set_pixel(x: int, y: int, color: Tuple[int, int, int]):
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            image[offset : offset + 3] = bytes(color)

    def draw_line(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]):
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def draw_circle(cx: int, cy: int, radius: int, color: Tuple[int, int, int]):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    set_pixel(x, y, color)

    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom
    draw_line(plot_left, plot_top, plot_left, plot_bottom, (45, 45, 45))
    draw_line(plot_left, plot_bottom, plot_right, plot_bottom, (45, 45, 45))
    for i in range(6):
        y = plot_top + round((plot_bottom - plot_top) * i / 5)
        draw_line(plot_left, y, plot_right, y, (228, 228, 228))

    points = [(safe_float(row["predicted_affinity"]), safe_float(row["vina_score"])) for row in rows]
    if points:
        xs, ys = zip(*points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if math.isclose(min_x, max_x):
            min_x -= 1.0
            max_x += 1.0
        if math.isclose(min_y, max_y):
            min_y -= 1.0
            max_y += 1.0

        for affinity, vina_score in points:
            x = plot_left + round((affinity - min_x) / (max_x - min_x) * (plot_right - plot_left))
            y = plot_bottom - round((vina_score - min_y) / (max_y - min_y) * (plot_bottom - plot_top))
            draw_circle(x, y, 5, (31, 119, 180))

    raw_rows = []
    for y in range(height):
        start = y * width * 3
        raw_rows.append(b"\x00" + bytes(image[start : start + width * 3]))
    compressed = zlib.compress(b"".join(raw_rows), level=9)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")

    protein_pdb = Path(args.protein_pdb)
    out_dir = Path(args.out_dir)
    if not protein_pdb.exists():
        raise FileNotFoundError(f"Protein PDB not found: {protein_pdb}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "poses").mkdir(parents=True, exist_ok=True)

    obabel_command = resolve_tool(
        "obabel",
        [Path("C:/Program Files/OpenBabel-2.4.1/obabel.exe")],
    )
    vina_command = resolve_tool(
        "vina",
        [Path("C:/Program Files (x86)/The Scripps Research Institute/Vina/vina.exe")],
    )
    require_tool(obabel_command, "Open Babel", version_flag="-V")
    require_tool(vina_command, "AutoDock Vina")

    receptor_pdbqt = prepare_receptor(obabel_command, protein_pdb, out_dir)
    box = docking_box_from_protein(protein_pdb)

    if args.ligand_smiles:
        validate_single_ligand(args, protein_pdb, out_dir, receptor_pdbqt, obabel_command, vina_command, box)
    else:
        validate_generated_batch(args, protein_pdb, out_dir, receptor_pdbqt, obabel_command, vina_command, box)


if __name__ == "__main__":
    main()
