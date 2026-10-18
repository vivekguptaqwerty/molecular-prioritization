import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rdkit import RDLogger

from vina_validation import (
    convert_with_obabel,
    docking_box_from_protein,
    dock_smiles,
    prepare_receptor,
    require_tool,
    resolve_tool,
    run_vina,
)


DATA_ROOT = Path("data") / "pdbbind2020"
KNOWN_INPUT = Path("outputs") / "report_assets" / "top10_predicted_affinities.csv"
GENERATED_INPUT = Path("outputs") / "genai" / "top10_generated_candidates_by_gaps.csv"
TARGET_PROTEIN = DATA_ROOT / "pdbbind_subset" / "2011-2019" / "2l3r" / "2l3r_protein.pdb"
OUT_ROOT = Path("outputs") / "vina_validation"
KNOWN_OUT = OUT_ROOT / "known_top10"
GENERATED_OUT = OUT_ROOT / "generated_top10"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_complex_files(pdb_id: str) -> Tuple[Path, Path]:
    pdb_id = pdb_id.lower()
    protein_matches = sorted((DATA_ROOT / "pdbbind_subset").rglob(f"{pdb_id}_protein.pdb"))
    if not protein_matches:
        raise FileNotFoundError(f"Protein PDB not found for {pdb_id}")

    protein_pdb = protein_matches[0]
    complex_dir = protein_pdb.parent
    ligand_candidates = [
        complex_dir / f"{pdb_id}_ligand.sdf",
        complex_dir / f"{pdb_id}_ligand.mol2",
    ]
    ligand_candidates.extend(sorted(complex_dir.glob("*_ligand.sdf")))
    ligand_candidates.extend(sorted(complex_dir.glob("*_ligand.mol2")))
    ligand_candidates.extend(sorted(complex_dir.glob("*.sdf")))
    ligand_candidates.extend(sorted(complex_dir.glob("*.mol2")))

    for ligand_path in ligand_candidates:
        if ligand_path.exists():
            return protein_pdb, ligand_path

    raise FileNotFoundError(f"Ligand SDF/MOL2 not found for {pdb_id}")


def dock_existing_ligand(
    obabel_command: str,
    vina_command: str,
    protein_pdb: Path,
    ligand_file: Path,
    label: str,
    out_dir: Path,
) -> float:
    poses_dir = out_dir / "poses"
    work_dir = out_dir / "work"
    poses_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = prepare_receptor(obabel_command, protein_pdb, work_dir)
    ligand_pdbqt = work_dir / f"{label}_ligand.pdbqt"
    pose_path = poses_dir / f"{label}_vina_pose.pdbqt"
    log_path = work_dir / f"{label}_vina.log"
    box = docking_box_from_protein(protein_pdb)

    # Known PDBbind ligands are already structure files, so the workflow starts
    # at Open Babel PDBQT conversion before Vina docking.
    convert_with_obabel(obabel_command, ligand_file, ligand_pdbqt)
    return run_vina(vina_command, receptor_pdbqt, ligand_pdbqt, pose_path, log_path, box)


def validate_known_top10(obabel_command: str, vina_command: str) -> Tuple[List[Dict[str, object]], List[str]]:
    rows = read_csv(KNOWN_INPUT)[:10]
    results = []
    failures = []

    for row in rows:
        rank = row["rank"]
        pdb_id = row["pdb_id"].lower()
        label = f"rank_{int(rank):02d}_{pdb_id}"
        try:
            protein_pdb, ligand_file = find_complex_files(pdb_id)
            vina_score = dock_existing_ligand(
                obabel_command=obabel_command,
                vina_command=vina_command,
                protein_pdb=protein_pdb,
                ligand_file=ligand_file,
                label=label,
                out_dir=KNOWN_OUT,
            )
            print(f"known rank {rank} {pdb_id}: Vina score {vina_score:.4f}")
        except Exception as exc:
            vina_score = ""
            message = f"known rank {rank} {pdb_id} FAILED: {exc}"
            failures.append(message)
            print(message)

        results.append(
            {
                "rank": rank,
                "pdb_id": pdb_id,
                "predicted_affinity": row["predicted_affinity"],
                "true_affinity": row["true_affinity"],
                "abs_error": row.get("absolute_error", row.get("abs_error", "")),
                "vina_score": vina_score,
            }
        )

    write_csv(
        KNOWN_OUT / "known_top10_docking_results.csv",
        results,
        ["rank", "pdb_id", "predicted_affinity", "true_affinity", "abs_error", "vina_score"],
    )
    write_failures(KNOWN_OUT / "known_top10_failures.log", failures)
    return results, failures


def validate_generated_top10(obabel_command: str, vina_command: str) -> Tuple[List[Dict[str, object]], List[str]]:
    rows = read_csv(GENERATED_INPUT)[:10]
    results = []
    failures = []
    GENERATED_OUT.mkdir(parents=True, exist_ok=True)
    (GENERATED_OUT / "poses").mkdir(parents=True, exist_ok=True)
    (GENERATED_OUT / "work").mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = prepare_receptor(obabel_command, TARGET_PROTEIN, GENERATED_OUT / "work")
    box = docking_box_from_protein(TARGET_PROTEIN)

    for index, row in enumerate(rows, start=1):
        label = f"rank_{index:02d}_generated"
        try:
            docking = dock_smiles(
                smiles=row["smiles"],
                label=label,
                protein_pdb=TARGET_PROTEIN,
                receptor_pdbqt=receptor_pdbqt,
                obabel_command=obabel_command,
                vina_command=vina_command,
                box=box,
                work_dir=GENERATED_OUT / "work",
                poses_dir=GENERATED_OUT / "poses",
            )
            vina_score = docking["vina_score"]
            print(f"generated rank {index}: Vina score {float(vina_score):.4f}")
        except Exception as exc:
            vina_score = ""
            message = f"generated rank {index} FAILED: {exc}"
            failures.append(message)
            print(message)

        results.append(
            {
                "rank": index,
                "smiles": row["smiles"],
                "predicted_affinity": row["predicted_affinity"],
                "gaps_score": row.get("gaps", row.get("gaps_score", "")),
                "qed": row["qed"],
                "lipinski_violations": row["lipinski_violations"],
                "vina_score": vina_score,
            }
        )

    write_csv(
        GENERATED_OUT / "generated_top10_docking_results.csv",
        results,
        ["rank", "smiles", "predicted_affinity", "gaps_score", "qed", "lipinski_violations", "vina_score"],
    )
    write_failures(GENERATED_OUT / "generated_top10_failures.log", failures)
    return results, failures


def write_failures(path: Path, failures: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def save_scatter_plot(rows: Sequence[Dict[str, object]], x_key: str, y_key: str, path: Path):
    import struct
    import zlib

    width, height = 800, 520
    margin_left, margin_right = 75, 35
    margin_top, margin_bottom = 35, 70
    image = bytearray([255, 255, 255] * width * height)

    def set_pixel(x: int, y: int, color):
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            image[offset : offset + 3] = bytes(color)

    def draw_line(x0: int, y0: int, x1: int, y1: int, color):
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

    def draw_circle(cx: int, cy: int, radius: int, color):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    set_pixel(x, y, color)

    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom
    for i in range(6):
        y = plot_top + round((plot_bottom - plot_top) * i / 5)
        draw_line(plot_left, y, plot_right, y, (228, 228, 228))
    draw_line(plot_left, plot_top, plot_left, plot_bottom, (45, 45, 45))
    draw_line(plot_left, plot_bottom, plot_right, plot_bottom, (45, 45, 45))

    valid_points = [(safe_float(row[x_key]), safe_float(row[y_key])) for row in rows if row.get(y_key) != ""]
    if valid_points:
        xs, ys = zip(*valid_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if min_x == max_x:
            min_x -= 1.0
            max_x += 1.0
        if min_y == max_y:
            min_y -= 1.0
            max_y += 1.0
        for x_value, y_value in valid_points:
            x = plot_left + round((x_value - min_x) / (max_x - min_x) * (plot_right - plot_left))
            y = plot_bottom - round((y_value - min_y) / (max_y - min_y) * (plot_bottom - plot_top))
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


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> List[str]:
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("|" + "|".join(values) + "|")
    return lines


def save_final_comparison(known_rows: Sequence[Dict[str, object]], generated_rows: Sequence[Dict[str, object]]):
    lines = [
        "# Final Docking Comparison",
        "",
        "AutoDock Vina is used here as secondary computational validation for the ProtLigGNN and GenAI candidate rankings. Vina scores are not docking ground truth; they provide an independent physics-inspired check where more negative values are generally more favorable.",
        "",
        "## Known PDBbind Top 10",
        "",
    ]
    lines.extend(markdown_table(known_rows, ["rank", "pdb_id", "predicted_affinity", "true_affinity", "abs_error", "vina_score"]))
    lines.extend(["", "## GenAI Top 10", ""])
    lines.extend(
        markdown_table(
            generated_rows,
            ["rank", "smiles", "predicted_affinity", "gaps_score", "qed", "lipinski_violations", "vina_score"],
        )
    )
    lines.extend(
        [
            "",
            "## Short Interpretation",
            "",
            "- Known candidates test whether high ProtLigGNN-scored PDBbind complexes also receive plausible Vina docking scores in their own receptor contexts.",
            "- Generated candidates test whether GenAI molecules prioritized by GAPS remain plausible under a separate docking-based validation against the 2l3r target.",
            "- Any blank Vina score indicates a failed docking/conversion attempt that was logged and skipped without fabricating a score.",
        ]
    )
    (OUT_ROOT / "final_docking_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RDLogger.DisableLog("rdApp.*")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    obabel_command = resolve_tool("obabel", [Path("C:/Program Files/OpenBabel-2.4.1/obabel.exe")])
    vina_command = resolve_tool(
        "vina",
        [Path("C:/Program Files (x86)/The Scripps Research Institute/Vina/vina.exe")],
    )
    require_tool(obabel_command, "Open Babel", version_flag="-V")
    require_tool(vina_command, "AutoDock Vina")

    known_rows, known_failures = validate_known_top10(obabel_command, vina_command)
    generated_rows, generated_failures = validate_generated_top10(obabel_command, vina_command)

    save_scatter_plot(known_rows, "predicted_affinity", "vina_score", OUT_ROOT / "known_top10_vina_vs_affinity.png")
    save_scatter_plot(generated_rows, "gaps_score", "vina_score", OUT_ROOT / "generated_top10_vina_vs_gaps.png")
    save_final_comparison(known_rows, generated_rows)

    print(f"known docking results: {KNOWN_OUT / 'known_top10_docking_results.csv'}")
    print(f"generated docking results: {GENERATED_OUT / 'generated_top10_docking_results.csv'}")
    print(f"known failures: {len(known_failures)}")
    print(f"generated failures: {len(generated_failures)}")
    print(f"final comparison: {OUT_ROOT / 'final_docking_comparison.md'}")


if __name__ == "__main__":
    main()
