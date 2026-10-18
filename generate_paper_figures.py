import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


OUTPUTS_DIR = Path("outputs")
FIGURES_DIR = OUTPUTS_DIR / "paper_figures"


def warn(message: str) -> None:
    print(f"Warning: {message}")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_epoch_histories() -> Dict[str, List[Dict[str, str]]]:
    histories: Dict[str, List[Dict[str, str]]] = {}
    for path in sorted(OUTPUTS_DIR.glob("epoch_history_*.csv")):
        run_name = path.stem.replace("epoch_history_", "")
        try:
            histories[run_name] = read_csv_rows(path)
        except Exception as exc:
            warn(f"Could not read epoch history {path.name}: {exc}")
    if not histories:
        warn("No epoch history CSV files found.")
    return histories


def collect_prediction_tables() -> Dict[str, List[Dict[str, str]]]:
    predictions: Dict[str, List[Dict[str, str]]] = {}
    for path in sorted(OUTPUTS_DIR.glob("test_predictions_*.csv")):
        run_name = path.stem.replace("test_predictions_", "")
        try:
            predictions[run_name] = read_csv_rows(path)
        except Exception as exc:
            warn(f"Could not read prediction CSV {path.name}: {exc}")
    if not predictions:
        warn("No test prediction CSV files found.")
    return predictions


def parse_comparison_file(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"Run:\s*(?P<run>[^\r\n]+)\s+"
        r"(?:Train command|Command):\s*(?P<command>[^\r\n]+)\s+"
        r"Best epoch:\s*(?P<best_epoch>\d+)\s+"
        r"(?:Test\s+)?PCC:\s*(?P<pcc>[-0-9.]+)\s+"
        r"(?:Test\s+)?Spearman:\s*(?P<spearman>[-0-9.]+)\s+"
        r"(?:Test\s+)?RMSE:\s*(?P<rmse>[-0-9.]+)\s+"
        r"(?:Test\s+)?MAE:\s*(?P<mae>[-0-9.]+)",
        re.MULTILINE,
    )
    rows: List[Dict[str, object]] = []
    for match in pattern.finditer(text):
        rows.append(
            {
                "run_name": match.group("run").strip(),
                "command": match.group("command").strip(),
                "best_epoch": int(match.group("best_epoch")),
                "pcc": float(match.group("pcc")),
                "spearman": float(match.group("spearman")),
                "rmse": float(match.group("rmse")),
                "mae": float(match.group("mae")),
                "source_file": path.name,
            }
        )
    if not rows:
        warn(f"No model summaries parsed from {path.name}.")
    return rows


def collect_comparisons() -> List[Dict[str, object]]:
    comparison_files = sorted(OUTPUTS_DIR.glob("comparison*.txt")) + sorted((OUTPUTS_DIR / "logs").glob("comparison*.txt"))
    seen: set[str] = set()
    rows: List[Dict[str, object]] = []
    for path in comparison_files:
        for row in parse_comparison_file(path):
            key = str(row["run_name"])
            if key not in seen:
                seen.add(key)
                rows.append(row)
    if not rows:
        warn("No comparison summary files with parseable metrics were found.")
    return rows


def choose_primary_runs(names: List[str]) -> List[str]:
    large_runs = [name for name in names if "5000_e40" in name]
    if large_runs:
        return large_runs
    medium_runs = [name for name in names if "2000" in name]
    if medium_runs:
        return medium_runs
    return names


def save_training_loss_curve(histories: Dict[str, List[Dict[str, str]]]) -> None:
    if not histories:
        return
    selected_runs = choose_primary_runs(list(histories.keys()))
    plt.figure(figsize=(8, 5))
    plotted = False
    for run_name in selected_runs:
        rows = histories.get(run_name, [])
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        train_loss = [float(row["train_loss"]) for row in rows]
        val_loss = [float(row["val_loss"]) for row in rows]
        plt.plot(epochs, train_loss, linewidth=2, label=f"{run_name} train")
        plt.plot(epochs, val_loss, linewidth=2, linestyle="--", label=f"{run_name} val")
        plotted = True
    if not plotted:
        warn("Could not plot training loss curves from available histories.")
        plt.close()
        return
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_loss_curve.png", dpi=300)
    plt.close()


def save_validation_pcc_curve(histories: Dict[str, List[Dict[str, str]]]) -> None:
    if not histories:
        return
    selected_runs = choose_primary_runs(list(histories.keys()))
    plt.figure(figsize=(8, 5))
    plotted = False
    for run_name in selected_runs:
        rows = histories.get(run_name, [])
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        val_pcc = [float(row["val_pcc"]) for row in rows]
        plt.plot(epochs, val_pcc, linewidth=2, label=run_name)
        plotted = True
    if not plotted:
        warn("Could not plot validation PCC curves from available histories.")
        plt.close()
        return
    plt.title("Validation PCC Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation PCC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_pcc_curve.png", dpi=300)
    plt.close()


def save_predicted_vs_true_scatter(predictions: Dict[str, List[Dict[str, str]]]) -> None:
    if not predictions:
        return
    selected_runs = choose_primary_runs(list(predictions.keys()))
    plt.figure(figsize=(7, 7))
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    plotted = False
    for run_name in selected_runs:
        rows = predictions.get(run_name, [])
        if not rows:
            continue
        true_values = [float(row["true_affinity"]) for row in rows]
        pred_values = [float(row["predicted_affinity"]) for row in rows]
        plt.scatter(true_values, pred_values, alpha=0.7, s=22, label=run_name)
        local_min = min(min(true_values), min(pred_values))
        local_max = max(max(true_values), max(pred_values))
        min_val = local_min if min_val is None else min(min_val, local_min)
        max_val = local_max if max_val is None else max(max_val, local_max)
        plotted = True
    if not plotted or min_val is None or max_val is None:
        warn("Could not plot predicted-vs-true scatter from available predictions.")
        plt.close()
        return
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5, color="black")
    plt.title("Predicted vs True Affinity")
    plt.xlabel("True Affinity")
    plt.ylabel("Predicted Affinity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_true_scatter.png", dpi=300)
    plt.close()


def save_error_distribution_histogram(predictions: Dict[str, List[Dict[str, str]]]) -> None:
    if not predictions:
        return
    selected_runs = choose_primary_runs(list(predictions.keys()))
    plt.figure(figsize=(8, 5))
    plotted = False
    for run_name in selected_runs:
        rows = predictions.get(run_name, [])
        if not rows:
            continue
        errors = [float(row["error"]) for row in rows]
        plt.hist(errors, bins=25, alpha=0.45, label=run_name)
        plotted = True
    if not plotted:
        warn("Could not plot error distributions from available predictions.")
        plt.close()
        return
    plt.title("Prediction Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_distribution_histogram.png", dpi=300)
    plt.close()


def save_model_comparison_barplot(comparisons: List[Dict[str, object]]) -> None:
    if not comparisons:
        return
    target_rows = [row for row in comparisons if "5000_e40" in str(row["run_name"])]
    if not target_rows:
        target_rows = comparisons
    target_rows = sorted(target_rows, key=lambda row: str(row["run_name"]))

    metrics = ["pcc", "spearman", "rmse", "mae"]
    labels = ["PCC", "Spearman", "RMSE", "MAE"]
    x_positions = list(range(len(metrics)))
    width = 0.8 / max(len(target_rows), 1)

    plt.figure(figsize=(10, 5))
    for idx, row in enumerate(target_rows):
        offsets = [x + (idx - (len(target_rows) - 1) / 2) * width for x in x_positions]
        values = [float(row[metric]) for metric in metrics]
        plt.bar(offsets, values, width=width, label=str(row["run_name"]))

    plt.title("Model Comparison on Available Test Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(x_positions, labels)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_barplot.png", dpi=300)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    histories = collect_epoch_histories()
    predictions = collect_prediction_tables()
    comparisons = collect_comparisons()

    save_training_loss_curve(histories)
    save_validation_pcc_curve(histories)
    save_predicted_vs_true_scatter(predictions)
    save_error_distribution_histogram(predictions)
    save_model_comparison_barplot(comparisons)

    print(f"Saved paper figures to {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
