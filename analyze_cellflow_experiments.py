#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "mse",
    "mae",
    "l2",
    "pearson_delta",
    "pearson_delta_top20",
    "direction_sign_score",
    "des_recall",
    "des_accuracy",
    "de_spearman",
]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_metrics(input_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("metrics_summary_*.json")):
            row = load_json(path)
            row["metrics_file"] = str(path)
            row["output_dir"] = str(input_dir)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_bars(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    available = [col for col in METRIC_COLUMNS if col in metrics.columns]
    for metric in available:
        plot_df = metrics.dropna(subset=[metric]).copy()
        if plot_df.empty:
            continue
        plot_df = plot_df.sort_values(metric, ascending=metric in {"mse", "mae", "l2"})
        plt.figure(figsize=(max(8, 0.8 * len(plot_df)), 4.5))
        plt.bar(plot_df["run_label"].astype(str), plot_df[metric])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} by run")
        plt.tight_layout()
        plt.savefig(output_dir / f"metric_{metric}.png", dpi=200)
        plt.close()


def plot_loss_curves(input_dirs: list[Path], output_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    plotted = False
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("training_logs_*.csv")):
            df = pd.read_csv(path)
            if "loss" not in df:
                continue
            run_label = path.stem.replace("training_logs_", "")
            plt.plot(df["step"], df["loss"], label=run_label, linewidth=1.2, alpha=0.85)
            plotted = True
    if plotted:
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training loss")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss_curves.png", dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dirs", nargs="+", help="Output directories containing metrics_summary_*.json files.")
    p.add_argument("--output-dir", default="analysis_outputs")
    args = p.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = collect_metrics(input_dirs)
    if not metrics.empty:
        metrics.to_csv(output_dir / "metrics_summary_table.csv", index=False)
    plot_metric_bars(metrics, output_dir)
    plot_loss_curves(input_dirs, output_dir)
    print(f"Saved analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
