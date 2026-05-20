#!/usr/bin/env python3
"""Summarize raw and outlier-filtered experiment metrics from generated results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


RUNS = [
    {
        "task": "norman_additive",
        "method": "myflow",
        "summary": ROOT / "results/outputs/outputs_myflow_norman_additive_20260518_173550_609645/metrics_summary_norman_scdfm_additive.json",
        "condition": ROOT / "results/outputs/outputs_myflow_norman_additive_20260518_173550_609645/condition_metrics_norman_scdfm_additive.csv",
    },
    {
        "task": "norman_holdout",
        "method": "myflow",
        "summary": ROOT / "results/outputs/outputs_myflow_norman_holdout_20260518_173550_609645/metrics_summary_norman_scdfm_holdout.json",
        "condition": ROOT / "results/outputs/outputs_myflow_norman_holdout_20260518_173550_609645/condition_metrics_norman_scdfm_holdout.csv",
    },
    {
        "task": "loco",
        "method": "myflow",
        "summary": ROOT / "results/outputs/outputs_myflow_loco_20260518_183018_626024/metrics_summary_20260518_183034.json",
        "condition": ROOT / "results/outputs/outputs_myflow_loco_20260518_183018_626024/des_calibrated_blend_scale2_v2/des_per_condition_before.csv",
    },
    {
        "task": "norman_additive",
        "method": "cellflow",
        "summary": ROOT / "results/outputs/outputs_norman_baseline_additive_20260518_174602_719791/metrics_summary_norman_baseline_additive.json",
        "condition": ROOT / "results/outputs/outputs_norman_baseline_additive_20260518_174602_719791/condition_metrics_norman_baseline_additive.csv",
    },
    {
        "task": "norman_holdout",
        "method": "cellflow",
        "summary": ROOT / "results/outputs/outputs_norman_baseline_holdout_20260518_174602_719791/metrics_summary_norman_baseline_holdout.json",
        "condition": ROOT / "results/outputs/outputs_norman_baseline_holdout_20260518_174602_719791/condition_metrics_norman_baseline_holdout.csv",
    },
]


METRICS = ["mse", "mae", "l2", "pearson_delta", "pearson_delta_top20", "des_recall", "des_accuracy", "de_spearman"]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f, parse_constant=lambda _: math.nan)


def filtered_metrics(df: pd.DataFrame, trim: float) -> tuple[dict, list[str]]:
    keep = pd.Series(True, index=df.index)
    if "mse" in df:
        keep &= df["mse"] <= df["mse"].quantile(1 - trim)
    if "pearson_delta" in df:
        keep &= df["pearson_delta"] >= df["pearson_delta"].quantile(trim)
    if "des_recall" in df:
        keep &= df["des_recall"] >= df["des_recall"].quantile(trim)
    kept = df[keep]
    out = {}
    for metric in METRICS:
        if metric in kept:
            out[metric] = float(kept[metric].dropna().mean()) if kept[metric].dropna().size else math.nan
    dropped = df.loc[~keep, "condition"].astype(str).tolist() if "condition" in df else []
    out["conditions_count"] = int(len(kept))
    out["dropped_conditions_count"] = int(len(df) - len(kept))
    return out, dropped


def main() -> None:
    rows = []
    dropped = {}
    trim = 0.05
    for run in RUNS:
        raw = load_json(run["summary"])
        row = {"task": run["task"], "method": run["method"], "view": "raw", "conditions_count": raw.get("des_conditions_count")}
        for metric in METRICS:
            row[metric] = raw.get(metric, math.nan)
        rows.append(row)

        if run["condition"].exists():
            df = pd.read_csv(run["condition"])
            filt, dropped_conditions = filtered_metrics(df, trim)
            row = {"task": run["task"], "method": run["method"], "view": "filtered_5pct", **filt}
            rows.append(row)
            dropped[f"{run['task']}::{run['method']}"] = dropped_conditions

    out_dir = ROOT / "results/outputs/postprocess_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "raw_vs_filtered_summary.csv", index=False)
    with open(out_dir / "dropped_conditions_5pct.json", "w") as f:
        json.dump(dropped, f, indent=2)
    print(summary.to_string(index=False))
    print(f"Saved {out_dir / 'raw_vs_filtered_summary.csv'}")


if __name__ == "__main__":
    main()
