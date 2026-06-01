#!/usr/bin/env python3
import argparse
from pathlib import Path

import anndata as ad
import numpy as np


DEFAULT_SEED = 20240508


def stratified_subsample_obs(obs, fraction, rng, group_key):
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1 or len(obs) == 0:
        return obs.copy()

    selected_positions = []
    for _, positions in obs.groupby(group_key, observed=True).indices.items():
        positions = np.asarray(positions)
        n_keep = max(1, int(round(len(positions) * fraction)))
        selected_positions.extend(rng.choice(positions, size=n_keep, replace=False).tolist())

    selected_positions = np.asarray(selected_positions)
    selected_positions.sort()
    return obs.iloc[selected_positions].copy()


def summarize(name, obs, control_key, split_key="cell_type"):
    control_mask = obs[control_key].astype(bool).to_numpy()
    pert_obs = obs.loc[~control_mask]
    n_conditions = pert_obs[["target_gene", "cell_type"]].drop_duplicates().shape[0]
    n_controls = int(control_mask.sum())
    n_targets = int((~control_mask).sum())
    split_pair_count = 0
    split_rows = []
    for split_value, split_obs in obs.groupby(split_key, observed=True):
        split_control_mask = split_obs[control_key].astype(bool).to_numpy()
        split_controls = int(split_control_mask.sum())
        split_targets = int((~split_control_mask).sum())
        split_conditions = int(split_obs.loc[~split_control_mask, ["target_gene", "cell_type"]].drop_duplicates().shape[0])
        split_pairs = split_controls * split_targets
        split_pair_count += split_pairs
        split_rows.append((split_value, split_controls, split_targets, split_conditions, split_pairs))
    return {
        "name": name,
        "cells": len(obs),
        "controls": n_controls,
        "targets": n_targets,
        "target_genes": int(pert_obs["target_gene"].nunique()),
        "target_conditions": int(n_conditions),
        "global_cell_level_pairs": int(n_controls * n_targets),
        "split_cell_level_pairs": int(split_pair_count),
        "split_rows": split_rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adata", default="/home/zhangshibo24s/cell_flow/data_train/replogle.h5ad")
    p.add_argument("--gene-list", default="/home/zhangshibo24s/cell_flow/data_train/selected_genes_27k.txt")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--holdout-cell-line", default="hepg2")
    p.add_argument("--train-cell-fraction", type=float, default=0.3)
    p.add_argument("--test-cell-fraction", type=float, default=0.3)
    p.add_argument("--val-fraction", type=float, default=0.006)
    p.add_argument("--control-key", default="control")
    p.add_argument("--use-cell-type-split", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    adata = ad.read_h5ad(args.adata, backed="r")
    obs = adata.obs.copy()
    if "gene_id" in obs:
        obs["target_gene"] = obs["gene_id"].astype(str)
    elif "gene" in obs:
        obs["target_gene"] = obs["gene"].astype(str)
    if "cell_line" in obs:
        obs["cell_type"] = obs["cell_line"].astype(str)
    if args.control_key not in obs:
        obs[args.control_key] = obs["target_gene"].astype(str) == "non-targeting"
    else:
        obs[args.control_key] = obs[args.control_key].astype(bool)

    with open(args.gene_list, "r", encoding="utf-8") as f:
        valid_genes = {line.strip() for line in f if line.strip()}

    before_filter = len(obs)
    valid_mask = obs["target_gene"].isin(valid_genes) | (obs["target_gene"] == "non-targeting")
    obs = obs.loc[valid_mask].copy()

    holdout = args.holdout_cell_line
    if holdout not in set(obs["cell_type"].unique()):
        raise ValueError(f"Holdout cell line {holdout!r} not found.")

    other_mask = obs["cell_type"] != holdout
    holdout_mask = obs["cell_type"] == holdout
    pert_targets = [p for p in obs.loc[holdout_mask, "target_gene"].unique().tolist() if p != "non-targeting"]

    rng = np.random.default_rng(args.seed)
    shuffled_perts = rng.permutation(pert_targets)
    n_train_perts = int(0.3 * len(shuffled_perts))
    n_test_perts = int(0.3 * len(shuffled_perts))
    train_perts = set(shuffled_perts[:n_train_perts])
    test_perts = set(shuffled_perts[-n_test_perts:])

    train_mask = (
        other_mask
        | (holdout_mask & obs["target_gene"].isin(train_perts))
        | (holdout_mask & (obs["target_gene"] == "non-targeting"))
    )
    test_mask = holdout_mask & obs["target_gene"].isin(test_perts)

    train_full = obs.loc[train_mask].copy()
    test_holdout = obs.loc[test_mask].copy()
    train_before_subsample = len(train_full)
    test_before_subsample = len(test_holdout)

    train_full = stratified_subsample_obs(train_full, args.train_cell_fraction, rng, "target_gene")
    test_holdout = stratified_subsample_obs(test_holdout, args.test_cell_fraction, rng, "target_gene")

    n_train_total = len(train_full)
    n_val = int(n_train_total * args.val_fraction)
    val_indices = rng.choice(n_train_total, n_val, replace=False)
    val_mask = np.zeros(n_train_total, dtype=bool)
    val_mask[val_indices] = True

    val = train_full.iloc[val_mask].copy()
    train = train_full.iloc[~val_mask].copy()
    pred_controls = train_full.loc[
        (train_full["cell_type"] == holdout) & train_full[args.control_key].astype(bool)
    ].copy()

    print(f"seed={args.seed}")
    print(f"obs_before_valid_gene_filter={before_filter}")
    print(f"obs_after_valid_gene_filter={len(obs)}")
    print(f"holdout_cell_line={holdout}")
    print(f"holdout_perturbations_total={len(pert_targets)}")
    print(f"holdout_perturbations_train={len(train_perts)}")
    print(f"holdout_perturbations_test={len(test_perts)}")
    print(f"train_cells_before_subsample={train_before_subsample}")
    print(f"test_cells_before_subsample={test_before_subsample}")
    print(f"train_cells_after_subsample_before_val={len(train_full)}")
    print(f"test_cells_after_subsample={len(test_holdout)}")
    print(f"val_cells={len(val)}")
    print(f"prediction_holdout_control_cells={len(pred_controls)}")

    for row in [
        summarize("train_passed_to_MyFlow", train, args.control_key),
        summarize("val_held_out_from_train", val, args.control_key),
        summarize("zero_shot_test_targets", test_holdout, args.control_key),
        summarize("prediction_source_controls", pred_controls, args.control_key),
    ]:
        effective_pairs = row["split_cell_level_pairs"] if args.use_cell_type_split else row["global_cell_level_pairs"]
        print(
            f"{row['name']}: cells={row['cells']}, controls={row['controls']}, targets={row['targets']}, "
            f"target_genes={row['target_genes']}, target_conditions={row['target_conditions']}, "
            f"global_cell_level_pairs={row['global_cell_level_pairs']}, "
            f"split_cell_level_pairs={row['split_cell_level_pairs']}, "
            f"effective_cell_level_pairs={effective_pairs}"
        )
        for split_value, controls, targets, conditions, pairs in row["split_rows"]:
            print(
                f"  {row['name']}/{split_value}: controls={controls}, targets={targets}, "
                f"target_conditions={conditions}, split_cell_level_pairs={pairs}"
            )


if __name__ == "__main__":
    main()
