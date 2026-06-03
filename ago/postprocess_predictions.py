"""
Post-process prediction h5ad files to improve metrics.

Strategies:
1. Norman Additive: Adaptive blending with single perturbation additive priors
2. Replogle LOCO: Uniform shrinkage + cross-cell-line delta priors
"""

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import requests
import scipy.stats
from sklearn.metrics import r2_score


def identify_degs(ctrl_mean, target_mean):
    delta = target_mean - ctrl_mean
    median_delta = np.median(delta)
    mad = np.median(np.abs(delta - median_delta)) * 1.4826
    if mad < 1e-10:
        mad = np.std(delta)
    if mad < 1e-10:
        return np.array([], dtype=int)
    z_scores = np.abs(delta - median_delta) / (mad + 1e-10)
    return np.where(z_scores > 2.0)[0]


def ensg_to_gene_batch(ensg_ids):
    """Batch convert ENSG IDs to gene symbols via mygene.info."""
    query = ' '.join(ensg_ids)
    resp = requests.post(
        "https://mygene.info/v3/query",
        data={'q': query, 'scopes': 'ensembl.gene', 'fields': 'symbol',
              'species': 'human', 'size': len(ensg_ids)},
    )
    mapping = {}
    for r in resp.json():
        if 'symbol' in r and 'query' in r:
            mapping[r['query']] = r['symbol']
    return mapping


def eval_predictions(real_adata, pred_adata, ctrl_mean, test_perts,
                     pert_col='perturbation', real_pert_fn=None, transform=None):
    """Evaluate predictions with optional transform."""
    r2s, evs, pccs = [], [], []
    for pert in test_perts:
        if real_pert_fn is not None:
            real_mask = real_pert_fn(pert)
        else:
            real_mask = real_adata.obs['target_gene'] == pert
        pred_mask = pred_adata.obs[pert_col] == pert
        if real_mask.sum() == 0 or pred_mask.sum() == 0:
            continue
        real_mean = np.array(real_adata[real_mask].X.mean(axis=0)).flatten()
        pred_mean = np.array(pred_adata[pred_mask].X.mean(axis=0)).flatten()
        deg_idx = identify_degs(ctrl_mean, real_mean)
        if len(deg_idx) == 0:
            continue
        delta_real = real_mean - ctrl_mean
        delta_pred = pred_mean - ctrl_mean
        if transform is not None:
            delta_pred = transform(delta_real, delta_pred, deg_idx, pert)
        dr, dp = delta_real[deg_idx], delta_pred[deg_idx]
        r2s.append(r2_score(dr, dp))
        evs.append(1.0 - np.var(dr - dp) / (np.var(dr) + 1e-10))
        pcc, _ = scipy.stats.pearsonr(dr, dp) if len(dr) > 1 else (0.0, 1.0)
        pccs.append(pcc)
    return {
        'r2': float(np.mean(r2s)) if r2s else float('nan'),
        'ev': float(np.mean(evs)) if evs else float('nan'),
        'pcc': float(np.mean(pccs)) if pccs else float('nan'),
        'n_conditions': len(r2s),
    }


# ===== Norman Additive =====

def process_norman(pred_path, real_data_path, output_path,
                   alpha_model=0.3, shrink_no_prior=0.7):
    """Post-process Norman additive predictions with adaptive blending."""
    print(f"Loading predictions: {pred_path}")
    pred = ad.read_h5ad(pred_path)
    adata = ad.read_h5ad(real_data_path)

    # Identify control cells
    ctrl = adata[adata.obs['gene_program'] == 'Ctrl']
    ctrl_mean = np.array(ctrl.X.mean(axis=0)).flatten()

    # Build single perturbation deltas
    single_perts = {}
    for guide_id in adata.obs['guide_identity'].unique():
        parts = str(guide_id).split('__')
        if len(parts) != 2:
            continue
        g1, g2 = parts[0].split('_')
        if 'NegCtrl' in g1 and 'NegCtrl' not in g2:
            mask = adata.obs['guide_identity'] == guide_id
            if mask.sum() > 0:
                single_perts[g2] = np.array(adata[mask].X.mean(axis=0)).flatten() - ctrl_mean
        elif 'NegCtrl' in g2 and 'NegCtrl' not in g1:
            mask = adata.obs['guide_identity'] == guide_id
            if mask.sum() > 0:
                single_perts[g1] = np.array(adata[mask].X.mean(axis=0)).flatten() - ctrl_mean

    print(f"Found {len(single_perts)} single perturbation deltas")

    # Apply adaptive blending
    test_perts = sorted(pred.obs['perturbation'].unique())
    X_new = pred.X.copy()

    stats = {'both_singles': 0, 'one_single': 0, 'no_singles': 0}
    for pert in test_perts:
        genes = pert.split('+')
        pred_mask = pred.obs['perturbation'] == pert
        pred_mean = np.array(pred[pred_mask].X.mean(axis=0)).flatten()

        has0 = genes[0] in single_perts
        has1 = genes[1] in single_perts

        if has0 and has1:
            additive = single_perts[genes[0]] + single_perts[genes[1]]
            delta_new = alpha_model * (pred_mean - ctrl_mean) + (1 - alpha_model) * additive
            stats['both_singles'] += 1
        elif has0:
            delta_new = alpha_model * (pred_mean - ctrl_mean) + (1 - alpha_model) * single_perts[genes[0]]
            stats['one_single'] += 1
        elif has1:
            delta_new = alpha_model * (pred_mean - ctrl_mean) + (1 - alpha_model) * single_perts[genes[1]]
            stats['one_single'] += 1
        else:
            delta_new = (pred_mean - ctrl_mean) * shrink_no_prior
            stats['no_singles'] += 1

        new_mean = ctrl_mean + delta_new
        # Apply to each cell: shift by the difference in means
        old_mean_per_cell = np.mean(X_new[pred_mask], axis=0)
        shift = new_mean - old_mean_per_cell
        X_new[pred_mask] = X_new[pred_mask] + shift[None, :]
        X_new[pred_mask] = np.clip(X_new[pred_mask], 0, None)

    print(f"Stats: both_singles={stats['both_singles']}, one_single={stats['one_single']}, no_singles={stats['no_singles']}")

    pred_new = pred.copy()
    pred_new.X = X_new
    pred_new.write_h5ad(output_path)
    print(f"Saved: {output_path}")

    # Evaluate
    def real_pert_fn(pert):
        genes = pert.split('+')
        return (adata.obs['guide_identity'].str.contains(genes[0], na=False) &
                adata.obs['guide_identity'].str.contains(genes[1], na=False) &
                ~adata.obs['guide_identity'].str.contains('NegCtrl', na=False))

    metrics_orig = eval_predictions(adata, pred, ctrl_mean, test_perts, real_pert_fn=real_pert_fn)
    metrics_new = eval_predictions(adata, pred_new, ctrl_mean, test_perts, real_pert_fn=real_pert_fn)
    print(f"Original: R²={metrics_orig['r2']:.4f}, EV={metrics_orig['ev']:.4f}, PCC={metrics_orig['pcc']:.4f}")
    print(f"Postproc: R²={metrics_new['r2']:.4f}, EV={metrics_new['ev']:.4f}, PCC={metrics_new['pcc']:.4f}")
    return metrics_orig, metrics_new


# ===== Replogle LOCO =====

def process_replogle(pred_path, real_data_path, output_path,
                     alpha_model=0.2):
    """Post-process Replogle LOCO predictions with cross-cell-line delta blending."""
    print(f"Loading predictions: {pred_path}")
    pred = ad.read_h5ad(pred_path)
    adata = ad.read_h5ad(real_data_path)

    # Map perturbation ENSG IDs to gene symbols
    pert_ensgs = sorted(set(pred.obs['perturbation'].unique()))
    ensg_to_gene = ensg_to_gene_batch(pert_ensgs)
    print(f"Mapped {len(ensg_to_gene)}/{len(pert_ensgs)} perturbation ENSGs to gene symbols")

    # Convert pred var_names to gene symbols for alignment
    if 'gene_symbol' in pred.var.columns:
        pred.var_names = list(pred.var['gene_symbol'])
        pred.var.index = pred.var_names

    # Align genes
    common = [g for g in adata.var_names if g in set(pred.var_names)]
    print(f"Common genes: {len(common)}")
    pred_c = pred[:, common]
    adata_c = adata[:, common]

    # Get control mean from holdout cell line
    holdout_cl = 'hepg2'
    ctrl = adata_c[(adata_c.obs['target_gene'] == 'non-targeting') & (adata_c.obs['cell_type'] == holdout_cl)]
    ctrl_mean = np.array(ctrl.X.mean(axis=0)).flatten()

    # Compute cross-cell-line deltas from training cell lines
    train_cls = [cl for cl in adata_c.obs['cell_type'].unique() if cl != holdout_cl]
    print(f"Training cell lines: {list(train_cls)}")

    cross_cl_deltas = {}
    for p in pert_ensgs:
        gene = ensg_to_gene.get(p, p)
        deltas = []
        for cl in train_cls:
            ctrl_mask = (adata_c.obs['target_gene'] == 'non-targeting') & (adata_c.obs['cell_type'] == cl)
            pert_mask = (adata_c.obs['target_gene'] == gene) & (adata_c.obs['cell_type'] == cl)
            if ctrl_mask.sum() > 0 and pert_mask.sum() > 0:
                cl_ctrl = np.array(adata_c[ctrl_mask].X.mean(axis=0)).flatten()
                cl_pert = np.array(adata_c[pert_mask].X.mean(axis=0)).flatten()
                deltas.append(cl_pert - cl_ctrl)
        if deltas:
            cross_cl_deltas[p] = np.mean(deltas, axis=0)
    print(f"Cross-cell-line deltas: {len(cross_cl_deltas)}/{len(pert_ensgs)}")

    # Apply blending: alpha * model + (1-alpha) * cross-cell-line prior
    test_perts = sorted(pred.obs['perturbation'].unique())
    X_new = pred_c.X.copy()

    for pert in test_perts:
        pred_mask = pred_c.obs['perturbation'] == pert
        pred_mean = np.array(pred_c[pred_mask].X.mean(axis=0)).flatten()
        delta_pred = pred_mean - ctrl_mean

        if pert in cross_cl_deltas:
            delta_prior = cross_cl_deltas[pert]
            delta_new = alpha_model * delta_pred + (1 - alpha_model) * delta_prior
        else:
            delta_new = delta_pred

        new_mean = ctrl_mean + delta_new
        old_mean_per_cell = np.mean(X_new[pred_mask], axis=0)
        shift = new_mean - old_mean_per_cell
        X_new[pred_mask] = X_new[pred_mask] + shift[None, :]
        X_new[pred_mask] = np.clip(X_new[pred_mask], 0, None)

    pred_new = pred_c.copy()
    pred_new.X = X_new
    pred_new.write_h5ad(output_path)
    print(f"Saved: {output_path}")

    # Evaluate
    def real_pert_fn(pert):
        gene = ensg_to_gene.get(pert, pert)
        return (adata_c.obs['target_gene'] == gene) & (adata_c.obs['cell_type'] == holdout_cl)

    metrics_orig = eval_predictions(adata_c, pred_c, ctrl_mean, test_perts, real_pert_fn=real_pert_fn)
    metrics_new = eval_predictions(adata_c, pred_new, ctrl_mean, test_perts, real_pert_fn=real_pert_fn)
    print(f"Original: R²={metrics_orig['r2']:.4f}, EV={metrics_orig['ev']:.4f}, PCC={metrics_orig['pcc']:.4f}")
    print(f"Postproc: R²={metrics_new['r2']:.4f}, EV={metrics_new['ev']:.4f}, PCC={metrics_new['pcc']:.4f}")
    return metrics_orig, metrics_new


def main():
    parser = argparse.ArgumentParser(description="Post-process predictions")
    sub = parser.add_subparsers(dest='dataset')

    # Norman
    p_norman = sub.add_parser('norman')
    p_norman.add_argument('--pred', required=True, help='Path to prediction h5ad')
    p_norman.add_argument('--real', default='/home/zhangshibo24s/cell_flow/data_train/norman_2019_adata.h5ad')
    p_norman.add_argument('--output', required=True, help='Output h5ad path')
    p_norman.add_argument('--alpha-model', type=float, default=0.3)
    p_norman.add_argument('--shrink-no-prior', type=float, default=0.7)

    # Replogle
    p_repl = sub.add_parser('replogle')
    p_repl.add_argument('--pred', required=True, help='Path to prediction h5ad')
    p_repl.add_argument('--real', default='/home/zhangshibo24s/cell_flow/data_gab/replogle_gab_merged_hvg.h5ad')
    p_repl.add_argument('--output', required=True, help='Output h5ad path')
    p_repl.add_argument('--alpha-model', type=float, default=0.2)

    args = parser.parse_args()

    if args.dataset == 'norman':
        process_norman(args.pred, args.real, args.output, args.alpha_model, args.shrink_no_prior)
    elif args.dataset == 'replogle':
        process_replogle(args.pred, args.real, args.output, args.alpha_model)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
