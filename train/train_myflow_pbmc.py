#!/usr/bin/env python3
"""PBMC MyFlow training script migrated from train_100_pbmc.ipynb."""

import argparse
import functools
import os
import warnings

import anndata as ad
import flax.linen as nn
import numpy as np
import optax
import pandas as pd
import scanpy as sc
from pandas.errors import SettingWithCopyWarning

import myflow
from myflow.model import MyFlow
from myflow.utils import match_linear


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)


def _subsample_by_condition(adata_in: ad.AnnData, n_obs: int) -> ad.AnnData:
    chunks = []
    for cond in adata_in.obs["condition"].unique():
        chunks.append(sc.pp.subsample(adata_in[adata_in.obs["condition"] == cond], n_obs=n_obs, copy=True))
    return ad.concat(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyFlow on PBMC cytokine data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/zhangshibo24s/cell_flow/data_train/pbmc_adata_for_myflow_datasets_with_embeddings.h5ad",
        help="Path to pbmc h5ad file.",
    )
    parser.add_argument("--cuda-device", type=str, default="2", help="CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--num-iterations", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--valid-freq", type=int, default=20_000)
    parser.add_argument("--train-subsample", type=int, default=1000)
    parser.add_argument("--test-subsample", type=int, default=2000)

    # Optional combined Sinkhorn + Energy regularizer in OTFM.
    parser.add_argument("--combined-loss-weight", type=float, default=0.0)
    parser.add_argument("--combined-sinkhorn-weight", type=float, default=0.001)
    parser.add_argument("--combined-energy-weight", type=float, default=1.0)
    parser.add_argument("--combined-epsilon", type=float, default=1e-2)
    parser.add_argument("--output-dir", type=str, default="/home/zhangshibo24s/cell_flow/outputs")
    parser.add_argument("--save-prefix", type=str, default="pbmc")
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    adata = sc.read_h5ad(args.data_path)
    adata.obs["condition"] = adata.obs.apply(lambda x: x["donor"] + "_" + x["cytokine"], axis=1)
    adata.obs["is_control"] = adata.obs.apply(lambda x: x["cytokine"] == "PBS", axis=1)

    # sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata_train = adata[(adata.obs["cytokine"] != "IL-15") | (adata.obs["donor"] == "Donor8")].copy()
    adata_test = adata[
        ((adata.obs["cytokine"] == "IL-15") & (adata.obs["donor"] != "Donor8")) | (adata.obs["cytokine"] == "PBS")
    ].copy()
    print(f"train/test n_obs: {adata_train.n_obs}/{adata_test.n_obs}")

    cf = MyFlow(adata_train, solver="otfm")
    cf.prepare_data(
        sample_rep="X",
        control_key="is_control",
        perturbation_covariates={"cytokine_treatment": ("cytokine",)},
        perturbation_covariate_reps={"cytokine_treatment": "esm2_embeddings"},
        sample_covariates=["donor"],
        sample_covariate_reps={"donor": "donor_embeddings"},
        split_covariates=["donor"],
        max_combination_length=1,
        null_value=0.0,
    )

    adata_train_for_validation = _subsample_by_condition(adata_train, n_obs=args.train_subsample)
    adata_test_for_validation = _subsample_by_condition(adata_test, n_obs=args.test_subsample)
    adata_train_for_validation.uns = adata_train.uns.copy()
    adata_test_for_validation.uns = adata_test.uns.copy()

    cf.prepare_validation_data(
        adata_train_for_validation,
        name="train",
        n_conditions_on_log_iteration=10,
        n_conditions_on_train_end=10,
    )
    cf.prepare_validation_data(
        adata_test_for_validation,
        name="test",
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
    )

    layers_before_pool = {
        "cytokine_treatment": {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
        "donor": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
    }
    layers_after_pool = {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.0}
    match_fn = functools.partial(match_linear, epsilon=0.5, tau_a=1.0, tau_b=1.0)

    solver_kwargs = {
        "condition_combined_loss_weight": args.combined_loss_weight,
        "condition_combined_sinkhorn_weight": args.combined_sinkhorn_weight,
        "condition_combined_energy_weight": args.combined_energy_weight,
        "condition_combined_epsilon": args.combined_epsilon,
    }

    cf.prepare_model(
        condition_mode="deterministic",
        regularization=0.0,
        pooling="attention_token",
        pooling_kwargs={},
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        condition_embedding_dim=256,
        cond_output_dropout=0.9,
        condition_encoder_kwargs={},
        pool_sample_covariates=True,
        time_freqs=1024,
        time_encoder_dims=[1024, 1024, 1024],
        time_encoder_dropout=0.0,
        hidden_dims=[2048, 2048, 2048],
        hidden_dropout=0.0,
        conditioning="concatenation",
        decoder_dims=[4096, 4096, 4096],
        vf_act_fn=nn.silu,
        vf_kwargs=None,
        probability_path={"constant_noise": 0.5},
        match_fn=match_fn,
        optimizer=optax.MultiSteps(optax.adam(5e-5), 20),
        solver_kwargs=solver_kwargs,
        layer_norm_before_concatenation=False,
        linear_projection_before_concatenation=False,
    )

    metrics_callback = myflow.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    callbacks = [metrics_callback]

    cf.train(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        callbacks=callbacks,
        valid_freq=args.valid_freq,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    cf.save(args.output_dir, file_prefix=args.save_prefix, overwrite=True)

    # Prediction pipeline: use control cells as sources and non-control conditions as targets.
    adata_ctrl_for_prediction = adata_test_for_validation[adata_test_for_validation.obs["is_control"].to_numpy()].copy()
    covariate_data_pred = (
        adata_test_for_validation[~adata_test_for_validation.obs["is_control"].to_numpy()]
        .obs.drop_duplicates(subset=["condition"])
        .copy()
    )

    preds = cf.predict(
        adata=adata_ctrl_for_prediction,
        sample_rep="X",
        condition_id_key="condition",
        covariate_data=covariate_data_pred,
        predict_batch_size=args.predict_batch_size,
    )

    if preds is None:
        raise RuntimeError("Prediction returned None unexpectedly.")

    np.savez_compressed(
        os.path.join(args.output_dir, f"{args.save_prefix}_predictions.npz"),
        **{k: np.asarray(v) for k, v in preds.items()},
    )

    adata_preds = []
    for cond, arr in preds.items():
        arr_np = np.asarray(arr)
        obs_df = pd.DataFrame(index=[f"{cond}_{i}" for i in range(arr_np.shape[0])])
        obs_df["condition"] = cond
        pred_adata = ad.AnnData(X=arr_np, obs=obs_df, var=adata_train.var.copy())
        adata_preds.append(pred_adata)

    adata_preds_concat = ad.concat(adata_preds, merge="same")
    adata_preds_concat.write_h5ad(os.path.join(args.output_dir, f"{args.save_prefix}_predictions.h5ad"))

    print("Training logs keys:")
    print(cf.trainer.training_logs.keys())
    print(f"Saved model and predictions to: {args.output_dir}")


if __name__ == "__main__":
    main()
