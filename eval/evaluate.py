from cell_eval import MetricsEvaluator
import scanpy as sc
import anndata as ad
# from cell_eval.data import build_random_anndata, downsample_cells
adata_ctrl = sc.read_h5ad("/home/zhangshibo24s/liaoning-Squidiff-main/data/k562_ctrl.h5ad")
adata_ctrl.obs["target_gene"] = "non-targeting"
adata_real = sc.read_h5ad("/home/zhangshibo24s/liaoning-Squidiff-main/data/k562_validation.h5ad")
adata_pred_ours = sc.read_h5ad("/home/zhangshibo24s/liaoning-Squidiff-main/result_data/k562_ctrl_ours_ours.h5ad")
adata_pred_state = sc.read_h5ad("/home/zhangshibo24s/liaoning-Squidiff-main/result_data/k562_ctrl_state.h5ad")

adata_real = ad.concat(
        [
            adata_real,
            adata_ctrl
        ]
)
adata_pred_ours = ad.concat(
        [
            adata_pred_ours,
            adata_ctrl
        ]
)
adata_pred_state = ad.concat(
        [
            adata_pred_state,
            adata_ctrl
        ]
)

evaluator = MetricsEvaluator(
    adata_pred=adata_pred_ours,
    adata_real=adata_real,
    control_pert="non-targeting",
    pert_col="target_gene",
    num_threads=64,
    outdir = "./ours"
)
(results, agg_results) = evaluator.compute()

evaluator = MetricsEvaluator(
    adata_pred=adata_pred_state,
    adata_real=adata_real,
    control_pert="non-targeting",
    pert_col="target_gene",
    num_threads=64,
    outdir = "./state"
)
(results, agg_results) = evaluator.compute()