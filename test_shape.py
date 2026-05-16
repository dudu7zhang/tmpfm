import anndata as ad
import numpy as np

def check_cell_type_encoder():
    adata = ad.read_h5ad('/home/zhangshibo24s/cell_flow/data_train/replogle.h5ad', backed='r')
    print('unique cell lines:', adata.obs['cell_line'].unique())
    print('dtype of cell_line:', adata.obs['cell_line'].dtype)

check_cell_type_encoder()
