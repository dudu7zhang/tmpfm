import anndata as ad
import numpy as np

adata = ad.read_h5ad('data_train/replogle.h5ad', backed='r')
print("Unique cell lines:")
if 'cell_line' in adata.obs:
    print(adata.obs['cell_line'].unique())
