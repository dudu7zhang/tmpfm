from myflow.preprocessing._pca import centered_pca, project_pca, reconstruct_pca
from myflow.preprocessing._preprocessing import annotate_compounds, encode_onehot, get_molecular_fingerprints
from myflow.preprocessing._wknn import compute_wknn, transfer_labels

try:
    from myflow.preprocessing._gene_emb import (
        GeneInfo,
        get_esm_embedding,
        prot_sequence_from_ensembl,
        protein_features_from_genes,
    )
except ImportError:
    GeneInfo = None
    get_esm_embedding = None
    prot_sequence_from_ensembl = None
    protein_features_from_genes = None
