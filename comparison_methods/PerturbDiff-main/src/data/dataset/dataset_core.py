"""Module `data/dataset_core.py`."""
import ctypes
import ctypes.util
from logging import Logger
from typing import Any, Dict, List, Tuple

import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.data.dataset.dataset_grouping import (
    get_selected_gene_vars,
    register_mapping_indices,
    split_out_control,
)
from src.data.dataset.dataset_io import retrieve_counts
from src.data.file_handle import H5Store
from src.data.metadata_cache import GlobalH5MetadataCache

_libc_name = ctypes.util.find_library("c")
libc = ctypes.CDLL(_libc_name) if _libc_name else None


def compute_index(dataset, ds_name: str, idx: int):
    """Execute `compute_index` and return values used by downstream logic."""
    assert ds_name in dataset.data_indices and len(dataset.data_indices[ds_name]) > 0
    try:
        local = int(dataset.data_indices[ds_name][idx])
    except Exception:
        local = int(dataset.data_indices[ds_name]["perturb"][idx])
    return ds_name, int(local)


def collate_fn(dataset, batch):
    """Execute `collate_fn` and return values used by downstream logic."""
    bsz = len(batch)
    _ = bsz

    pca_mode = isinstance(dataset.data_args.embed_key, str) and dataset.data_args.embed_key.startswith("X_pca")
    _ = pca_mode

    assert dataset.data_args.pad_length == dataset.data_args.embed_shape, "Not supporting padding truncation anymore"
    assert len(batch[0][0]) == 1

    batch_counts, batch_mapped_counts = [], []
    pert_var_list, mapped_pert_var_list = [], []
    cell_type_var_list, batch_var_list = [], []
    is_padded_list = []
    ds_name_list = []
    col_gene_list = []

    for (
        counts,
        mapped_counts,
        gene_vars,
        pert_var,
        mapped_pert_var,
        cell_type_var,
        batch_var,
        is_padded,
        tmp_ds_name,
    ) in batch:
        ds_name_list.append(tmp_ds_name)
        col_gene_list.append(gene_vars)

        pert_var_list.append(pert_var)
        mapped_pert_var_list.append(mapped_pert_var)
        cell_type_var_list.append(cell_type_var)
        batch_var_list.append(batch_var)
        is_padded_list.append(is_padded)

        batch_counts.append(counts[0])
        batch_mapped_counts.append(mapped_counts[0])

    batch_counts = torch.stack(batch_counts)
    batch_mapped_counts = torch.stack(batch_mapped_counts)

    if dataset.data_args.normalize_counts is not None and dataset.data_args.normalize_counts:
        batch_counts /= dataset.data_args.normalize_counts
        batch_mapped_counts /= dataset.data_args.normalize_counts

    pert_var_list = torch.as_tensor(pert_var_list, dtype=torch.long)
    mapped_pert_var_list = torch.as_tensor(mapped_pert_var_list, dtype=torch.long)
    cell_type_var_list = torch.as_tensor(cell_type_var_list, dtype=torch.long)
    batch_var_list = torch.as_tensor(batch_var_list, dtype=torch.long)
    is_padded_list = torch.as_tensor(is_padded_list, dtype=torch.long)

    if dataset.data_args.use_cell_set is not None:
        s_val = dataset.data_args.use_cell_set

        batch_counts = batch_counts.reshape(-1, s_val, batch_counts.shape[-1])
        batch_mapped_counts = batch_mapped_counts.reshape(-1, s_val, batch_mapped_counts.shape[-1])
        pert_var_list = pert_var_list.reshape(-1, s_val)
        mapped_pert_var_list = mapped_pert_var_list.reshape(-1, s_val)
        cell_type_var_list = cell_type_var_list.reshape(-1, s_val)
        batch_var_list = batch_var_list.reshape(-1, s_val)
        is_padded_list = is_padded_list.reshape(-1, s_val)

        if dataset.data_args.use_cell_set > 1:
            assert ds_name_list[:: dataset.data_args.use_cell_set] == ds_name_list[1:: dataset.data_args.use_cell_set]
            assert col_gene_list[:: dataset.data_args.use_cell_set] == col_gene_list[1:: dataset.data_args.use_cell_set]

        ds_name_list = ds_name_list[:: dataset.data_args.use_cell_set]
        col_gene_list = col_gene_list[:: dataset.data_args.use_cell_set]

    return {
        "pert_emb": batch_counts,
        "col_genes": col_gene_list,
        "cont_emb": batch_mapped_counts,
        "cov_pert": pert_var_list,
        "cov_mapped_pert": mapped_pert_var_list,
        "cov_celltype": cell_type_var_list,
        "cov_batch": batch_var_list,
        "is_padded_list": is_padded_list,
        "ds_name": ds_name_list,
    }


def mapping_cells(dataset, ds_name, local_idx):
    """Execute `mapping_cells` and return values used by downstream logic."""
    assert dataset.data_args.mapping_strategy == "random"
    randint = lambda high: int(np.random.randint(0, high, size=(1))[0])
    choice = lambda arr: np.random.choice(arr, 1)[0]

    cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

    key1 = cache.cell_type_codes[local_idx]
    key2 = cache.batch_codes[local_idx]

    n_cells = dataset.grouped_num_cell[ds_name][(key1, key2)]
    if n_cells == 0:
        select_key1 = []
        for (i_val, j_val), v in dataset.grouped_num_cell[ds_name].items():
            if j_val == key2 and v != 0:
                select_key1.append(i_val)
        try:
            assert len(select_key1) > 0
            key1 = choice(select_key1)
        except Exception:
            select_key2 = []
            for (i_val, j_val), v in dataset.grouped_num_cell[ds_name].items():
                if i_val == key1 and v != 0:
                    select_key2.append(j_val)
            assert len(select_key2) > 0
            key2 = choice(select_key2)

        n_cells = dataset.grouped_num_cell[ds_name][(key1, key2)]

    selected_idx = randint(n_cells)
    if dataset.control_type[ds_name] is not None:
        selected_local_idx = dataset.data_indices[ds_name]["control"][
            dataset.grouped_data_indices[ds_name][(key1, key2)][selected_idx]
        ]
    else:
        selected_local_idx = dataset.data_indices[ds_name][dataset.grouped_data_indices[ds_name][(key1, key2)][selected_idx]]

    return int(selected_local_idx)


def get_covarates(dataset, ds_name, local_idx):
    """Execute `get_covarates` and return values used by downstream logic."""
    cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

    try:
        pert_idx = cache.pert_codes[local_idx]
        pert_type: str = cache.pert_categories[pert_idx]
        pert_var = dataset.meta_cache.pert_dict[pert_type]
    except Exception:
        pert_var = -1

    try:
        cell_type_idx = cache.cell_type_codes[local_idx]
        cell_type: str = cache.cell_type_categories[cell_type_idx]
        cell_type_var = dataset.meta_cache.cell_type_dict[cell_type]
    except Exception:
        dataset.py_logger.info(
            "[Error]: both datasets should have cell type infos "
            "and codes. It's cell_type column for RNA-seq, while cell_line_id column for Perturb-seq"
        )
        assert 0

    try:
        batch_idx = cache.batch_codes[local_idx]
        global_batch_type = ds_name + "_" + cache.batch_categories[batch_idx]
        assert global_batch_type in dataset.meta_cache.batch_dict
        batch_var = dataset.meta_cache.batch_dict[global_batch_type]
    except Exception:
        dataset.py_logger.info("[Error]: both datasets should have batch infos and codes.")
        assert 0

    return pert_var, cell_type_var, batch_var


def getitem_impl(dataset, index):
    """Execute `getitem_impl` and return values used by downstream logic."""
    ds_name, local_idx, is_ignore = index

    h5f = dataset.store.dataset_file(ds_name)

    local_idx_list = [local_idx]
    is_padded_list = [is_ignore]

    counts_list, mapped_counts_list = [], []
    gene_var_list = []
    pert_var_list, mapped_pert_var_list = [], []
    cell_type_var_list, batch_var_list = [], []

    for local_idx in local_idx_list:
        counts = dataset._retrieve_counts(h5f, ds_name, local_idx)

        if dataset.control_type[ds_name] is None:
            mapped_local_idx = local_idx
            mapped_counts = counts.clone()
        else:
            mapped_local_idx = dataset._mapping_cells(ds_name, local_idx)
            mapped_counts = dataset._retrieve_counts(h5f, ds_name, mapped_local_idx)

        counts_list.append(counts)
        mapped_counts_list.append(mapped_counts)
        if isinstance(dataset.selected_genes, dict):
            gene_var_list.append(dataset.selected_genes[ds_name])
        else:
            gene_var_list.append(dataset.selected_genes)

        pert_var, cell_type_var, batch_var = dataset._get_covarates(ds_name, local_idx)
        mapped_pert_var, mapped_cell_type_var, mapped_batch_var = dataset._get_covarates(ds_name, mapped_local_idx)
        _ = mapped_cell_type_var
        _ = mapped_batch_var

        pert_var_list.append(pert_var)
        mapped_pert_var_list.append(mapped_pert_var)
        cell_type_var_list.append(cell_type_var)
        batch_var_list.append(batch_var)

    if libc is not None and hasattr(libc, "malloc_trim") and np.random.rand() < 0.1:
        libc.malloc_trim(0)

    return (
        counts_list[0],
        mapped_counts_list[0],
        gene_var_list[0],
        pert_var_list[0],
        mapped_pert_var_list[0],
        cell_type_var_list[0],
        batch_var_list[0],
        is_padded_list[0],
        [ds_name][0],
    )


class H5adSentenceDataset(Dataset):
    """
    A virtual dataset containing the indices for each h5/h5ad dataset,
    and during indexing, it has to iterate through all indices to find the correct
    dataset file and local index.
    """

    def __init__(
        self,
        stage: str,
        meta_cache: GlobalH5MetadataCache,
        dataset_path_map: Dict[str, str],
        selected_genes_list: Dict[str, list],
        data_indices: Dict[str, np.ndarray],
        num_cell: Dict[str, int],
        control_type: Dict[str, str],
        data_args: DictConfig,
        py_logger: Logger,
    ) -> None:
        """
        Initialize the class instance.

        :param stage: Input `stage` value.
        :param meta_cache: Input `meta_cache` value.
        :param dataset_path_map: Input `dataset_path_map` value.
        :param selected_genes_list: List of values used in this step.
        :param data_indices: Input `data_indices` value.
        :param num_cell: Count used to control loop/shape behavior.
        :param control_type: Input `control_type` value.
        :param data_args: Input `data_args` value.
        :param py_logger: Input `py_logger` value.
        :return: None.
        """
        super(H5adSentenceDataset, self).__init__()
        """
        control_type: a dict for each dataset, specify the perturbation label for control cell
        """
        self.stage = stage
        self.meta_cache = meta_cache
        self.data_indices = data_indices
        self.num_cell = num_cell
        self.control_type = control_type
        self.data_args = data_args
        self.py_logger = py_logger

        # fix order for datasets to ensure reproducibility
        self._names = list(data_indices.keys())

        # filter needed dataset paths
        self.dataset_path_map = {k: v for k, v in dataset_path_map.items() if k in set(self._names)}
        self.selected_genes_list = {k: v for k, v in selected_genes_list.items() if k in set(self._names)}

        # File handle store (read-only, process-local)
        self.store = H5Store(self.dataset_path_map, max_open=data_args.max_open_files)

        # split data_indices into perturb & control
        self.split_out_control()

        # group data_indices into different categories for perturb & control cell mapping
        self.register_mapping_indices()

        # cumulative number of cells for computing index
        self._cum = np.cumsum([self.num_cell[n] for n in self._names])
        self.total_num_cell = int(self._cum[-1]) if len(self._cum) else 0

        # mapping dataset_name -> integer id
        self.datasets_to_num = {n: i for i, n in enumerate(self._names)}

        # get gene variables for each dataset
        self._get_selected_gene_vars()

        self._obsm_dim = {}

    def split_out_control(self):
        """Execute `split_out_control` and return values used by downstream logic."""
        return split_out_control(self)

    def register_mapping_indices(self):
        """Execute `register_mapping_indices` and return values used by downstream logic."""
        return register_mapping_indices(self)

    def _get_selected_gene_vars(self):
        """Execute `_get_selected_gene_vars` and return values used by downstream logic."""
        return get_selected_gene_vars(self)

    def _compute_index(self, ds_name: str, idx: int):
        """Execute `_compute_index` and return values used by downstream logic."""
        return compute_index(self, ds_name, idx)

    def collate_fn(self, batch: List[Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]]):
        """Execute `collate_fn` and return values used by downstream logic."""
        return collate_fn(self, batch)

    def _mapping_cells(self, ds_name, local_idx):
        """Execute `_mapping_cells` and return values used by downstream logic."""
        return mapping_cells(self, ds_name, local_idx)

    def _retrieve_counts(self, h5f, ds_name, local_idx):
        """Execute `_retrieve_counts` and return values used by downstream logic."""
        return retrieve_counts(self, h5f, ds_name, local_idx)

    def _get_covarates(self, ds_name, local_idx):
        """Execute `_get_covarates` and return values used by downstream logic."""
        return get_covarates(self, ds_name, local_idx)

    def __getitem__(self, index):
        """Special method `__getitem__`."""
        return getitem_impl(self, index)

    def __len__(self) -> int:
        """Special method `__len__`."""
        return self.total_num_cell

    def close(self):
        """Execute `close` and return values used by downstream logic."""
        self.store.close_all()

    def __del__(self):
        """Special method `__del__`."""
        try:
            self.store.close_all()
        except Exception:
            pass
