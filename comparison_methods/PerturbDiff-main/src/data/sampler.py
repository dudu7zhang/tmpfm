"""Unified sampler definitions and exports for cell-set batching."""

import math
import logging
from typing import Iterator, Optional, Tuple

import numpy as np
import omegaconf
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


Triplet = Tuple[str, int, int]  # (ds_name, local_idx, is_padded_value)
logger = logging.getLogger(__name__)


class DistributedCellSetFixPairingBatchSampler(Sampler[Triplet]):
    """Distributedcellsetfixpairingbatchsampler implementation used by the PerturbDiff pipeline."""
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """Special method `__init__`."""
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        logger.warning("this sampler is ignoring the drop_last argument")

        self._build_blocks(for_epoch=False)
        self.num_samples = math.ceil(self.total_samples / self.num_replicas)
        logger.warning("after cellset resampling, results in total cells: %s", self.num_samples)

    def __len__(self) -> int:
        """Special method `__len__`."""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Execute `set_epoch` and return values used by downstream logic."""
        self.epoch = int(epoch)
        self._build_blocks(for_epoch=True)

    def _build_blocks(self, for_epoch: bool = False):
        """Execute `_build_blocks` and return values used by downstream logic."""
        if "keep_one_sample" in self.dataset.data_args and self.dataset.data_args.keep_one_sample:
            assert 0, "not supported now"

        rng = np.random.default_rng(self.seed + (self.epoch if for_epoch else 0))

        assert not isinstance(self.dataset.data_args.use_cell_set, omegaconf.dictconfig.DictConfig)
        all_block_indices, all_block_ignored = [], []
        all_block_dsname_idx, dsname_ref = [], []

        self.total_samples = 0
        original_total = 0

        for ds_name in self.dataset.dataset_path_map.keys():
            real_batch_size = self.dataset.data_args.use_cell_set
            if real_batch_size is None:
                real_batch_size = 1

            dsname_ref.append(ds_name)

            groups = self.dataset.grouped_pert_data_indices[ds_name]
            for keys, val in groups.items():
                if len(val) == 0:
                    continue
                original_total += len(val)

                if self.dataset.control_type[ds_name] is None:
                    if "cellxgene" in ds_name.lower():
                        celltype_code = keys
                    else:
                        raise NotImplementedError
                else:
                    pert_code, celltype_code, batch_code = keys

                if self.shuffle and for_epoch:
                    rng.shuffle(val)

                if len(val) % real_batch_size != 0:
                    total_padded = int((len(val) + real_batch_size - 1) // real_batch_size) * real_batch_size - len(val)

                    group_seed = rng.integers(np.iinfo(np.uint32).max, dtype=np.uint32)
                    group_rng = np.random.default_rng(group_seed)
                    pad_samples = group_rng.choice(val, size=total_padded, replace=True)

                    idxs = np.concatenate([val, pad_samples], axis=0)
                    ignored = np.concatenate(
                        [
                            np.zeros(len(val), dtype=np.int32),
                            np.ones(total_padded, dtype=np.int32),
                        ]
                    )
                else:
                    idxs = val
                    ignored = np.zeros(len(val), dtype=np.int32)

                all_block_indices.append(idxs)
                all_block_ignored.append(ignored)
                all_block_dsname_idx.append(np.ones(len(idxs), dtype=np.int32) * len(dsname_ref))

                self.total_samples += len(idxs)

        logger.warning("replicating results in %s from originally %s", self.total_samples, original_total)
        self._all_block_indices = np.concatenate(all_block_indices)
        self._all_block_ignored = np.concatenate(all_block_ignored)
        self._all_block_dsname_idx = np.concatenate(all_block_dsname_idx)
        self.dsname_ref = dsname_ref
        logger.info("dsname_ref: %s", self.dsname_ref)

    def __iter__(self) -> Iterator[Triplet]:
        """Special method `__iter__`."""
        real_batch_size = self.dataset.data_args.use_cell_set
        if real_batch_size is None:
            real_batch_size = 1
        assert len(self._all_block_indices) % real_batch_size == 0
        n_block = len(self._all_block_indices) // real_batch_size
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n_block, generator=g).tolist()
        else:
            logger.warning(
                "still shuffling within batches for validation. Therefore, do not use this sampler for generation"
            )
            indices = list(range(n_block))

        indices = indices[self.rank::self.num_replicas]

        for idx in indices:
            for i in range(idx * real_batch_size, (idx + 1) * real_batch_size):
                gidx, ign, ds_name_idx = self._all_block_indices[i], self._all_block_ignored[i], self._all_block_dsname_idx[i]
                ds_name_idx = ds_name_idx - 1
                ds_name_hint = self.dsname_ref[ds_name_idx]
                ds_name, local_idx = self.dataset._compute_index(ds_name_hint, gidx)
                assert ds_name == ds_name_hint, "ds_name should be consistent"
                yield (ds_name, local_idx, ign)


class CellSetBatchSampler(DistributedCellSetFixPairingBatchSampler):
    """Cellsetbatchsampler implementation used by the PerturbDiff pipeline."""
    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """Special method `__init__`."""
        logger.warning("this sampler should only be used when doing sampling")

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        logger.warning("this sampler is ignoring the drop_last argument")

        self._build_blocks(for_epoch=False)
        assert not dist.is_initialized()
        self.num_samples = self.total_samples

    def __iter__(self) -> Iterator[Triplet]:
        """Special method `__iter__`."""
        real_batch_size = self.dataset.data_args.use_cell_set
        if real_batch_size is None:
            real_batch_size = 1
        assert len(self._all_block_indices) % real_batch_size == 0
        n_block = len(self._all_block_indices) // real_batch_size
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n_block, generator=g).tolist()
        else:
            logger.warning(
                "still shuffling within batches for validation. Therefore, do not use this sampler for generation"
            )
            indices = list(range(n_block))

        for idx in indices:
            for i in range(idx * real_batch_size, (idx + 1) * real_batch_size):
                gidx, ign, ds_name_idx = self._all_block_indices[i], self._all_block_ignored[i], self._all_block_dsname_idx[i]
                ds_name_idx = ds_name_idx - 1
                ds_name_hint = self.dsname_ref[ds_name_idx]
                ds_name, local_idx = self.dataset._compute_index(ds_name_hint, gidx)
                assert ds_name == ds_name_hint, "ds_name should be consistent"
                yield (ds_name, local_idx, ign)

__all__ = [
    "Triplet",
    "DistributedCellSetFixPairingBatchSampler",
    "CellSetBatchSampler",
]
