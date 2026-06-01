from collections.abc import Sequence
import csv
import heapq
from pathlib import Path
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from flax.typing import FrozenDict

from myflow._types import ArrayLike, Layers_separate_input_t, Layers_t
from myflow.networks import _utils as nn_utils

__all__ = [
    "TwoLayerMLP",
    "Gene2VecPositionalEmbedding",
    "FlaxGraphEncoder",
    "GOResponsePriorEncoder",
    "PerturbationGraphPriorEncoder",
    "GraphPerturbationTokenFusion",
    "OptimizedGraphEncoder",
    "OptimizedGraphPerturbationFusion",
    "ConditionEncoder",
]


class TwoLayerMLP(nn.Module):
    """Two-layer MLP for mapping scalar gene values to dense features."""

    hidden_dim: int = 50
    output_dim: int = 200

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class Gene2VecPositionalEmbedding(nn.Module):
    """Gene2Vec positional embedding loaded from a numpy file."""

    max_seq_len: int
    gene2vec_file: str = "data/gene2vec_16906.npy"

    def setup(self):
        gene2vec_weight = np.load(self.gene2vec_file)
        gene2vec_weight = np.concatenate(
            (gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))),
            axis=0,
        )
        self.gene2vec_weight = jnp.array(gene2vec_weight, dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len} for Gene2Vec embedding."
            )
        t = jnp.arange(seq_len)
        return self.gene2vec_weight[t]


class FlaxGraphEncoder(nn.Module):
    """Graph-style encoder with expression lifting + gene2vec + sparse graph propagation."""

    dim: int = 200
    max_seq_len: int = 16906
    gene2vec_file: str = ""
    gene_ids_file: str = ""
    gene2go_graph_file: str = ""
    max_edges: int = 200000

    def setup(self):
        gene2vec_weight = np.load(self.gene2vec_file).astype(np.float32)
        if gene2vec_weight.shape[0] < self.max_seq_len:
            raise ValueError(
                f"Gene2Vec rows ({gene2vec_weight.shape[0]}) are smaller than max_seq_len ({self.max_seq_len})."
            )
        if gene2vec_weight.shape[1] != self.dim:
            raise ValueError(
                f"Gene2Vec dimension ({gene2vec_weight.shape[1]}) must match encoder dim ({self.dim})."
            )

        with open(self.gene_ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip().upper() for line in f if line.strip()]
        if len(ids) < self.max_seq_len:
            raise ValueError(
                f"Gene ID list size ({len(ids)}) is smaller than max_seq_len ({self.max_seq_len})."
            )

        ids = ids[: self.max_seq_len]
        self.gene2vec_weight = jnp.array(gene2vec_weight[: self.max_seq_len], dtype=jnp.float32)

        id_to_idx = {gid: i for i, gid in enumerate(ids)}
        edge_src: list[int] = []
        edge_tgt: list[int] = []
        edge_w: list[float] = []

        graph_path = Path(self.gene2go_graph_file)
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = str(row.get("source", "")).upper()
                    tgt = str(row.get("target", "")).upper()
                    if src in id_to_idx and tgt in id_to_idx:
                        edge_src.append(id_to_idx[src])
                        edge_tgt.append(id_to_idx[tgt])
                        edge_w.append(float(row.get("importance", 1.0)))

        if edge_src:
            src_arr = np.asarray(edge_src, dtype=np.int32)
            tgt_arr = np.asarray(edge_tgt, dtype=np.int32)
            w_arr = np.asarray(edge_w, dtype=np.float32)

            if self.max_edges > 0 and src_arr.shape[0] > self.max_edges:
                top_idx = np.argpartition(w_arr, -self.max_edges)[-self.max_edges :]
                src_arr = src_arr[top_idx]
                tgt_arr = tgt_arr[top_idx]
                w_arr = w_arr[top_idx]

            deg = np.zeros((self.max_seq_len,), dtype=np.float32)
            np.add.at(deg, tgt_arr, w_arr)
            norm_w = w_arr / (deg[tgt_arr] + 1e-8)

            self.edge_src = jnp.array(src_arr, dtype=jnp.int32)
            self.edge_tgt = jnp.array(tgt_arr, dtype=jnp.int32)
            self.edge_w = jnp.array(norm_w, dtype=jnp.float32)
            self.has_edges = True
        else:
            self.edge_src = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_tgt = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_w = jnp.zeros((0,), dtype=jnp.float32)
            self.has_edges = False

    @nn.compact
    def __call__(self, x_expr: jnp.ndarray, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        del deterministic
        x = x_expr[..., None]
        seq_len = x_expr.shape[1]
        if seq_len != self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} must match configured max_seq_len {self.max_seq_len}."
            )

        x_embedded = TwoLayerMLP(output_dim=self.dim)(x)
        node_features = x_embedded + jnp.expand_dims(self.gene2vec_weight, axis=0)

        if self.has_edges:
            msg = node_features[:, self.edge_src, :] * self.edge_w[None, :, None]
            agg = jnp.zeros_like(node_features).at[:, self.edge_tgt, :].add(msg)
            node_features = nn.LayerNorm()(node_features + agg)

        pooled = jnp.mean(node_features, axis=1)
        return node_features, pooled


class GOResponsePriorEncoder(nn.Module):
    """Build gene-wise perturbation response priors in a shared GO latent space.

    Both output genes and perturbation genes share a single combined gene2vec
    table and GO graph.  After graph message passing the node embeddings are
    split into output-gene embeddings z_i and perturbation-gene embeddings z_p,
    ensuring they live in the same latent space.
    """

    dim: int = 200
    rho_dim: int = 128
    max_seq_len: int = 16906
    gene2vec_file: str = ""
    gene_ids_file: str = ""
    gene2go_graph_file: str = ""
    edge_cache_file: str = ""
    top_k: int = 20
    num_layers: int = 1
    weight_power: float = 1.0
    perturb_indices_in_combined: tuple[int, ...] | list[int] = ()

    def setup(self):
        gene2vec_weight = np.load(self.gene2vec_file).astype(np.float32)
        if gene2vec_weight.shape[1] != self.dim:
            raise ValueError(
                f"Gene2Vec dimension ({gene2vec_weight.shape[1]}) must match GO prior dim ({self.dim})."
            )

        with open(self.gene_ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip().upper() for line in f if line.strip()]

        num_all_genes = min(len(ids), gene2vec_weight.shape[0])
        ids = ids[:num_all_genes]
        id_to_idx = {gid: i for i, gid in enumerate(ids)}
        self.gene2vec_weight = jnp.array(gene2vec_weight[:num_all_genes], dtype=jnp.float32)

        # Output gene indices: first max_seq_len entries in the combined table
        self.output_indices = jnp.arange(min(self.max_seq_len, num_all_genes), dtype=jnp.int32)
        # Perturbation gene indices: provided by the training script
        self.perturb_indices = jnp.array(
            [i for i in self.perturb_indices_in_combined if 0 <= i < num_all_genes],
            dtype=jnp.int32,
        )

        # Pair-processing Dense layers
        self.rho_in = nn.Dense(self.rho_dim, name="rho_in")
        self.rho_out = nn.Dense(self.rho_dim, name="rho_out")
        self.synergy_in = nn.Dense(self.rho_dim, name="synergy_in")
        self.synergy_out = nn.Dense(self.rho_dim, name="synergy_out")

        # Graph message passing layers
        self.graph_norms = [nn.LayerNorm(name=f"go_prior_norm_{i}") for i in range(max(int(self.num_layers), 1))]
        self.graph_mlps = [nn.Dense(self.dim, name=f"go_prior_mlp_{i}") for i in range(max(int(self.num_layers), 1) - 1)]

        cache_path = Path(self.edge_cache_file) if self.edge_cache_file else None
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            src_arr = cached["edge_src"].astype(np.int32)
            tgt_arr = cached["edge_tgt"].astype(np.int32)
            norm_w = cached["edge_w"].astype(np.float32)
            if src_arr.shape != tgt_arr.shape or src_arr.shape != norm_w.shape:
                raise ValueError(f"Invalid GO edge cache shapes in {cache_path}.")
            self.edge_src = jnp.array(src_arr, dtype=jnp.int32)
            self.edge_tgt = jnp.array(tgt_arr, dtype=jnp.int32)
            self.edge_w = jnp.array(norm_w, dtype=jnp.float32)
            self.has_edges = bool(src_arr.shape[0] > 0)
            return

        per_target: dict[int, list[tuple[float, int]]] = {}
        graph_path = Path(self.gene2go_graph_file)
        if graph_path.exists():
            keep_k = max(int(self.top_k), 0) + 1
            with open(graph_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = str(row.get("source", "")).upper()
                    tgt = str(row.get("target", "")).upper()
                    if src not in id_to_idx or tgt not in id_to_idx:
                        continue
                    src_idx = id_to_idx[src]
                    tgt_idx = id_to_idx[tgt]
                    weight = float(row.get("importance", 1.0))
                    if self.weight_power != 1.0:
                        weight = weight ** self.weight_power
                    heap = per_target.setdefault(tgt_idx, [])
                    item = (weight, src_idx)
                    if len(heap) < keep_k:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)

        edge_src: list[int] = []
        edge_tgt: list[int] = []
        edge_w: list[float] = []
        for tgt_idx, heap in per_target.items():
            for weight, src_idx in heap:
                edge_src.append(src_idx)
                edge_tgt.append(tgt_idx)
                edge_w.append(weight)

        if edge_src:
            src_arr = np.asarray(edge_src, dtype=np.int32)
            tgt_arr = np.asarray(edge_tgt, dtype=np.int32)
            w_arr = np.asarray(edge_w, dtype=np.float32)
            deg = np.zeros((num_all_genes,), dtype=np.float32)
            np.add.at(deg, tgt_arr, w_arr)
            norm_w = w_arr / (deg[tgt_arr] + 1e-8)

            self.edge_src = jnp.array(src_arr, dtype=jnp.int32)
            self.edge_tgt = jnp.array(tgt_arr, dtype=jnp.int32)
            self.edge_w = jnp.array(norm_w, dtype=jnp.float32)
            self.has_edges = True
        else:
            self.edge_src = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_tgt = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_w = jnp.zeros((0,), dtype=jnp.float32)
            self.has_edges = False

    def __call__(
        self,
        perturb_tokens: jnp.ndarray,
        deterministic: bool = True,
        perturb_indices: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        del deterministic
        # --- Graph message passing on ALL genes (output + perturb) ---
        graph_nodes = self.gene2vec_weight
        if self.has_edges:
            for layer_idx in range(max(int(self.num_layers), 1)):
                msg = graph_nodes[self.edge_src] * self.edge_w[:, None]
                agg = jnp.zeros_like(graph_nodes).at[self.edge_tgt].add(msg)
                graph_nodes = self.graph_norms[layer_idx](graph_nodes + agg)
                if layer_idx < self.num_layers - 1:
                    graph_nodes = nn.relu(self.graph_mlps[layer_idx](graph_nodes))

        # --- Split into output genes (z_i) and perturbation genes (z_p) ---
        z_i_nodes = graph_nodes[self.output_indices]  # (num_output_genes, dim)

        # Select per-cell perturbation gene from graph-encoded table
        if perturb_indices is not None:
            # perturb_indices: (B,) — index into combined table for each cell's perturbation
            idx = jnp.clip(perturb_indices.ravel(), 0, graph_nodes.shape[0] - 1)
            z_p = graph_nodes[idx]  # (B, dim) — one perturbation gene per cell
        else:
            # Fallback during init: use first perturbation gene for all cells
            z_p = graph_nodes[self.perturb_indices[0:1]]  # (1, dim)
            z_p = jnp.broadcast_to(z_p, (perturb_tokens.shape[0], z_p.shape[-1]))

        # Pair features: z_i (O, dim) × z_p (B, dim) → (B, O, 4*dim)
        z_i_exp = jnp.broadcast_to(z_i_nodes[None, :, :], (z_p.shape[0], z_i_nodes.shape[0], z_i_nodes.shape[1]))
        z_p_exp = jnp.broadcast_to(z_p[:, None, :], z_i_exp.shape)  # (B, O, dim)
        pair = jnp.concatenate([z_i_exp, z_p_exp, z_i_exp * z_p_exp,
                                jnp.abs(z_i_exp - z_p_exp)], axis=-1)  # (B, O, 4*dim)

        rho = self.rho_in(pair)
        rho = nn.silu(rho)
        rho = self.rho_out(rho)  # (B, O, rho_dim)

        return rho


class PerturbationGraphPriorEncoder(nn.Module):
    """Perturbation-level PPI graph prior encoder.

    Loads a STRING PPI graph between perturbation genes and produces
    per-output-gene response features rho via pair features + MLP.
    Parallel to GOResponsePriorEncoder but operates on perturbation
    gene functional interactions rather than GO term similarity.
    """

    dim: int = 200
    rho_dim: int = 128
    max_seq_len: int = 16906
    gene2vec_file: str = ""
    gene_ids_file: str = ""
    ppi_edge_file: str = ""
    edge_cache_file: str = ""
    num_layers: int = 1
    top_k: int = 0
    perturb_indices_in_combined: tuple[int, ...] | list[int] = ()

    def setup(self):
        gene2vec_weight = np.load(self.gene2vec_file).astype(np.float32)
        if gene2vec_weight.shape[1] != self.dim:
            raise ValueError(
                f"Gene2Vec dimension ({gene2vec_weight.shape[1]}) must match dim ({self.dim})."
            )

        with open(self.gene_ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip().upper() for line in f if line.strip()]

        num_all_genes = min(len(ids), gene2vec_weight.shape[0])
        ids = ids[:num_all_genes]
        id_to_idx = {gid: i for i, gid in enumerate(ids)}
        self.gene2vec_weight = jnp.array(gene2vec_weight[:num_all_genes], dtype=jnp.float32)

        self.output_indices = jnp.arange(min(self.max_seq_len, num_all_genes), dtype=jnp.int32)
        self.perturb_indices = jnp.array(
            [i for i in self.perturb_indices_in_combined if 0 <= i < num_all_genes],
            dtype=jnp.int32,
        )

        # Pair-processing Dense layers
        self.rho_in = nn.Dense(self.rho_dim, name="pert_rho_in")
        self.rho_out = nn.Dense(self.rho_dim, name="pert_rho_out")

        # Graph message passing layers
        self.graph_norms = [nn.LayerNorm(name=f"pert_graph_norm_{i}") for i in range(max(int(self.num_layers), 1))]
        self.graph_mlps = [nn.Dense(self.dim, name=f"pert_graph_mlp_{i}") for i in range(max(int(self.num_layers), 1) - 1)]

        # Try loading from edge cache first
        cache_path = Path(self.edge_cache_file) if self.edge_cache_file else None
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            src_arr = cached["edge_src"].astype(np.int32)
            tgt_arr = cached["edge_tgt"].astype(np.int32)
            norm_w = cached["edge_w"].astype(np.float32)
            if src_arr.shape != tgt_arr.shape or src_arr.shape != norm_w.shape:
                raise ValueError(f"Invalid PPI edge cache shapes in {cache_path}.")
            self.edge_src = jnp.array(src_arr, dtype=jnp.int32)
            self.edge_tgt = jnp.array(tgt_arr, dtype=jnp.int32)
            self.edge_w = jnp.array(norm_w, dtype=jnp.float32)
            self.has_edges = bool(src_arr.shape[0] > 0)
            return

        # Build perturbation-only subgraph from PPI file
        perturb_set = set(int(i) for i in self.perturb_indices.tolist())
        edge_src: list[int] = []
        edge_tgt: list[int] = []
        edge_w: list[float] = []

        ppi_path = Path(self.ppi_edge_file)
        if ppi_path.exists() and ppi_path.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(ppi_path)
            for _, row in df.iterrows():
                src_gene = str(row.get("regulator", "")).upper()
                tgt_gene = str(row.get("target", "")).upper()
                if src_gene not in id_to_idx or tgt_gene not in id_to_idx:
                    continue
                src_idx = id_to_idx[src_gene]
                tgt_idx = id_to_idx[tgt_gene]
                # Only keep edges between perturbation genes
                if src_idx not in perturb_set or tgt_idx not in perturb_set:
                    continue
                weight = float(row.get("weight", 1.0))
                edge_src.append(src_idx)
                edge_tgt.append(tgt_idx)
                edge_w.append(weight)
        elif ppi_path.exists():
            # CSV format fallback
            with open(ppi_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src_gene = str(row.get("regulator", row.get("source", ""))).upper()
                    tgt_gene = str(row.get("target", "")).upper()
                    if src_gene not in id_to_idx or tgt_gene not in id_to_idx:
                        continue
                    src_idx = id_to_idx[src_gene]
                    tgt_idx = id_to_idx[tgt_gene]
                    if src_idx not in perturb_set or tgt_idx not in perturb_set:
                        continue
                    weight = float(row.get("weight", row.get("importance", 1.0)))
                    edge_src.append(src_idx)
                    edge_tgt.append(tgt_idx)
                    edge_w.append(weight)

        if edge_src:
            src_arr = np.asarray(edge_src, dtype=np.int32)
            tgt_arr = np.asarray(edge_tgt, dtype=np.int32)
            w_arr = np.asarray(edge_w, dtype=np.float32)
            deg = np.zeros((num_all_genes,), dtype=np.float32)
            np.add.at(deg, tgt_arr, w_arr)
            norm_w = w_arr / (deg[tgt_arr] + 1e-8)

            self.edge_src = jnp.array(src_arr, dtype=jnp.int32)
            self.edge_tgt = jnp.array(tgt_arr, dtype=jnp.int32)
            self.edge_w = jnp.array(norm_w, dtype=jnp.float32)
            self.has_edges = True
        else:
            self.edge_src = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_tgt = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_w = jnp.zeros((0,), dtype=jnp.float32)
            self.has_edges = False

    def __call__(
        self,
        perturb_tokens: jnp.ndarray,
        deterministic: bool = True,
        perturb_indices: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        del deterministic
        # --- Graph message passing on ALL genes (to update perturbation gene embeddings) ---
        graph_nodes = self.gene2vec_weight
        if self.has_edges:
            for layer_idx in range(max(int(self.num_layers), 1)):
                msg = graph_nodes[self.edge_src] * self.edge_w[:, None]
                agg = jnp.zeros_like(graph_nodes).at[self.edge_tgt].add(msg)
                graph_nodes = self.graph_norms[layer_idx](graph_nodes + agg)
                if layer_idx < self.num_layers - 1:
                    graph_nodes = nn.relu(self.graph_mlps[layer_idx](graph_nodes))

        # --- Split into output genes (z_i) and perturbation genes (z_p) ---
        z_i_nodes = graph_nodes[self.output_indices]

        if perturb_indices is not None:
            idx = jnp.clip(perturb_indices.ravel(), 0, graph_nodes.shape[0] - 1)
            z_p = graph_nodes[idx]
        else:
            z_p = graph_nodes[self.perturb_indices[0:1]]
            z_p = jnp.broadcast_to(z_p, (perturb_tokens.shape[0], z_p.shape[-1]))

        # Pair features: z_i (O, dim) x z_p (B, dim) -> (B, O, 4*dim)
        z_i_exp = jnp.broadcast_to(z_i_nodes[None, :, :], (z_p.shape[0], z_i_nodes.shape[0], z_i_nodes.shape[1]))
        z_p_exp = jnp.broadcast_to(z_p[:, None, :], z_i_exp.shape)
        pair = jnp.concatenate([z_i_exp, z_p_exp, z_i_exp * z_p_exp,
                                jnp.abs(z_i_exp - z_p_exp)], axis=-1)

        rho = self.rho_in(pair)
        rho = nn.silu(rho)
        rho = self.rho_out(rho)

        return rho


class GraphPerturbationTokenFusion(nn.Module):
    """Fuse perturbation gene tokens with GO-neighborhood context.

    Supports two attention modes:
    - Full attention (neighborhood_only=False): attend to all genes, then mask to neighbors.
    - Neighborhood-only attention (neighborhood_only=True): gather k-hop GO neighbors
      and compute attention only over that small set (~50-100 genes). Much faster.
    """

    dim: int = 200
    max_seq_len: int = 16906
    gene2vec_file: str = ""
    gene_ids_file: str = ""
    gene2go_graph_file: str = ""
    max_edges: int = 200000
    neighborhood_only: bool = True
    neighborhood_hops: int = 2
    max_neighbors: int = 128

    def setup(self):
        gene2vec_weight = np.load(self.gene2vec_file).astype(np.float32)
        if gene2vec_weight.shape[0] < self.max_seq_len:
            raise ValueError(
                f"Gene2Vec rows ({gene2vec_weight.shape[0]}) are smaller than max_seq_len ({self.max_seq_len})."
            )
        if gene2vec_weight.shape[1] != self.dim:
            raise ValueError(
                f"Gene2Vec dimension ({gene2vec_weight.shape[1]}) must match graph fusion dim ({self.dim})."
            )

        with open(self.gene_ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip().upper() for line in f if line.strip()]
        if len(ids) < self.max_seq_len:
            raise ValueError(
                f"Gene ID list size ({len(ids)}) is smaller than max_seq_len ({self.max_seq_len})."
            )

        ids = ids[: self.max_seq_len]
        self.gene2vec_weight = jnp.array(gene2vec_weight[: self.max_seq_len], dtype=jnp.float32)

        id_to_idx = {gid: i for i, gid in enumerate(ids)}
        edge_src: list[int] = []
        edge_tgt: list[int] = []
        edge_w: list[float] = []

        graph_path = Path(self.gene2go_graph_file)
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = str(row.get("source", "")).upper()
                    tgt = str(row.get("target", "")).upper()
                    if src in id_to_idx and tgt in id_to_idx:
                        edge_src.append(id_to_idx[src])
                        edge_tgt.append(id_to_idx[tgt])
                        edge_w.append(float(row.get("importance", 1.0)))

        if edge_src:
            src_arr = np.asarray(edge_src, dtype=np.int32)
            tgt_arr = np.asarray(edge_tgt, dtype=np.int32)
            w_arr = np.asarray(edge_w, dtype=np.float32)

            if self.max_edges > 0 and src_arr.shape[0] > self.max_edges:
                top_idx = np.argpartition(w_arr, -self.max_edges)[-self.max_edges :]
                src_arr = src_arr[top_idx]
                tgt_arr = tgt_arr[top_idx]
                w_arr = w_arr[top_idx]

            deg = np.zeros((self.max_seq_len,), dtype=np.float32)
            np.add.at(deg, tgt_arr, w_arr)
            norm_w = w_arr / (deg[tgt_arr] + 1e-8)

            self.edge_src = jnp.array(src_arr, dtype=np.int32)
            self.edge_tgt = jnp.array(tgt_arr, dtype=jnp.int32)
            self.edge_w = jnp.array(norm_w, dtype=jnp.float32)
            self.has_edges = True

            # Precompute adjacency list for k-hop neighborhood lookup
            adj: dict[int, list[tuple[int, float]]] = {}
            for s, t, w in zip(src_arr.tolist(), tgt_arr.tolist(), norm_w.tolist()):
                adj.setdefault(s, []).append((t, w))
                adj.setdefault(t, []).append((s, w))  # undirected

            # Build k-hop neighborhood for all nodes, padded to max_neighbors
            # Using -1 as padding index; we'll add a zero row at index max_seq_len
            neighbor_idx = np.full((self.max_seq_len, self.max_neighbors), self.max_seq_len, dtype=np.int32)
            neighbor_w = np.zeros((self.max_seq_len, self.max_neighbors), dtype=np.float32)

            for node in range(self.max_seq_len):
                if node not in adj:
                    continue
                # BFS for k-hop
                visited: dict[int, float] = {}
                frontier = {node: 1.0}
                for _ in range(self.neighborhood_hops):
                    next_frontier: dict[int, float] = {}
                    for cur, cur_w in frontier.items():
                        for nb, nb_w in adj.get(cur, []):
                            combined_w = cur_w * nb_w
                            if nb not in visited or combined_w > visited[nb]:
                                visited[nb] = combined_w
                                next_frontier[nb] = combined_w
                    frontier = next_frontier

                # Remove self, sort by weight, take top-k
                visited.pop(node, None)
                if not visited:
                    continue
                sorted_nbs = sorted(visited.items(), key=lambda x: -x[1])[: self.max_neighbors]
                for i, (nb, w) in enumerate(sorted_nbs):
                    neighbor_idx[node, i] = nb
                    neighbor_w[node, i] = w

            self.neighbor_idx = jnp.array(neighbor_idx, dtype=jnp.int32)
            self.neighbor_w_arr = jnp.array(neighbor_w, dtype=jnp.float32)
        else:
            self.edge_src = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_tgt = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_w = jnp.zeros((0,), dtype=jnp.float32)
            self.has_edges = False
            self.neighbor_idx = jnp.zeros((self.max_seq_len, self.max_neighbors), dtype=jnp.int32)
            self.neighbor_w_arr = jnp.zeros((self.max_seq_len, self.max_neighbors), dtype=jnp.float32)

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        del deterministic
        base_nodes = self.gene2vec_weight
        graph_nodes = base_nodes
        if self.has_edges:
            msg = graph_nodes[self.edge_src] * self.edge_w[:, None]
            agg = jnp.zeros_like(graph_nodes).at[self.edge_tgt].add(msg)
            graph_nodes = nn.LayerNorm()(graph_nodes + agg)

        # Add a zero-padding row at index max_seq_len for safe gather
        pad_row = jnp.zeros((1, self.dim), dtype=graph_nodes.dtype)
        graph_nodes_padded = jnp.concatenate([graph_nodes, pad_row], axis=0)

        if self.neighborhood_only and self.has_edges:
            # Find which graph node each token best matches (cosine similarity)
            token_for_match = tokens if tokens.shape[-1] == self.dim else nn.Dense(self.dim, name="match_projection")(tokens)
            token_norm = token_for_match / (jnp.linalg.norm(token_for_match, axis=-1, keepdims=True) + 1e-8)
            node_norm = base_nodes / (jnp.linalg.norm(base_nodes, axis=-1, keepdims=True) + 1e-8)
            pert_idx = jnp.argmax(jnp.einsum("bld,gd->blg", token_norm, node_norm), axis=-1)

            # Gather k-hop neighbors for each perturbation token
            # pert_idx: (batch, seq_len) -> neighbor indices: (batch, seq_len, max_neighbors)
            nb_idx = self.neighbor_idx[pert_idx]   # (batch, seq_len, max_neighbors)
            nb_w = self.neighbor_w_arr[pert_idx]   # (batch, seq_len, max_neighbors)
            nb_nodes = graph_nodes_padded[nb_idx]  # (batch, seq_len, max_neighbors, dim)

            # Neighborhood attention: tokens attend only to their k-hop neighbors
            query = nn.Dense(self.dim)(tokens)  # (batch, seq_len, dim)
            attn_logits = jnp.einsum("bld,blnd->bln", query, nb_nodes) / jnp.sqrt(float(self.dim))

            # Mask padding neighbors
            pad_mask = (nb_idx < self.max_seq_len).astype(jnp.float32)  # (batch, seq_len, max_neighbors)
            attn_logits = jnp.where(pad_mask > 0, attn_logits, -1e9)

            attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (batch, seq_len, max_neighbors)
            # Weight by edge importance
            attn_weights = attn_weights * nb_w
            attn_weights = attn_weights / (jnp.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)

            graph_context = jnp.einsum("bln,blnd->bld", attn_weights, nb_nodes)  # (batch, seq_len, dim)
        else:
            # Full attention (original behavior)
            query = nn.Dense(self.dim)(tokens)
            attn_logits = jnp.einsum("bld,gd->blg", query, graph_nodes) / jnp.sqrt(float(self.dim))
            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            graph_context = jnp.einsum("blg,gd->bld", attn_weights, graph_nodes)

        graph_context = nn.Dense(tokens.shape[-1])(graph_context)
        gate = nn.sigmoid(nn.Dense(tokens.shape[-1])(jnp.concatenate([tokens, graph_context], axis=-1)))
        fused = gate * tokens + (1.0 - gate) * graph_context

        token_mask = jnp.any(tokens != 0.0, axis=-1, keepdims=True)
        return jnp.where(token_mask, fused, tokens)


class OptimizedGraphEncoder(nn.Module):
    """Optimized graph encoder with graph dropout and multi-scale residual propagation.

    Novel features:
    1. Graph Dropout: Randomly drops edges during training for regularization
    2. Multi-scale Residual: 2-layer propagation with residual connections
    3. HVG-only subgraph: Operates on reduced gene set (e.g., 5000 instead of 27000)
    """

    dim: int = 128
    max_seq_len: int = 5000
    gene2vec_file: str = ""
    gene_ids_file: str = ""
    gene2go_graph_file: str = ""
    max_edges: int = 50000
    graph_dropout: float = 0.2
    num_layers: int = 2

    def setup(self):
        gene2vec_weight = np.load(self.gene2vec_file).astype(np.float32)
        if gene2vec_weight.shape[0] < self.max_seq_len:
            raise ValueError(
                f"Gene2Vec rows ({gene2vec_weight.shape[0]}) are smaller than max_seq_len ({self.max_seq_len})."
            )
        if gene2vec_weight.shape[1] != self.dim:
            raise ValueError(
                f"Gene2Vec dimension ({gene2vec_weight.shape[1]}) must match encoder dim ({self.dim})."
            )

        with open(self.gene_ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip().upper() for line in f if line.strip()]
        if len(ids) < self.max_seq_len:
            raise ValueError(
                f"Gene ID list size ({len(ids)}) is smaller than max_seq_len ({self.max_seq_len})."
            )

        ids = ids[: self.max_seq_len]
        self.gene2vec_weight = jnp.array(gene2vec_weight[: self.max_seq_len], dtype=jnp.float32)

        id_to_idx = {gid: i for i, gid in enumerate(ids)}
        edge_src: list[int] = []
        edge_tgt: list[int] = []
        edge_w: list[float] = []

        graph_path = Path(self.gene2go_graph_file)
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = str(row.get("source", "")).upper()
                    tgt = str(row.get("target", "")).upper()
                    if src in id_to_idx and tgt in id_to_idx:
                        edge_src.append(id_to_idx[src])
                        edge_tgt.append(id_to_idx[tgt])
                        edge_w.append(float(row.get("importance", 1.0)))

        if edge_src:
            src_arr = np.asarray(edge_src, dtype=np.int32)
            tgt_arr = np.asarray(edge_tgt, dtype=np.int32)
            w_arr = np.asarray(edge_w, dtype=np.float32)

            if self.max_edges > 0 and src_arr.shape[0] > self.max_edges:
                top_idx = np.argpartition(w_arr, -self.max_edges)[-self.max_edges :]
                src_arr = src_arr[top_idx]
                tgt_arr = tgt_arr[top_idx]
                w_arr = w_arr[top_idx]

            deg = np.zeros((self.max_seq_len,), dtype=np.float32)
            np.add.at(deg, tgt_arr, w_arr)
            norm_w = w_arr / (deg[tgt_arr] + 1e-8)

            self.edge_src = jnp.array(src_arr, dtype=jnp.int32)
            self.edge_tgt = jnp.array(tgt_arr, dtype=jnp.int32)
            self.edge_w = jnp.array(norm_w, dtype=jnp.float32)
            self.has_edges = True
        else:
            self.edge_src = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_tgt = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_w = jnp.zeros((0,), dtype=jnp.float32)
            self.has_edges = False

    @nn.compact
    def __call__(self, x_expr: jnp.ndarray, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = x_expr[..., None]
        seq_len = x_expr.shape[1]
        if seq_len != self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} must match configured max_seq_len {self.max_seq_len}."
            )

        x_embedded = TwoLayerMLP(output_dim=self.dim)(x)
        node_features = x_embedded + jnp.expand_dims(self.gene2vec_weight, axis=0)

        if self.has_edges:
            # Multi-scale residual graph propagation
            for layer_idx in range(self.num_layers):
                # Graph dropout: randomly mask edges during training
                if not deterministic and self.graph_dropout > 0:
                    rng = self.make_rng("graph_dropout")
                    dropout_mask = jax.random.bernoulli(rng, p=1.0 - self.graph_dropout, shape=(self.edge_src.shape[0],))
                    edge_w = self.edge_w * dropout_mask
                else:
                    edge_w = self.edge_w

                # Message passing
                msg = node_features[:, self.edge_src, :] * edge_w[None, :, None]
                agg = jnp.zeros_like(node_features).at[:, self.edge_tgt, :].add(msg)

                # Residual connection + LayerNorm
                node_features = nn.LayerNorm(name=f"graph_layer_norm_{layer_idx}")(node_features + agg)

                # Optional: add a small MLP between layers
                if layer_idx < self.num_layers - 1:
                    node_features = nn.Dense(self.dim, name=f"graph_mlp_{layer_idx}")(node_features)
                    node_features = nn.relu(node_features)

        pooled = jnp.mean(node_features, axis=1)
        return node_features, pooled


class OptimizedGraphPerturbationFusion(nn.Module):
    """Optimized perturbation-side graph fusion with graph dropout and multi-scale propagation.

    Novel features:
    1. Graph Dropout: Randomly drops edges during training
    2. Multi-scale Residual: 2-layer propagation with residual connections
    3. Sparse Attention: Only attend to top-k neighbors instead of all nodes
    """

    dim: int = 128
    max_seq_len: int = 5000
    gene2vec_file: str = ""
    gene_ids_file: str = ""
    gene2go_graph_file: str = ""
    max_edges: int = 50000
    graph_dropout: float = 0.2
    num_layers: int = 2
    top_k_attn: int = 50

    def setup(self):
        gene2vec_weight = np.load(self.gene2vec_file).astype(np.float32)
        if gene2vec_weight.shape[0] < self.max_seq_len:
            raise ValueError(
                f"Gene2Vec rows ({gene2vec_weight.shape[0]}) are smaller than max_seq_len ({self.max_seq_len})."
            )
        if gene2vec_weight.shape[1] != self.dim:
            raise ValueError(
                f"Gene2Vec dimension ({gene2vec_weight.shape[1]}) must match graph fusion dim ({self.dim})."
            )

        with open(self.gene_ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip().upper() for line in f if line.strip()]
        if len(ids) < self.max_seq_len:
            raise ValueError(
                f"Gene ID list size ({len(ids)}) is smaller than max_seq_len ({self.max_seq_len})."
            )

        ids = ids[: self.max_seq_len]
        self.gene2vec_weight = jnp.array(gene2vec_weight[: self.max_seq_len], dtype=jnp.float32)

        id_to_idx = {gid: i for i, gid in enumerate(ids)}
        edge_src: list[int] = []
        edge_tgt: list[int] = []
        edge_w: list[float] = []

        graph_path = Path(self.gene2go_graph_file)
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = str(row.get("source", "")).upper()
                    tgt = str(row.get("target", "")).upper()
                    if src in id_to_idx and tgt in id_to_idx:
                        edge_src.append(id_to_idx[src])
                        edge_tgt.append(id_to_idx[tgt])
                        edge_w.append(float(row.get("importance", 1.0)))

        if edge_src:
            src_arr = np.asarray(edge_src, dtype=np.int32)
            tgt_arr = np.asarray(edge_tgt, dtype=np.int32)
            w_arr = np.asarray(edge_w, dtype=np.float32)

            if self.max_edges > 0 and src_arr.shape[0] > self.max_edges:
                top_idx = np.argpartition(w_arr, -self.max_edges)[-self.max_edges :]
                src_arr = src_arr[top_idx]
                tgt_arr = tgt_arr[top_idx]
                w_arr = w_arr[top_idx]

            deg = np.zeros((self.max_seq_len,), dtype=np.float32)
            np.add.at(deg, tgt_arr, w_arr)
            norm_w = w_arr / (deg[tgt_arr] + 1e-8)

            self.edge_src = jnp.array(src_arr, dtype=jnp.int32)
            self.edge_tgt = jnp.array(tgt_arr, dtype=jnp.int32)
            self.edge_w = jnp.array(norm_w, dtype=jnp.float32)
            self.has_edges = True
        else:
            self.edge_src = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_tgt = jnp.zeros((0,), dtype=jnp.int32)
            self.edge_w = jnp.zeros((0,), dtype=jnp.float32)
            self.has_edges = False

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        base_nodes = self.gene2vec_weight
        graph_nodes = base_nodes

        if self.has_edges:
            # Multi-scale residual graph propagation
            for layer_idx in range(self.num_layers):
                # Graph dropout
                if not deterministic and self.graph_dropout > 0:
                    rng = self.make_rng("graph_dropout")
                    dropout_mask = jax.random.bernoulli(rng, p=1.0 - self.graph_dropout, shape=(self.edge_src.shape[0],))
                    edge_w = self.edge_w * dropout_mask
                else:
                    edge_w = self.edge_w

                # Message passing
                msg = graph_nodes[self.edge_src] * edge_w[:, None]
                agg = jnp.zeros_like(graph_nodes).at[self.edge_tgt].add(msg)

                # Residual + LayerNorm
                graph_nodes = nn.LayerNorm(name=f"pert_graph_layer_norm_{layer_idx}")(graph_nodes + agg)

                # MLP between layers
                if layer_idx < self.num_layers - 1:
                    graph_nodes = nn.Dense(self.dim, name=f"pert_graph_mlp_{layer_idx}")(graph_nodes)
                    graph_nodes = nn.relu(graph_nodes)

        # Sparse attention: only attend to top-k neighbors
        query = nn.Dense(self.dim)(tokens)
        attn_logits = jnp.einsum("bld,gd->blg", query, graph_nodes) / jnp.sqrt(float(self.dim))

        if self.top_k_attn > 0 and self.top_k_attn < self.max_seq_len:
            # Keep only top-k attention weights
            top_k_vals, top_k_idx = jax.lax.top_k(attn_logits, self.top_k_attn)
            mask = jnp.zeros_like(attn_logits).at[:, :, top_k_idx].set(1.0)
            attn_logits = jnp.where(mask > 0, attn_logits, -1e9)

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        graph_context = jnp.einsum("blg,gd->bld", attn_weights, graph_nodes)

        graph_context = nn.Dense(tokens.shape[-1])(graph_context)
        gate = nn.sigmoid(nn.Dense(tokens.shape[-1])(jnp.concatenate([tokens, graph_context], axis=-1)))
        fused = gate * tokens + (1.0 - gate) * graph_context

        token_mask = jnp.any(tokens != 0.0, axis=-1, keepdims=True)
        return jnp.where(token_mask, fused, tokens)


class ConditionEncoder(nn_utils.BaseModule):
    """
    Encoder for conditions represented as sets of perturbations.

    Parameters
    ----------
    output_dim
        Dimensionality of the output.
    condition_mode
        Mode of the encoder, should be one of:

        - ``'deterministic'``: Learns condition encoding point-wise.
        - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.
    regularization
        Regularization strength in the latent space:

        - For deterministic mode, it is the strength of the L2 regularization.
        - For stochastic mode, it is the strength of the KL divergence regularization.
    decoder
        Whether to use a decoder.
    pooling
        Pooling method, should be one of:

        - ``'mean'``: Aggregates combinations of covariates by the mean of their learned
          embeddings.
        - ``'attention_token'``: Aggregates combinations of covariates by an attention mechanism
          with a token.
        - ``'attention_seed'``: Aggregates combinations of covariates by an attention mechanism
          with a seed.
    pooling_kwargs
        Keyword arguments for the pooling method.
    covariates_not_pooled
        Covariates that will escape pooling (should be identical across all set elements).
    layers_before_pool
        Layers before pooling. Either a sequence of tuples with layer type and parameters or a
        dictionary with input-specific layers.
    layers_after_pool
        Layers after pooling.
    layers_decoder
        Layers for the decoder. Only relevant if ``'decoder'=True``.
    mask_value
        Value for masked elements used in input conditions.
    """

    output_dim: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 0.0
    decoder: bool = False
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=list)
    layers_before_pool: Layers_t | Layers_separate_input_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    layers_decoder: Layers_t = dc_field(default_factory=lambda: [])
    output_dropout: float = 0.0
    mask_value: float = 0.0

    def setup(self):
        """Initialize the modules."""
        # modules before pooling
        self.separate_inputs = isinstance(self.layers_before_pool, (dict | FrozenDict))
        if self.separate_inputs:
            # different layers for different inputs, before_pool_modules is of type Layers_separate_input_t
            self.before_pool_modules: dict[str, list[nn.Module]] | list[nn.Module] = {
                key: nn_utils._get_layers(layers)
                for key, layers in self.layers_before_pool.items()  # type: ignore[union-attr]
            }
        else:
            self.before_pool_modules = nn_utils._get_layers(self.layers_before_pool)  # type: ignore[arg-type]

        # pooling
        if self.pooling == "mean":
            self.pool_module = lambda x, mask, training: jnp.mean(x * mask, axis=-2)
        elif self.pooling == "attention_token":
            self.pool_module = nn_utils.TokenAttentionPooling(**self.pooling_kwargs)
        elif self.pooling == "attention_seed":
            self.pool_module = nn_utils.SeedAttentionPooling(**self.pooling_kwargs)

        # modules after pooling
        self.after_pool_modules_mean = nn_utils._get_layers(self.layers_after_pool, self.output_dim)

        if self.condition_mode == "stochastic":
            self.after_pool_modules_var = nn_utils._get_layers(self.layers_after_pool, self.output_dim)

    def __call__(
        self,
        conditions: dict[str, jnp.ndarray],
        training: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply the set encoder.

        Parameters
        ----------
        conditions : dict[str, jnp.ndarray]
            Dictionary of batch of conditions of shape ``(batch_size, set_size, condition_dim)``.
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        Mean and log-variance of conditions of shape ``(batch_size, output_dim)``.
        """
        mask, attention_mask = self._get_masks(conditions)

        # apply modules before pooling
        if self.separate_inputs:
            processed_inputs_pooling = []
            processed_inputs_other = []
            for pert_cov, conditions_i in conditions.items():
                # apply separate modules for all inputs
                conditions_i = nn_utils._apply_modules(
                    self.before_pool_modules[pert_cov],  # type: ignore[call-overload]
                    conditions_i,
                    attention_mask,
                    training,
                )
                if pert_cov in self.covariates_not_pooled:
                    # only keep first set element for covariates that are not pooled
                    processed_inputs_other.append(conditions_i[:, 0, :])
                else:
                    processed_inputs_pooling.append(conditions_i)

            conditions_pooling_arr = jnp.concatenate(processed_inputs_pooling, axis=-1)
            conditions_not_pooled = (
                jnp.concatenate(processed_inputs_other, axis=-1) if self.covariates_not_pooled else None
            )
        else:
            # by default, no modules before pooling for covariates that are not pooled
            if self.covariates_not_pooled:
                # divide conditions into pooled and not pooled
                conditions_not_pooled = []
                conditions_pooling = []
                for pert_cov in conditions:
                    if pert_cov in self.covariates_not_pooled:
                        conditions_not_pooled.append(conditions[pert_cov][:, 0, :])
                    else:
                        conditions_pooling.append(conditions[pert_cov])
                conditions_not_pooled = jnp.concatenate(
                    conditions_not_pooled,
                    axis=-1,
                )
                conditions_pooling_arr = jnp.concatenate(
                    conditions_pooling,
                    axis=-1,
                )

                # apply modules to pooled covariates
                conditions_pooling_arr = nn_utils._apply_modules(
                    self.before_pool_modules,  # type: ignore[arg-type]
                    conditions_pooling_arr,
                    attention_mask,
                    training,
                )
            else:
                conditions = jnp.concatenate(list(conditions.values()), axis=-1)
                conditions_pooling_arr = nn_utils._apply_modules(
                    self.before_pool_modules,
                    conditions,
                    attention_mask,
                    training,  # type: ignore[arg-type]
                )

        # pooling
        pool_mask = mask if self.pooling == "mean" else attention_mask
        conditions = self.pool_module(conditions_pooling_arr, pool_mask, training=training)
        if self.covariates_not_pooled:
            conditions = jnp.concatenate([conditions, conditions_not_pooled], axis=-1)

        # apply modules after pooling
        conditions = nn_utils._apply_modules(self.after_pool_modules_mean, conditions, None, training)

        if self.condition_mode == "stochastic":
            conditions_logvar = nn_utils._apply_modules(self.after_pool_modules_var, conditions, None, training)
        else:
            conditions_logvar = jnp.zeros_like(conditions)
        return conditions, conditions_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        conditions: dict[str, jnp.ndarray],
        **kwargs: Any,
    ):
        """Create initial training state."""
        params = self.init(
            rng,
            conditions={k: jnp.empty((1, v.shape[1], v.shape[2])) for k, v in conditions.items()},
            training=False,
        )["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            **kwargs,
        )

    def _get_masks(self, conditions: dict[str, ArrayLike]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get mask for padded conditions tensor."""
        # mask of shape (batch_size, set_size)
        mask = 1 - jnp.all(
            jnp.array(
                [jnp.all(c == self.mask_value, axis=-1) for c in conditions.values()],
            ),
            axis=0,
        )
        mask = jnp.expand_dims(mask, -1)

        # attention mask of shape (batch_size, 1, set_size, set_size)
        attention_mask = mask & jnp.matrix_transpose(mask)
        attention_mask = jnp.expand_dims(attention_mask, 1)

        return mask, attention_mask
