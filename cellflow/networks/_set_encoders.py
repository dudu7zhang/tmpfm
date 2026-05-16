from collections.abc import Sequence
import csv
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

from cellflow._types import ArrayLike, Layers_separate_input_t, Layers_t
from cellflow.networks import _utils as nn_utils

__all__ = [
    "TwoLayerMLP",
    "Gene2VecPositionalEmbedding",
    "FlaxGraphEncoder",
    "GraphPerturbationTokenFusion",
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
    gene2vec_file: str = "/home/zhangshibo24s/cell_flow/data_train/selected_gene2vec_27k.npy"
    gene_ids_file: str = "/home/zhangshibo24s/cell_flow/data_train/selected_genes_27k.txt"
    gene2go_graph_file: str = "/home/zhangshibo24s/cell_flow/data_train/human_ens_gene2go_graph.csv"
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


class GraphPerturbationTokenFusion(nn.Module):
    """Fuse perturbation gene tokens with GO-neighborhood context."""

    dim: int = 200
    max_seq_len: int = 16906
    gene2vec_file: str = "/home/zhangshibo24s/cell_flow/data_train/selected_gene2vec_27k.npy"
    gene_ids_file: str = "/home/zhangshibo24s/cell_flow/data_train/selected_genes_27k.txt"
    gene2go_graph_file: str = "/home/zhangshibo24s/cell_flow/data_train/human_ens_gene2go_graph.csv"
    max_edges: int = 200000
    neighborhood_only: bool = False

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
        del deterministic
        base_nodes = self.gene2vec_weight
        graph_nodes = base_nodes
        if self.has_edges:
            msg = graph_nodes[self.edge_src] * self.edge_w[:, None]
            agg = jnp.zeros_like(graph_nodes).at[self.edge_tgt].add(msg)
            graph_nodes = nn.LayerNorm()(graph_nodes + agg)

        query = nn.Dense(self.dim)(tokens)
        attn_logits = jnp.einsum("bld,gd->blg", query, graph_nodes) / jnp.sqrt(float(self.dim))
        if self.neighborhood_only and self.has_edges:
            token_for_match = tokens if tokens.shape[-1] == self.dim else nn.Dense(self.dim, name="match_projection")(tokens)
            token_norm = token_for_match / (jnp.linalg.norm(token_for_match, axis=-1, keepdims=True) + 1e-8)
            node_norm = base_nodes / (jnp.linalg.norm(base_nodes, axis=-1, keepdims=True) + 1e-8)
            pert_idx = jnp.argmax(jnp.einsum("bld,gd->blg", token_norm, node_norm), axis=-1)

            outgoing = self.edge_src[None, None, :] == pert_idx[..., None]
            incoming = self.edge_tgt[None, None, :] == pert_idx[..., None]
            neighborhood_mask = jnp.zeros_like(attn_logits, dtype=bool)
            neighborhood_mask = neighborhood_mask.at[:, :, self.edge_tgt].max(outgoing)
            neighborhood_mask = neighborhood_mask.at[:, :, self.edge_src].max(incoming)
            neighborhood_mask = neighborhood_mask | jax.nn.one_hot(pert_idx, self.max_seq_len).astype(bool)
            attn_logits = jnp.where(neighborhood_mask, attn_logits, -1e9)

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
