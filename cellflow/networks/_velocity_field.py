import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from cellflow._types import Layers_separate_input_t, Layers_t
from cellflow.networks._set_encoders import ConditionEncoder, FlaxGraphEncoder, GraphPerturbationTokenFusion
from cellflow.networks._utils import FilmBlock, MLPBlock, ResNetBlock, sinusoidal_time_encoder

__all__ = ["ConditionalVelocityField"]


class ConditionalVelocityField(nn.Module):
    """Parameterized neural vector field with conditions."""

    output_dim: int
    max_combination_length: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 1.0
    condition_embedding_dim: int = 32
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=lambda: [])
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    cond_output_dropout: float = 0.0
    mask_value: float = 0.0
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    use_graph_encoder: bool = False
    graph_encoder_dim: int = 200
    graph_encoder_max_seq_len: int = 16906
    graph_encoder_gene2vec_file: str = "data/gene2vec_16906.npy"
    graph_encoder_gene_ids_file: str = "/home/zhangshibo24s/cell_flow/data_train/selected_genes_27k.txt"
    graph_encoder_gene2go_graph_file: str = "/home/zhangshibo24s/cell_flow/data_train/human_ens_gene2go_graph.csv"
    graph_encoder_max_edges: int = 200000
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_max_period: int = 10000
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    conditioning: Literal["concatenation", "film", "resnet"] = "concatenation"
    conditioning_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    def setup(self):
        """Initialize the network."""
        if isinstance(self.conditioning_kwargs, dataclasses.Field):
            conditioning_kwargs = dict(self.conditioning_kwargs.default_factory())
        else:
            conditioning_kwargs = dict(self.conditioning_kwargs)

        condition_encoder_kwargs = dict(self.condition_encoder_kwargs)
        x_graph_fusion_kwargs = condition_encoder_kwargs.pop("x_graph_fusion_kwargs", None)
        condition_graph_fusion_kwargs = condition_encoder_kwargs.pop("condition_graph_fusion_kwargs", None)

        self.use_condition_graph_fusion = False
        if condition_graph_fusion_kwargs is None:
            condition_graph_fusion_kwargs = x_graph_fusion_kwargs
        if (
            condition_graph_fusion_kwargs is not None
            and hasattr(condition_graph_fusion_kwargs, "get")
            and condition_graph_fusion_kwargs.get("enabled", False)
        ):
            self.use_condition_graph_fusion = True
            self.condition_graph_covariate_key = condition_graph_fusion_kwargs.get("covariate_key", "gene_perturbation")
            graph_dim = int(condition_graph_fusion_kwargs.get("dim", self.graph_encoder_dim))
            graph_seq_len = int(condition_graph_fusion_kwargs.get("max_seq_len", self.output_dim))
            graph_gene2vec_file = condition_graph_fusion_kwargs.get("gene2vec_file", self.graph_encoder_gene2vec_file)
            graph_gene_ids_file = condition_graph_fusion_kwargs.get("gene_ids_file", self.graph_encoder_gene_ids_file)
            graph_gene2go_file = condition_graph_fusion_kwargs.get("gene2go_graph_file", self.graph_encoder_gene2go_graph_file)
            graph_max_edges = int(condition_graph_fusion_kwargs.get("max_edges", self.graph_encoder_max_edges))
            graph_neighborhood_only = bool(condition_graph_fusion_kwargs.get("neighborhood_only", False))
            self.condition_graph_fusion = GraphPerturbationTokenFusion(
                dim=graph_dim,
                max_seq_len=graph_seq_len,
                gene2vec_file=graph_gene2vec_file,
                gene_ids_file=graph_gene_ids_file,
                gene2go_graph_file=graph_gene2go_file,
                max_edges=graph_max_edges,
                neighborhood_only=graph_neighborhood_only,
            )

        self.condition_encoder = ConditionEncoder(
            condition_mode=self.condition_mode,
            regularization=self.regularization,
            output_dim=self.condition_embedding_dim,
            pooling=self.pooling,
            pooling_kwargs=self.pooling_kwargs,
            layers_before_pool=self.layers_before_pool,
            layers_after_pool=self.layers_after_pool,
            output_dropout=self.cond_output_dropout,
            covariates_not_pooled=self.covariates_not_pooled,
            mask_value=self.mask_value,
            **condition_encoder_kwargs,
        )

        self.layer_cond_output_dropout = nn.Dropout(rate=self.cond_output_dropout)
        self.layer_norm_condition = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.use_x_graph_fusion = False
        if x_graph_fusion_kwargs is not None:
            if hasattr(x_graph_fusion_kwargs, "get") and x_graph_fusion_kwargs.get("enabled", False):
                self.use_x_graph_fusion = True
                graph_dim = int(x_graph_fusion_kwargs.get("dim", self.graph_encoder_dim))
                graph_seq_len = int(x_graph_fusion_kwargs.get("max_seq_len", self.output_dim))
                graph_gene2vec_file = x_graph_fusion_kwargs.get("gene2vec_file", self.graph_encoder_gene2vec_file)
                graph_gene_ids_file = x_graph_fusion_kwargs.get("gene_ids_file", self.graph_encoder_gene_ids_file)
                graph_gene2go_file = x_graph_fusion_kwargs.get("gene2go_graph_file", self.graph_encoder_gene2go_graph_file)
                graph_max_edges = int(x_graph_fusion_kwargs.get("max_edges", self.graph_encoder_max_edges))
                self.x_graph_encoder = FlaxGraphEncoder(
                    dim=graph_dim,
                    max_seq_len=graph_seq_len,
                    gene2vec_file=graph_gene2vec_file,
                    gene_ids_file=graph_gene_ids_file,
                    gene2go_graph_file=graph_gene2go_file,
                    max_edges=graph_max_edges,
                )
                self.graph_query_proj = nn.Dense(graph_dim)
                self.x_graph_proj = nn.Dense(self.hidden_dims[-1])
                self.x_graph_gate = nn.Dense(self.hidden_dims[-1])

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

        if self.conditioning == "film":
            self.film_block = FilmBlock(
                input_dim=self.hidden_dims[-1],
                cond_dim=self.time_encoder_dims[-1] + self.condition_embedding_dim,
                **conditioning_kwargs,
            )
        elif self.conditioning == "resnet":
            self.resnet_block = ResNetBlock(
                input_dim=self.hidden_dims[-1],
                **conditioning_kwargs,
            )
        elif self.conditioning == "concatenation":
            if len(conditioning_kwargs) > 0:
                raise ValueError("If `conditioning=='concatenation' mode, no conditioning kwargs can be passed.")
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}")

    def _apply_condition_graph_fusion(self, cond: dict[str, jnp.ndarray], train: bool) -> dict[str, jnp.ndarray]:
        if not self.use_condition_graph_fusion or self.condition_graph_covariate_key not in cond:
            return cond
        cond = dict(cond)
        cond[self.condition_graph_covariate_key] = self.condition_graph_fusion(
            cond[self.condition_graph_covariate_key],
            deterministic=not train,
        )
        return cond

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        squeeze = x_t.ndim == 1
        cond = self._apply_condition_graph_fusion(cond, train=train)
        cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
        if self.condition_mode == "deterministic":
            cond_embedding = cond_mean
        else:
            cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)

        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)

        t_encoded = sinusoidal_time_encoder(t, time_freqs=self.time_freqs, time_max_period=self.time_max_period)
        t_encoded = self.time_encoder(t_encoded, training=train)
        x_encoded = self.x_encoder(x_t, training=train)
        if self.use_x_graph_fusion:
            x_graph_input = jnp.expand_dims(x_t, 0) if squeeze else x_t
            node_features, _ = self.x_graph_encoder(x_graph_input, deterministic=not train)

            c_emb = cond_embedding
            if c_emb.ndim == 1:
                c_emb = jnp.expand_dims(c_emb, 0)
            if c_emb.shape[0] != x_graph_input.shape[0]:
                c_emb = jnp.tile(c_emb, (x_graph_input.shape[0], 1))

            query = self.graph_query_proj(c_emb)
            attn_logits = jnp.einsum("bd,bsd->bs", query, node_features) / jnp.sqrt(query.shape[-1])
            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            graph_pooled = jnp.einsum("bs,bsd->bd", attn_weights, node_features)

            graph_proj = self.x_graph_proj(graph_pooled)
            if squeeze:
                graph_proj = jnp.squeeze(graph_proj, 0)

            gate_input = jnp.concatenate((x_encoded, graph_proj), axis=-1)
            gate = nn.sigmoid(self.x_graph_gate(gate_input))
            x_encoded = gate * x_encoded + (1.0 - gate) * graph_proj

        t_encoded = self.layer_norm_time(t_encoded)
        x_encoded = self.layer_norm_x(x_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)
        elif cond_embedding.shape[0] != x_t.shape[0]:
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        if self.conditioning == "concatenation":
            out = jnp.concatenate((t_encoded, x_encoded, cond_embedding), axis=-1)
        elif self.conditioning == "film":
            out = self.film_block(x_encoded, jnp.concatenate((t_encoded, cond_embedding), axis=-1))
        elif self.conditioning == "resnet":
            out = self.resnet_block(x_encoded, jnp.concatenate((t_encoded, cond_embedding), axis=-1))
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}.")

        out = self.decoder(out, training=train)
        return self.output_layer(out), cond_mean, cond_logvar

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        condition = self._apply_condition_graph_fusion(condition, train=False)
        condition_mean, condition_logvar = self.condition_encoder(condition, training=False)
        return condition_mean, condition_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        t, x_t = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        params_rng, condition_encoder_rng = jax.random.split(rng, 2)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng},
            t=t,
            x_t=x_t,
            cond=cond,
            encoder_noise=encoder_noise,
            train=False,
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

    @property
    def output_dims(self):
        """Dimensions of the output layers."""
        return tuple(self.decoder_dims) + (self.output_dim,)

    @property
    def time_encoder(self):
        """The time encoder used."""
        return self._time_encoder

    @time_encoder.setter
    def time_encoder(self, encoder):
        """Set the time encoder."""
        self._time_encoder = encoder

    @property
    def x_encoder(self):
        """The x encoder used."""
        return self._x_encoder

    @x_encoder.setter
    def x_encoder(self, encoder):
        """Set the x encoder."""
        self._x_encoder = encoder

    @property
    def decoder(self):
        """The decoder used."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        """Set the decoder."""
        self._decoder = decoder
