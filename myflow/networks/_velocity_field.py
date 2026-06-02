import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from myflow._types import Layers_separate_input_t, Layers_t
from myflow.networks._set_encoders import (
    ConditionEncoder,
    GOResponsePriorEncoder,
    PerturbationGraphPriorEncoder,
    GraphPerturbationTokenFusion,
    OptimizedGraphPerturbationFusion,
)
from myflow.networks._utils import FilmBlock, MLPBlock, ResNetBlock, sinusoidal_time_encoder

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
    graph_encoder_gene_ids_file: str = ""
    graph_encoder_gene2go_graph_file: str = ""
    graph_encoder_max_edges: int = 200000
    go_response_enabled: bool = False
    go_response_dim: int = 128
    go_response_top_k: int = 20
    go_response_num_layers: int = 1
    go_response_weight_power: float = 1.5
    pert_graph_enabled: bool = False
    pert_graph_dim: int = 200
    pert_graph_rho_dim: int = 128
    pert_graph_num_layers: int = 1
    pert_graph_top_k: int = 0
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
        go_response_kwargs = condition_encoder_kwargs.pop("go_response_kwargs", None)
        pert_graph_kwargs = condition_encoder_kwargs.pop("pert_graph_kwargs", None)

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
            graph_neighborhood_hops = int(condition_graph_fusion_kwargs.get("neighborhood_hops", 2))
            graph_max_neighbors = int(condition_graph_fusion_kwargs.get("max_neighbors", 128))
            graph_dropout = float(condition_graph_fusion_kwargs.get("graph_dropout", 0.0))
            graph_num_layers = int(condition_graph_fusion_kwargs.get("num_layers", 1))
            graph_top_k_attn = int(condition_graph_fusion_kwargs.get("top_k_attn", 0))

            # Use optimized encoder if graph_dropout or num_layers > 1
            if graph_dropout > 0 or graph_num_layers > 1:
                self.condition_graph_fusion = OptimizedGraphPerturbationFusion(
                    dim=graph_dim,
                    max_seq_len=graph_seq_len,
                    gene2vec_file=graph_gene2vec_file,
                    gene_ids_file=graph_gene_ids_file,
                    gene2go_graph_file=graph_gene2go_file,
                    max_edges=graph_max_edges,
                    graph_dropout=graph_dropout,
                    num_layers=graph_num_layers,
                    top_k_attn=graph_top_k_attn,
                )
            else:
                self.condition_graph_fusion = GraphPerturbationTokenFusion(
                    dim=graph_dim,
                    max_seq_len=graph_seq_len,
                    gene2vec_file=graph_gene2vec_file,
                    gene_ids_file=graph_gene_ids_file,
                    gene2go_graph_file=graph_gene2go_file,
                    max_edges=graph_max_edges,
                    neighborhood_only=graph_neighborhood_only,
                    neighborhood_hops=graph_neighborhood_hops,
                    max_neighbors=graph_max_neighbors,
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

        self.use_go_response_prior = False
        if go_response_kwargs is None:
            go_response_kwargs = x_graph_fusion_kwargs
        if (
            self.go_response_enabled
            or (
                go_response_kwargs is not None
                and hasattr(go_response_kwargs, "get")
                and go_response_kwargs.get("enabled", False)
            )
        ):
            self.use_go_response_prior = True
            go_kwargs = go_response_kwargs if hasattr(go_response_kwargs, "get") else {}
            go_dim = int(go_kwargs.get("dim", self.graph_encoder_dim))
            go_seq_len = int(go_kwargs.get("max_seq_len", self.output_dim))
            go_gene2vec_file = go_kwargs.get("gene2vec_file", self.graph_encoder_gene2vec_file)
            go_gene_ids_file = go_kwargs.get("gene_ids_file", self.graph_encoder_gene_ids_file)
            go_gene2go_file = go_kwargs.get("gene2go_graph_file", self.graph_encoder_gene2go_graph_file)
            go_edge_cache_file = go_kwargs.get("edge_cache_file", "")
            go_rho_dim = int(go_kwargs.get("rho_dim", self.go_response_dim))
            go_top_k = int(go_kwargs.get("top_k", self.go_response_top_k))
            go_num_layers = int(go_kwargs.get("num_layers", self.go_response_num_layers))
            go_weight_power = float(go_kwargs.get("weight_power", self.go_response_weight_power))
            go_perturb_indices = tuple(go_kwargs.get("perturb_indices_in_combined", ()))
            self.go_response_covariate_key = go_kwargs.get("covariate_key", "gene_perturbation")
            self.exclude_go_response_from_base_condition = bool(
                go_kwargs.get("exclude_from_base_condition", False)
            )
            self.go_response_prior = GOResponsePriorEncoder(
                dim=go_dim,
                rho_dim=go_rho_dim,
                max_seq_len=go_seq_len,
                gene2vec_file=go_gene2vec_file,
                gene_ids_file=go_gene_ids_file,
                gene2go_graph_file=go_gene2go_file,
                edge_cache_file=go_edge_cache_file,
                top_k=go_top_k,
                num_layers=go_num_layers,
                weight_power=go_weight_power,
                perturb_indices_in_combined=go_perturb_indices,
            )
            self.gene_expr_encoder = nn.Dense(go_rho_dim)
            self.go_time_gate_proj = nn.Dense(go_rho_dim)
            self.go_rho_time_proj = nn.Dense(go_rho_dim)
            self.go_gamma_proj = nn.Dense(go_rho_dim)
            self.go_beta_proj = nn.Dense(go_rho_dim)
            self.go_gate_proj = nn.Dense(1)
            self.go_residual_head = nn.Dense(1)
            self.go_drive_norm = nn.LayerNorm()
            self.go_state_norm = nn.LayerNorm()
        else:
            self.go_response_covariate_key = "gene_perturbation"
            self.exclude_go_response_from_base_condition = False

        # --- Perturbation-level PPI graph prior ---
        self.use_pert_graph_prior = False
        if (
            self.pert_graph_enabled
            or (
                pert_graph_kwargs is not None
                and hasattr(pert_graph_kwargs, "get")
                and pert_graph_kwargs.get("enabled", False)
            )
        ):
            self.use_pert_graph_prior = True
            pg_kwargs = pert_graph_kwargs if hasattr(pert_graph_kwargs, "get") else {}
            pg_dim = int(pg_kwargs.get("dim", self.pert_graph_dim))
            pg_rho_dim = int(pg_kwargs.get("rho_dim", self.pert_graph_rho_dim))
            pg_seq_len = int(pg_kwargs.get("max_seq_len", self.output_dim))
            pg_gene2vec_file = pg_kwargs.get("gene2vec_file", self.graph_encoder_gene2vec_file)
            pg_gene_ids_file = pg_kwargs.get("gene_ids_file", self.graph_encoder_gene_ids_file)
            pg_ppi_file = pg_kwargs.get("ppi_edge_file", "")
            pg_edge_cache = pg_kwargs.get("edge_cache_file", "")
            pg_num_layers = int(pg_kwargs.get("num_layers", self.pert_graph_num_layers))
            pg_top_k = int(pg_kwargs.get("top_k", self.pert_graph_top_k))
            pg_perturb_indices = tuple(pg_kwargs.get("perturb_indices_in_combined", ()))
            self.pert_graph_covariate_key = pg_kwargs.get("covariate_key", "gene_perturbation")
            self.exclude_pert_graph_from_base_condition = bool(
                pg_kwargs.get("exclude_from_base_condition", False)
            )
            self.pert_graph_prior = PerturbationGraphPriorEncoder(
                dim=pg_dim,
                rho_dim=pg_rho_dim,
                max_seq_len=pg_seq_len,
                gene2vec_file=pg_gene2vec_file,
                gene_ids_file=pg_gene_ids_file,
                ppi_edge_file=pg_ppi_file,
                edge_cache_file=pg_edge_cache,
                num_layers=pg_num_layers,
                top_k=pg_top_k,
                perturb_indices_in_combined=pg_perturb_indices,
            )
            self.pert_gene_expr_encoder = nn.Dense(pg_rho_dim)
            self.pert_time_gate_proj = nn.Dense(pg_rho_dim)
            self.pert_rho_time_proj = nn.Dense(pg_rho_dim)
            self.pert_gamma_proj = nn.Dense(pg_rho_dim)
            self.pert_beta_proj = nn.Dense(pg_rho_dim)
            self.pert_gate_proj = nn.Dense(1)
            self.pert_residual_head = nn.Dense(1)
            self.pert_drive_norm = nn.LayerNorm()
            self.pert_state_norm = nn.LayerNorm()
        else:
            self.pert_graph_covariate_key = "gene_perturbation"
            self.exclude_pert_graph_from_base_condition = False

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

    _EXTRA_COND_KEYS = frozenset({"gene_perturbation_indices"})

    def _base_condition_inputs(self, cond: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        """Return condition inputs used by the base velocity path.

        By default, perturbation gene tokens enter both the base condition
        encoder and the GO response prior branch so that both branches
        can output velocity informed by perturbation information.
        """
        exclude = set(self._EXTRA_COND_KEYS)
        if self.exclude_go_response_from_base_condition:
            exclude.add(self.go_response_covariate_key)
        if self.use_pert_graph_prior and self.exclude_pert_graph_from_base_condition:
            exclude.add(self.pert_graph_covariate_key)
        if not exclude:
            return cond
        return {k: v for k, v in cond.items() if k not in exclude}

    def _zero_condition_embedding(self, x_t: jnp.ndarray, squeeze: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size = 1 if squeeze else x_t.shape[0]
        cond = jnp.zeros((batch_size, self.condition_embedding_dim), dtype=x_t.dtype)
        return cond, cond

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        squeeze = x_t.ndim == 1
        go_cond_source = cond
        cond_for_base = self._base_condition_inputs(cond)
        cond_for_base = self._apply_condition_graph_fusion(cond_for_base, train=train)
        if cond_for_base:
            cond_mean, cond_logvar = self.condition_encoder(cond_for_base, training=train)
        else:
            cond_mean, cond_logvar = self._zero_condition_embedding(x_t, squeeze=squeeze)
        if self.condition_mode == "deterministic":
            cond_embedding = cond_mean
        else:
            cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)

        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)

        t_encoded = sinusoidal_time_encoder(t, time_freqs=self.time_freqs, time_max_period=self.time_max_period)
        t_encoded = self.time_encoder(t_encoded, training=train)
        x_encoded = self.x_encoder(x_t, training=train)

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
        velocity = self.output_layer(out)

        if self.use_go_response_prior and self.go_response_covariate_key in go_cond_source:
            go_cond = go_cond_source[self.go_response_covariate_key]
            if go_cond.ndim == 2:
                go_cond = jnp.expand_dims(go_cond, 0)
            if not squeeze and go_cond.shape[0] != x_t.shape[0]:
                go_cond = jnp.tile(go_cond, (x_t.shape[0], 1, 1))
            pert_idx = go_cond_source.get("gene_perturbation_indices")
            go_valid_gate = None
            if pert_idx is not None:
                pert_idx = pert_idx.astype(jnp.int32)
                if pert_idx.ndim == 1:
                    pert_idx = jnp.expand_dims(pert_idx, 0)
                if not squeeze and pert_idx.shape[0] != x_t.shape[0]:
                    pert_idx = jnp.tile(pert_idx, (x_t.shape[0], 1, 1))
                pert_idx = jnp.squeeze(pert_idx, axis=-1) if pert_idx.ndim == 3 else pert_idx
                go_valid_gate = (jnp.any(pert_idx >= 0, axis=-1)).astype(x_t.dtype)
            rho = self.go_response_prior(go_cond, deterministic=not train, perturb_indices=pert_idx)

            x_gene = jnp.expand_dims(x_t, -1)
            gene_state = self.gene_expr_encoder(x_gene)

            time_gate = nn.sigmoid(self.go_time_gate_proj(t_encoded))
            if squeeze:
                time_gate = jnp.expand_dims(time_gate, 0)
                gene_state = jnp.expand_dims(gene_state, 0)
                velocity_base = jnp.expand_dims(velocity, 0)
            else:
                velocity_base = velocity

            rho_shift = self.go_rho_time_proj(rho)
            drive = self.go_drive_norm(rho + jnp.expand_dims(time_gate, axis=1) * rho_shift)
            gamma = nn.tanh(self.go_gamma_proj(drive))
            beta = self.go_beta_proj(drive)
            dynamic_gate = nn.sigmoid(self.go_gate_proj(drive))
            h = self.go_state_norm(gene_state * (1.0 + gamma) + beta)

            go_residual = dynamic_gate * self.go_residual_head(h)
            go_residual = jnp.squeeze(go_residual, axis=-1)
            if go_valid_gate is not None:
                go_residual = go_residual * go_valid_gate[:, None]
            velocity = velocity_base + go_residual
            if squeeze:
                velocity = jnp.squeeze(velocity, 0)

        # --- Perturbation-level PPI graph prior ---
        if self.use_pert_graph_prior and self.pert_graph_covariate_key in go_cond_source:
            pert_cond = go_cond_source[self.pert_graph_covariate_key]
            if pert_cond.ndim == 2:
                pert_cond = jnp.expand_dims(pert_cond, 0)
            if not squeeze and pert_cond.shape[0] != x_t.shape[0]:
                pert_cond = jnp.tile(pert_cond, (x_t.shape[0], 1, 1))
            pert_idx = go_cond_source.get("gene_perturbation_indices")
            pert_valid_gate = None
            if pert_idx is not None:
                pert_idx = pert_idx.astype(jnp.int32)
                if pert_idx.ndim == 1:
                    pert_idx = jnp.expand_dims(pert_idx, 0)
                if not squeeze and pert_idx.shape[0] != x_t.shape[0]:
                    pert_idx = jnp.tile(pert_idx, (x_t.shape[0], 1, 1))
                pert_idx = jnp.squeeze(pert_idx, axis=-1) if pert_idx.ndim == 3 else pert_idx
                pert_valid_gate = (jnp.any(pert_idx >= 0, axis=-1)).astype(x_t.dtype)
            rho_pert = self.pert_graph_prior(pert_cond, deterministic=not train, perturb_indices=pert_idx)

            x_gene_pert = jnp.expand_dims(x_t, -1) if x_t.ndim == 1 else jnp.expand_dims(x_t, -1)
            gene_state_pert = self.pert_gene_expr_encoder(x_gene_pert)

            time_gate_pert = nn.sigmoid(self.pert_time_gate_proj(t_encoded))
            if squeeze:
                time_gate_pert = jnp.expand_dims(time_gate_pert, 0)
                gene_state_pert = jnp.expand_dims(gene_state_pert, 0)
                velocity_for_pert = jnp.expand_dims(velocity, 0)
            else:
                velocity_for_pert = velocity

            rho_shift_pert = self.pert_rho_time_proj(rho_pert)
            drive_pert = self.pert_drive_norm(rho_pert + jnp.expand_dims(time_gate_pert, axis=1) * rho_shift_pert)
            gamma_pert = nn.tanh(self.pert_gamma_proj(drive_pert))
            beta_pert = self.pert_beta_proj(drive_pert)
            dynamic_gate_pert = nn.sigmoid(self.pert_gate_proj(drive_pert))
            h_pert = self.pert_state_norm(gene_state_pert * (1.0 + gamma_pert) + beta_pert)

            pert_residual = dynamic_gate_pert * self.pert_residual_head(h_pert)
            pert_residual = jnp.squeeze(pert_residual, axis=-1)
            if pert_valid_gate is not None:
                pert_residual = pert_residual * pert_valid_gate[:, None]
            velocity = velocity_for_pert + pert_residual
            if squeeze:
                velocity = jnp.squeeze(velocity, 0)

        return velocity, cond_mean, cond_logvar

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        original_condition = condition
        condition = self._base_condition_inputs(condition)
        condition = self._apply_condition_graph_fusion(condition, train=False)
        if condition:
            condition_mean, condition_logvar = self.condition_encoder(condition, training=False)
        else:
            first = next(iter(original_condition.values()), jnp.zeros((1, 1, 1), dtype=jnp.float32))
            batch_size = first.shape[0]
            condition_mean = jnp.zeros((batch_size, self.condition_embedding_dim), dtype=first.dtype)
            condition_logvar = jnp.zeros_like(condition_mean)
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
        params_rng, condition_encoder_rng, graph_dropout_rng = jax.random.split(rng, 3)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng, "graph_dropout": graph_dropout_rng},
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
