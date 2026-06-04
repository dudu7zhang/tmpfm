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
from myflow.networks._set_encoders import ConditionEncoder, PerturbationGNN
from myflow.networks._utils import FilmBlock, MLPBlock, ResNetBlock, sinusoidal_time_encoder

__all__ = ["ConditionalVelocityField"]


class ConditionalVelocityField(nn.Module):
    """Conditional velocity field for flow matching.

    Encodes time, cell state, and perturbation condition independently,
    then fuses them through gene-level self-attention + condition→gene cross-attention.
    """

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

    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_max_period: int = 10000
    time_encoder_dims: Sequence[int] = (512, 512, 512)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (512, 512, 512)
    hidden_dropout: float = 0.0
    conditioning: Literal["concatenation", "film", "resnet"] = "concatenation"
    conditioning_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    decoder_dims: Sequence[int] = (512, 512, 512)
    decoder_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    # Perturbation-side GNN prior: GO+STRING graph over perturbation genes
    perturbation_gnn_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})

    # Perturbation-conditioned gene mask: sparse velocity output
    gene_mask_enabled: bool = True

    # Gene-level self-attention / cross-attention
    gene_attn_dim: int = 16
    gene_self_attn_heads: int = 4
    gene_self_attn_layers: int = 0  # O(d²), 0 to skip
    gene_self_attn_dropout: float = 0.0
    cross_attn_heads: int = 4
    cross_attn_layers: int = 1
    cross_attn_dropout: float = 0.0

    def _resolve_kwargs(self, value: Any) -> dict[str, Any]:
        """Resolve a dc_field default_factory or plain dict to a real dict."""
        if isinstance(value, dataclasses.Field):
            return value.default_factory()
        return dict(value) if value else {}

    def setup(self):
        conditioning_kwargs = self._resolve_kwargs(self.conditioning_kwargs)
        condition_encoder_kwargs = self._resolve_kwargs(self.condition_encoder_kwargs)
        for _key in ("go_response_kwargs", "pert_graph_kwargs", "x_graph_fusion_kwargs",
                      "condition_graph_fusion_kwargs"):
            condition_encoder_kwargs.pop(_key, None)

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

        # ---- Perturbation-side GNN: learnable node embeddings + GO/STRING graph ----
        gnn_config = getattr(self, 'x_gnn_config', None) or {}
        n_pert = gnn_config.get('num_pert_genes', 0)
        if n_pert > 0:
            self.pert_embeddings = self.param(
                "pert_embeddings",
                nn.initializers.normal(0.02),
                (n_pert, self.gene_attn_dim),
            )
            self.perturbation_gnn = PerturbationGNN(
                hidden_dim=self.gene_attn_dim,
                num_layers=2,
                edge_src=gnn_config.get('edge_src'),
                edge_tgt=gnn_config.get('edge_tgt'),
                edge_w=gnn_config.get('edge_w'),
            )
        else:
            self.perturbation_gnn = None

        self.layer_cond_output_dropout = nn.Dropout(rate=self.cond_output_dropout)
        self.layer_norm_condition = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        # ---- Learned gene identity embedding ----
        self.gene_val_proj = nn.Dense(self.gene_attn_dim)  # expression value → dim
        self.gene_id_emb = self.param(
            "gene_id_emb",
            nn.initializers.normal(0.02),
            (self.output_dim, self.gene_attn_dim),
        )
        self.gene_attn_norms = [
            nn.LayerNorm() for _ in range(self.gene_self_attn_layers)
        ]
        self.gene_attns = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.gene_self_attn_heads,
                qkv_features=self.gene_attn_dim,
                dropout_rate=self.gene_self_attn_dropout,
            )
            for _ in range(self.gene_self_attn_layers)
        ]
        self.gene_ffn_norms = [
            nn.LayerNorm() for _ in range(self.gene_self_attn_layers)
        ]
        self.gene_ffn_in = [
            nn.Dense(self.gene_attn_dim * 4) for _ in range(self.gene_self_attn_layers)
        ]
        self.gene_ffn_out = [
            nn.Dense(self.gene_attn_dim) for _ in range(self.gene_self_attn_layers)
        ]

        # ---- Condition → gene cross-attention (stackable) ----
        self.cross_q_projs = [
            nn.Dense(self.gene_attn_dim) for _ in range(self.cross_attn_layers)
        ]
        self.cross_attns = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.cross_attn_heads,
                qkv_features=self.gene_attn_dim,
                dropout_rate=self.cross_attn_dropout,
            )
            for _ in range(self.cross_attn_layers)
        ]
        self.cross_out_norms = [
            nn.LayerNorm() for _ in range(self.cross_attn_layers)
        ]

        # ---- Project cross-attn output to hidden_dims[-1] ----
        self.fusion_proj = nn.Dense(self.hidden_dims[-1])

        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

        # Perturbation-conditioned gene mask: learns which genes each perturbation affects
        self.gene_mask_head = nn.Dense(self.output_dim) if self.gene_mask_enabled else None

        if self.conditioning == "film":
            self.film_block = FilmBlock(
                input_dim=self.hidden_dims[-1],
                cond_dim=self.time_encoder_dims[-1],
                **conditioning_kwargs,
            )
        elif self.conditioning == "resnet":
            self.resnet_block = ResNetBlock(
                input_dim=self.hidden_dims[-1],
                **conditioning_kwargs,
            )
        elif self.conditioning == "concatenation":
            if conditioning_kwargs:
                raise ValueError("concatenation mode takes no conditioning_kwargs.")
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}")

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        squeeze = x_t.ndim == 1

        # Perturbation gene: look up learnable embeddings by index, then GNN
        if self.perturbation_gnn is not None and "gene_perturbation_indices" in cond:
            cond = dict(cond)
            cond["gene_perturbation"] = self.perturbation_gnn(
                cond["gene_perturbation_indices"],
                self.pert_embeddings,
                deterministic=not train,
            )
            cond.pop("gene_perturbation_indices", None)

        if cond:
            cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
        else:
            batch_size = 1 if squeeze else x_t.shape[0]
            cond_mean = jnp.zeros((batch_size, self.condition_embedding_dim), dtype=x_t.dtype)
            cond_logvar = jnp.zeros_like(cond_mean)

        if self.condition_mode == "deterministic":
            cond_embedding = cond_mean
        else:
            cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)

        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)

        t_encoded = sinusoidal_time_encoder(t, time_freqs=self.time_freqs, time_max_period=self.time_max_period)
        if squeeze:
            t_encoded = t_encoded[None, :]  # (freqs,) → (1, freqs)
        t_encoded = self.time_encoder(t_encoded, training=train)

        # ---- Broadcast cond_embedding to match cell batch size ----
        if squeeze:
            pass  # already (1, d) from condition_encoder
        elif cond_embedding.shape[0] != x_t.shape[0]:
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        # ---- Gene embedding: expression value + learnable gene identity ----
        if squeeze:
            x_t = x_t[None, :]  # (d,) → (1, d)
        h_val = self.gene_val_proj(x_t[:, :, None])   # (B, d, 16) — what
        h_id = self.gene_id_emb[None, :, :]           # (1, d, 16) — who
        h_genes = h_val + h_id

        for i in range(self.gene_self_attn_layers):
            # Self-attention
            residual = h_genes
            h_genes = self.gene_attn_norms[i](h_genes)
            h_genes = self.gene_attns[i](h_genes, deterministic=not train)
            h_genes = h_genes + residual
            # FFN
            residual = h_genes
            h_genes = self.gene_ffn_norms[i](h_genes)
            h_genes = nn.relu(self.gene_ffn_in[i](h_genes))
            h_genes = self.gene_ffn_out[i](h_genes)
            h_genes = h_genes + residual

        # ---- Condition → gene cross-attention: condition selects relevant genes ----
        # z_c (now broadcast to cell batch) queries the gene feature sequence
        h_cross = cond_embedding
        for i in range(self.cross_attn_layers):
            z_q = self.cross_q_projs[i](h_cross)[:, None, :]       # (B, 1, gene_attn_dim)
            h_cross = self.cross_attns[i](
                inputs_q=z_q, inputs_kv=h_genes, deterministic=not train,
            )                                                       # (B, 1, gene_attn_dim)
            h_cross = jnp.squeeze(h_cross, axis=1)                  # (B, gene_attn_dim)
            h_cross = self.cross_out_norms[i](h_cross)

        # ---- Project to hidden_dims[-1] ----
        x_encoded = self.fusion_proj(h_cross)
        x_encoded = self.act_fn(x_encoded)

        t_encoded = self.layer_norm_time(t_encoded)
        x_encoded = self.layer_norm_x(x_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)

        if self.conditioning == "concatenation":
            out = jnp.concatenate((t_encoded, x_encoded), axis=-1)
        elif self.conditioning == "film":
            out = self.film_block(x_encoded, t_encoded)
        elif self.conditioning == "resnet":
            out = self.resnet_block(x_encoded, t_encoded)

        out = self.decoder(out, training=train)
        velocity = self.output_layer(out)

        # Perturbation-conditioned gene mask: learns which genes each perturbation affects,
        # enforcing sparse, perturbation-specific velocity predictions.
        if self.gene_mask_head is not None:
            gene_mask = nn.sigmoid(self.gene_mask_head(cond_embedding))
            velocity = velocity * gene_mask

        if squeeze:
            velocity = jnp.squeeze(velocity, axis=0)

        return velocity, cond_mean, cond_logvar

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        if condition:
            condition_mean, condition_logvar = self.condition_encoder(condition, training=False)
        else:
            first = next(iter(condition.values()), jnp.zeros((1, 1, 1), dtype=jnp.float32))
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
        return tuple(self.decoder_dims) + (self.output_dim,)

    @property
    def time_encoder(self):
        return self._time_encoder

    @time_encoder.setter
    def time_encoder(self, encoder):
        self._time_encoder = encoder

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
