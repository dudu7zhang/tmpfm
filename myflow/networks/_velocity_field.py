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
    then combines them through concat/film/resnet conditioning.
    No per-gene prior modulation — priors belong on the perturbation side only.
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

    # Perturbation-side GNN prior (optional)
    perturbation_gnn_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})

    # Delta-gated encoding: focus on genes that deviate from control
    delta_gate_enabled: bool = True
    delta_gate_init_temp: float = 0.1  # temperature for |delta| softmax

    # Perturbation-conditioned gene mask: sparse velocity output
    gene_mask_enabled: bool = True

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

        # Perturbation-side GNN: enriches perturbation tokens with GO+STRING graph context
        gnn_kwargs = self._resolve_kwargs(self.perturbation_gnn_kwargs)
        if gnn_kwargs.get("enabled", False):
            gnn_kwargs.pop("enabled", None)
            self.perturbation_gnn = PerturbationGNN(**gnn_kwargs)
            self._gnn_covariate_key = gnn_kwargs.get("covariate_key", "gene_perturbation")
        else:
            self.perturbation_gnn = None
            self._gnn_covariate_key = "gene_perturbation"

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

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

        # Gene-level attention: soft focus on genes with expression deviating from mean
        self.delta_temp = self.param("delta_temp", nn.initializers.constant(0.1), ())
        self.delta_scale = self.param("delta_scale", nn.initializers.zeros, ())

        # Perturbation-conditioned gene mask: learns which genes each perturbation affects
        self.gene_mask_head = nn.Dense(self.output_dim) if self.gene_mask_enabled else None

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

        # Apply perturbation-side GNN to enrich perturbation tokens
        if self.perturbation_gnn is not None and self._gnn_covariate_key in cond:
            cond = dict(cond)
            cond[self._gnn_covariate_key] = self.perturbation_gnn(
                cond[self._gnn_covariate_key], deterministic=not train,
            )

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
        t_encoded = self.time_encoder(t_encoded, training=train)

        # Delta-gated input: amplify genes whose expression deviates from batch mean.
        # Exploits the sparsity of perturbation response — most genes don't change.
        batch_mean = jnp.mean(x_t, axis=0, keepdims=True)
        delta = x_t - batch_mean
        w = jax.nn.softmax(jnp.abs(delta) / (jnp.abs(self.delta_temp) + 1e-8), axis=-1)
        x_t = x_t * (1.0 + self.delta_scale * w)

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

        out = self.decoder(out, training=train)
        velocity = self.output_layer(out)

        # Perturbation-conditioned gene mask: learns which genes each perturbation affects,
        # enforcing sparse, perturbation-specific velocity predictions.
        if self.gene_mask_head is not None:
            gene_mask = nn.sigmoid(self.gene_mask_head(cond_embedding))
            velocity = velocity * gene_mask

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
    def x_encoder(self):
        return self._x_encoder

    @x_encoder.setter
    def x_encoder(self, encoder):
        self._x_encoder = encoder

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
