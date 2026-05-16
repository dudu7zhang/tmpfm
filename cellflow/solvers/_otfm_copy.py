from collections.abc import Callable
from functools import partial
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow.solvers.utils import ema_update

__all__ = ["OTFlowMatching"]


class OTFlowMatching:
    """(OT) flow matching :cite:`lipman:22` extended to the conditional setting.

    With an extension to OT-CFM :cite:`tong:23,pooladian:23`, and its
    unbalanced version :cite:`eyring:24`.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network.
        probability_path
            Probability path between the source and the target distributions.
        match_fn
            Function to match samples from the source and the target
            distributions. It has a ``(src, tgt) -> matching`` signature,
            see e.g. :func:`cellflow.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure probability_path matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        kwargs
            Keyword arguments for :meth:`cellflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: dynamics.BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        # Nonlinear path and sparsity options
        use_nonlinear_path: bool = False,
        g_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        g_prime_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        gene_mask: ArrayLike | None = None,
        lambda_sparse: float = 1e-4,
        top_k_frac: float | None = None,
        tf_gated: bool = False,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn)
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()
        # nonlinear path settings
        self.use_nonlinear_path = use_nonlinear_path
        # default identity
        self.g_fn = g_fn if g_fn is not None else (lambda t: t)
        self.g_prime_fn = g_prime_fn if g_prime_fn is not None else (lambda t: jnp.ones_like(t))
        self.lambda_sparse = lambda_sparse
        self.top_k_frac = top_k_frac
        self.tf_gated = tf_gated
        self.gene_mask = jnp.array(gene_mask) if gene_mask is not None else None

    def _get_vf_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                # compute x_t and u_t: support nonlinear probability path via g_fn
                if self.use_nonlinear_path:
                    # t: (n,)
                    g_t = self.g_fn(t)
                    g_t_prime = self.g_prime_fn(t)
                    # broadcast to gene dimension
                    delta = target - source
                    x_t = source + g_t[:, None] * delta
                    u_t = g_t_prime[:, None] * delta
                else:
                    x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                    u_t = self.probability_path.compute_ut(t, x_t, source, target)

                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )

                # prepare GRN mask (if provided)
                if self.gene_mask is not None:
                    # ensure shape (1, G) for broadcasting
                    gene_mask = jnp.reshape(self.gene_mask, (1, -1))
                else:
                    gene_mask = None

                # optional top-k hard sparsification (applied before GRN masking)
                v_theta = v_t
                if (self.top_k_frac is not None) and (self.top_k_frac > 0.0):
                    G = int(v_theta.shape[-1])
                    k = max(1, int(self.top_k_frac * G))
                    abs_v = jnp.abs(v_theta)
                    # compute per-sample threshold
                    thresh = jnp.sort(abs_v, axis=-1)[:, -k][:, None]
                    topk_mask = abs_v >= thresh
                    v_theta_topk = v_theta * topk_mask
                else:
                    v_theta_topk = v_theta

                # apply GRN mask for flow-matching loss
                if gene_mask is not None:
                    v_grn = v_theta_topk * gene_mask
                else:
                    v_grn = v_theta_topk

                # flow matching loss compares GRN-constrained velocity to u_t
                flow_matching_loss = jnp.mean((v_grn - u_t) ** 2)

                # Hoyer sparsity regularizer on the raw predicted velocity v_theta
                eps = 1e-8
                l1 = jnp.sum(jnp.abs(v_theta), axis=-1)
                l2 = jnp.sqrt(jnp.sum(v_theta ** 2, axis=-1) + eps)
                hoyer = l1 / (l2 + eps)
                sparse_loss = self.lambda_sparse * jnp.mean(hoyer)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0
                # return flow_matching_loss + encoder_loss
                return flow_matching_loss + encoder_loss + sparse_loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(vf_state.params, time, source, target, conditions, encoder_noise, rng)
            return vf_state.apply_gradients(grads=grads), loss

        return vf_step_fn

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ) -> float:
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            optionally ``condition``.

        Returns
        -------
        Loss value.
        """
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))
        # TODO: test whether it's better to sample the same noise for all samples or different ones

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
            src,
            tgt,
            condition,
            encoder_noise,
        )

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )
        return loss

    def get_condition_embedding(self, condition: dict[str, ArrayLike], return_as_numpy=True) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode
        return_as_numpy
            Whether to return the embeddings as numpy arrays.

        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state_inference.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    def _predict_jit(
        self, x: ArrayLike, condition: dict[str, ArrayLike], rng: jax.Array | None = None, **kwargs: Any
    ) -> ArrayLike:
        """See :meth:`OTFlowMatching.predict`."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params = self.vf_state_inference.params
            condition, encoder_noise = args
            v = self.vf_state_inference.apply_fn({"params": params}, t, x, condition, encoder_noise, train=False)[0]
            if self.gene_mask is not None:
                gm = jnp.reshape(self.gene_mask, (1, -1))
                v = v * gm
            return v

        def solve_ode(x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=(condition, encoder_noise),
                **kwargs,
            )
            return result.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None, None]))(x, condition, encoder_noise)
        return x_pred

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        batched: bool = False,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~cellflow.networks.ConditionalVelocityField`.

        Parameters
        ----------
        x
            A dictionary with keys indicating the name of the condition and values containing
            the input data as arrays. If ``batched=False`` provide an array of shape [batch_size, ...].
        condition
            A dictionary with keys indicating the name of the condition and values containing
            the condition of input data as arrays. If ``batched=False`` provide an array of shape
            [batch_size, ...].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        batched
            Whether to use batched prediction. This is only supported if the input has
            the same number of cells for each condition. For example, this works when using
            :class:`~cellflow.data.ValidationSampler` to sample the validation data.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if batched and not x:
            return {}

        if batched:
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            _predict_jit = jax.jit(lambda x, condition: self._predict_jit(x, condition, rng, **kwargs))
            batched_predict = jax.vmap(_predict_jit, in_axes=(0, dict.fromkeys(condition_keys, 0)))
            # assert that the number of cells is the same for each condition
            n_cells = x[keys[0]].shape[0]
            for k in keys:
                assert x[k].shape[0] == n_cells, "The number of cells must be the same for each condition"
            src_inputs = jnp.stack([x[k] for k in keys], axis=0)
            batched_conditions = {}
            for cond_key in condition_keys:
                batched_conditions[cond_key] = jnp.stack([condition[k][cond_key] for k in keys])

            pred_targets = batched_predict(src_inputs, batched_conditions)
            return {k: pred_targets[i] for i, k in enumerate(keys)}
        elif isinstance(x, dict):
            return jax.tree.map(
                partial(self._predict_jit, rng=rng, **kwargs),
                x,
                condition,  # type: ignore[attr-defined]
            )
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
