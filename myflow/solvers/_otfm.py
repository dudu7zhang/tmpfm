from collections.abc import Callable
from functools import partial
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from flax.training import train_state
from ott.geometry import costs, pointcloud
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from myflow import utils
from myflow._types import ArrayLike
from myflow.networks._velocity_field import ConditionalVelocityField
from myflow.solvers.utils import ema_update

__all__ = ["OTFlowMatching"]


def _energy_distance_jax(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute energy distance in JAX for two empirical distributions."""

    def _pairwise_sqeuclidean(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)

    sigma_x = jnp.mean(_pairwise_sqeuclidean(x, x))
    sigma_y = jnp.mean(_pairwise_sqeuclidean(y, y))
    delta = jnp.mean(_pairwise_sqeuclidean(x, y))
    return 2.0 * delta - sigma_x - sigma_y


def _combined_distribution_loss_jax(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    sinkhorn_weight: float = 0.001,
    energy_weight: float = 1.0,
    epsilon: float = 1e-2,
) -> jnp.ndarray:
    """JAX equivalent of CombinedLoss(Sinkhorn + Energy)."""
    if sinkhorn_weight > 0:
        sinkhorn_val, _ = sinkhorn_divergence(
            pointcloud.PointCloud,
            x=pred,
            y=target,
            cost_fn=costs.SqEuclidean(),
            epsilon=epsilon,
            scale_cost=1.0,
        )
    else:
        sinkhorn_val = jnp.asarray(0.0, dtype=pred.dtype)
    energy_val = _energy_distance_jax(pred, target)
    return sinkhorn_weight * sinkhorn_val + energy_weight * energy_val



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
            see e.g. :func:`myflow.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure probability_path matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        kwargs
            Keyword arguments for :meth:`myflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: dynamics.BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        use_nonlinear_path: bool = False,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self._cached_predict_fn = None
        self._cached_predict_kwargs = None
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn)
        self.match_every_n = int(kwargs.pop("match_every_n", 1))
        self._step_counter = 0
        self.ema = kwargs.pop("ema", 1.0)
        condition_gene_masks = kwargs.pop("condition_gene_masks", None)
        # Optional JAX combined distribution loss (Sinkhorn + Energy) on a terminal-state estimate.
        # Keep the solver default off; training scripts opt in through solver_kwargs.
        self.condition_combined_loss_weight = float(kwargs.pop("condition_combined_loss_weight", 0.0))
        self.sinkhorn_weight = float(kwargs.pop("condition_combined_sinkhorn_weight", 0.001))
        self.energy_weight = float(kwargs.pop("condition_combined_energy_weight", 1.0))
        self.condition_combined_epsilon = float(kwargs.pop("condition_combined_epsilon", 1e-2))
        self.condition_influence_weight = float(kwargs.pop("condition_influence_weight", 0.1))
        self.endpoint_mse_weight = float(kwargs.pop("endpoint_mse_weight", 0.0))
        self.cosine_loss_weight = float(kwargs.pop("cosine_loss_weight", 0.0))
        self.condition_mean_delta_weight = float(kwargs.pop("condition_mean_delta_weight", 0.0))
        self.high_delta_endpoint_weight = float(kwargs.pop("high_delta_endpoint_weight", 0.0))
        self.high_delta_max_weight = float(kwargs.pop("high_delta_max_weight", 4.0))
        self.terminal_loss_time_power = float(kwargs.pop("terminal_loss_time_power", 2.0))
        self.high_delta_eps = float(kwargs.pop("high_delta_eps", 1e-6))

        self.matrix = kwargs.pop("matrix", None)
        
        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()

        self.matrix = jnp.array(self.matrix) if self.matrix is not None else None
        self.condition_gene_masks = jnp.array(condition_gene_masks) if condition_gene_masks is not None else None


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
            condition_idx: jnp.ndarray | None,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
                condition_idx: jnp.ndarray | None,
            ) -> jnp.ndarray:
                rng_flow, rng_encoder, rng_dropout, rng_graph_dropout = jax.random.split(rng, 4)
                x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                u_t = self.probability_path.compute_ut(t, x_t, source, target)
                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder, "graph_dropout": rng_graph_dropout},
                )
                sq_err = (v_t - u_t) ** 2
                base_mean = jnp.mean(sq_err)
                flow_matching_loss = base_mean

                t_col = jnp.reshape(t, (-1, 1)).astype(v_t.dtype)
                terminal_gate = jnp.power(t_col, self.terminal_loss_time_power)

                def _condition_mean(values: jnp.ndarray) -> jnp.ndarray:
                    if condition_idx is None:
                        return jnp.mean(values, axis=0, keepdims=True)
                    same_cond = (condition_idx[:, None] == condition_idx[None, :]).astype(values.dtype)
                    denom = jnp.sum(same_cond, axis=1, keepdims=True) + self.high_delta_eps
                    return same_cond @ values / denom

                def _high_delta_weights(true_delta_abs: jnp.ndarray) -> jnp.ndarray:
                    axis = -1 if true_delta_abs.ndim == 2 else 0
                    keepdims = true_delta_abs.ndim == 2
                    scale = jnp.mean(true_delta_abs, axis=axis, keepdims=keepdims) + self.high_delta_eps
                    weights = true_delta_abs / scale
                    weights = 1.0 + self.high_delta_endpoint_weight * weights
                    return jnp.minimum(weights, self.high_delta_max_weight)

                # Optional: JAX combined Sinkhorn + Energy regularizer.
                if self.condition_combined_loss_weight > 0:
                    # Stop gradients for t < 0.5 to prevent noisy extrapolations at early stages
                    # from destroying the learned flow matching trajectory.
                    # We scale the gradient smoothly using t^2, so it strongly aligns at t->1
                    weight_t = terminal_gate
                    v_t_sinkhorn = jax.lax.stop_gradient(v_t) + (v_t - jax.lax.stop_gradient(v_t)) * weight_t
                    x1_hat = x_t + (1.0 - t_col) * v_t_sinkhorn
                    
                    combined_loss = _combined_distribution_loss_jax(
                        pred=x1_hat,
                        target=target,
                        sinkhorn_weight=self.sinkhorn_weight,
                        energy_weight=self.energy_weight,
                        epsilon=self.condition_combined_epsilon,
                    )
                    flow_matching_loss = flow_matching_loss + self.condition_combined_loss_weight * combined_loss

                # Direct endpoint MSE supervision (no stop_gradient).
                if self.endpoint_mse_weight > 0:
                    x1_pred = x_t + (1.0 - t_col) * v_t
                    endpoint_sq_err = (x1_pred - target) ** 2
                    if self.high_delta_endpoint_weight > 0:
                        true_delta_abs = jnp.abs(_condition_mean(target - source))
                        gene_weights = _high_delta_weights(true_delta_abs)
                        endpoint_sq_err = endpoint_sq_err * gene_weights
                    endpoint_sq_err = endpoint_sq_err * terminal_gate
                    endpoint_loss = jnp.mean(endpoint_sq_err)
                    flow_matching_loss = flow_matching_loss + self.endpoint_mse_weight * endpoint_loss

                # Condition-level mean supervision in delta space.
                if self.condition_mean_delta_weight > 0:
                    x1_pred_cm = x_t + (1.0 - t_col) * v_t
                    mean_delta_pred = _condition_mean(x1_pred_cm - source)
                    mean_delta_true = _condition_mean(target - source)
                    mean_delta_sq_err = (mean_delta_pred - mean_delta_true) ** 2
                    if self.high_delta_endpoint_weight > 0:
                        true_delta_abs = jnp.abs(mean_delta_true)
                        gene_weights = _high_delta_weights(true_delta_abs)
                        mean_delta_sq_err = mean_delta_sq_err * gene_weights
                    if mean_delta_sq_err.ndim == 2:
                        mean_delta_sq_err = mean_delta_sq_err * terminal_gate
                    else:
                        mean_delta_sq_err = mean_delta_sq_err * jnp.mean(terminal_gate)
                    condition_mean_delta_loss = jnp.mean(mean_delta_sq_err)
                    flow_matching_loss = flow_matching_loss + self.condition_mean_delta_weight * condition_mean_delta_loss

                # Cosine similarity loss on delta (directional accuracy).
                if self.cosine_loss_weight > 0:
                    x1_pred_cs = x_t + (1.0 - t_col) * v_t
                    delta_pred = x1_pred_cs - source
                    delta_true = target - source
                    # Normalize to unit vectors
                    pred_norm = jnp.linalg.norm(delta_pred, axis=-1, keepdims=True) + 1e-8
                    true_norm = jnp.linalg.norm(delta_true, axis=-1, keepdims=True) + 1e-8
                    cos_sim = jnp.sum((delta_pred / pred_norm) * (delta_true / true_norm), axis=-1)
                    cosine_loss = jnp.mean(1.0 - cos_sim)
                    flow_matching_loss = flow_matching_loss + self.cosine_loss_weight * cosine_loss


                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0
                return flow_matching_loss + encoder_loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(
                vf_state.params,
                time,
                source,
                target,
                conditions,
                encoder_noise,
                rng,
                condition_idx,
            )
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
        condition_idx = batch.get("condition_idx")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))
        # TODO: test whether it's better to sample the same noise for all samples or different ones

        if self.match_fn is not None and self.match_every_n > 0 and (self._step_counter % self.match_every_n == 0):
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]
        self._step_counter += 1

        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
            src,
            tgt,
            condition,
            encoder_noise,
            condition_idx,
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

        # Cache the compiled function to avoid recompilation on every call
        kwargs_key = tuple(sorted(kwargs.items()))
        if self._cached_predict_fn is None or self._cached_predict_kwargs != kwargs_key:
            def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
                params = self.vf_state_inference.params
                condition, encoder_noise = args
                return self.vf_state_inference.apply_fn({"params": params}, t, x, condition, encoder_noise, train=False)[0]

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

            self._cached_predict_fn = jax.jit(jax.vmap(solve_ode, in_axes=[0, None, None]))
            self._cached_predict_kwargs = kwargs_key

        x_pred = self._cached_predict_fn(x, condition, encoder_noise)
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
        the :class:`~myflow.networks.ConditionalVelocityField`.

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
            :class:`~myflow.data.ValidationSampler` to sample the validation data.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        predict_batch_size = kwargs.pop("predict_batch_size", None)

        def _predict_array(
            x_arr: ArrayLike,
            cond_arr: dict[str, ArrayLike],
        ) -> ArrayLike:
            if predict_batch_size is None:
                return self._predict_jit(x_arr, cond_arr, rng, **kwargs)

            batch_size = int(predict_batch_size)
            if batch_size <= 0:
                raise ValueError("`predict_batch_size` must be a positive integer.")

            n = x_arr.shape[0]
            if n <= batch_size:
                return self._predict_jit(x_arr, cond_arr, rng, **kwargs)

            chunks = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_chunk = x_arr[start:end]

                # Some condition tensors are cell-aligned (leading dim == n), while
                # others are shared per-condition metadata (e.g. leading dim == 1).
                # Slice only cell-aligned tensors to avoid creating empty arrays.
                cond_chunk = {}
                for k, v in cond_arr.items():
                    if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n:
                        cond_chunk[k] = v[start:end]
                    else:
                        cond_chunk[k] = v

                chunks.append(self._predict_jit(x_chunk, cond_chunk, rng, **kwargs))
            return jnp.concatenate(chunks, axis=0)

        if batched and not x:
            return {}

        if batched:
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            # Reuse the cached predict function instead of creating a new jit each time
            batched_predict = jax.vmap(lambda x, condition: self._predict_jit(x, condition, rng, **kwargs), in_axes=(0, dict.fromkeys(condition_keys, 0)))
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
                _predict_array,
                x,
                condition,  # type: ignore[attr-defined]
            )
        else:
            x_pred = _predict_array(x, condition)
            return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
