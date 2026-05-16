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

from cellflow import utils
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow.solvers.utils import ema_update

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
    sinkhorn_val, _ = sinkhorn_divergence(
        pointcloud.PointCloud,
        x=pred,
        y=target,
        cost_fn=costs.SqEuclidean(),
        epsilon=epsilon,
        scale_cost=1.0,
    )
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
        use_nonlinear_path: bool = False,
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
        condition_gene_masks = kwargs.pop("condition_gene_masks", None)
        # Optional JAX combined distribution loss (Sinkhorn + Energy) on a terminal-state estimate.
        # Keep the solver default off; training scripts opt in through solver_kwargs.
        self.condition_combined_loss_weight = float(kwargs.pop("condition_combined_loss_weight", 0.0))
        self.sinkhorn_weight = float(kwargs.pop("condition_combined_sinkhorn_weight", 0.001))
        self.energy_weight = float(kwargs.pop("condition_combined_energy_weight", 1.0))
        self.condition_combined_epsilon = float(kwargs.pop("condition_combined_epsilon", 1e-2))
        self.condition_influence_weight = float(kwargs.pop("condition_influence_weight", 0.1))

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
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
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
                sq_err = (v_t - u_t) ** 2
                base_mean = jnp.mean(sq_err)
                flow_matching_loss = base_mean

                # Optional: JAX combined Sinkhorn + Energy regularizer.
                if self.condition_combined_loss_weight > 0:
                    t_col = jnp.reshape(t, (-1, 1)).astype(v_t.dtype)
                    
                    # Stop gradients for t < 0.5 to prevent noisy extrapolations at early stages
                    # from destroying the learned flow matching trajectory.
                    # We scale the gradient smoothly using t^2, so it strongly aligns at t->1
                    weight_t = jnp.power(t_col, 2)
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
