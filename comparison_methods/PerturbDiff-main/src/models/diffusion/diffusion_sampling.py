"""Sampling methods for GaussianDiffusion."""

from typing import Optional

import numpy as np
import torch as th

from src.common.utils import _extract_into_tensor
from src.models.diffusion.diffusion_core import ModelMeanType, ModelVarType


class GaussianDiffusionSamplingMixin:
    """Gaussiandiffusionsamplingmixin implementation used by the PerturbDiff pipeline."""
    # ============================================================
    # Section 1: Per-step statistics
    # ============================================================
    def p_mean_variance(
        self,
        model,
        x,
        t,
        self_condition=None,
        guidance_strength=0.0,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        prev_pred: Optional[th.Tensor] = None,
        prev_pred_control: Optional[th.Tensor] = None,
        sample_unperturbed: bool = False,
        sample_kwargs=None,
    ):
        """
        Compute reverse-process mean/variance and predicted x0 at step `t`.

        :param model: Denoiser model.
        :param x: Current noisy sample `x_t`.
        :param t: Diffusion timestep tensor.
        :param self_condition: Conditioning inputs for the model.
        :param guidance_strength: Classifier-free guidance scale.
        :param clip_denoised: Whether to clamp small values by cutoff.
        :param denoised_fn: Optional post-process function for predicted x0.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param prev_pred: Previous x0 prediction used for self-conditioning.
        :param prev_pred_control: Previous control prediction for compatibility.
        :param sample_unperturbed: Unsupported branch flag (kept for API).
        :param sample_kwargs: Extra sampling options (reserved).
        :return: Dict with `mean`, `variance`, `log_variance`, and predicted x0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        bsz = x.shape[0]
        assert t.shape == (bsz,)

        x_in = x
        if prev_pred is not None:
            x_in = th.cat([x, prev_pred], dim=-1)

        assert model.model_name == "Cross_DiT"
        control_input_start = self_condition["cont_emb"]
        control_in_t = control_input_start
        control_input_start = th.zeros_like(control_input_start)
        if model.model_cfg.p_drop_control == 1:
            control_in_t = th.zeros_like(control_in_t)
        if prev_pred is not None:
            control_in_t = th.cat([control_in_t, control_input_start], dim=-1)

        if sample_unperturbed:
            raise NotImplementedError(
                "sample_unperturbed requires learn_control=True, but learn_control is fixed false in src."
            )
        output = model(
            x_in,
            control_in_t,
            self._scale_timesteps(t).unsqueeze(1),
            self_condition=self_condition,
        )
        model_output = output["x"]
        cond_control_output = output.get("x_control")

        pred_xstart_control = None
        assert self.model_var_type in [ModelVarType.FIXED_SMALL, ModelVarType.FIXED_LARGE], (
            "Only fixed variance model types are supported in src."
        )
        model_variance, model_log_variance = {
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x_):
            """Execute `process_xstart` and return values used by downstream logic."""
            if denoised_fn is not None:
                x_ = denoised_fn(x_)
            if clip_denoised:
                return x_.masked_fill(x_ < model.model_cfg.cutoff, 0)
            return x_

        if self_condition is not None and guidance_strength != 0.0:
            if sample_unperturbed:
                raise NotImplementedError("sample_unperturbed is not implemented for classifier-free guidance")
            output = model(
                x_in,
                control_in_t,
                self._scale_timesteps(t).unsqueeze(1),
                self_condition={"gene_emb": self_condition["gene_emb"], "ds_name": self_condition["ds_name"]},
                **model_kwargs,
            )
            uncond_output = output["x"]
            uncond_control_output = output.get("x_control")

            uncond_eps = self._predict_eps_from_xstart(x, t, uncond_output)
            cond_eps = self._predict_eps_from_xstart(x, t, model_output)
            guided_eps = (1 + guidance_strength) * cond_eps - guidance_strength * uncond_eps
            model_output = self._predict_xstart_from_eps(x, t, guided_eps)

            if cond_control_output is not None and uncond_control_output is not None:
                pred_xstart_control = (1 + guidance_strength) * cond_control_output - guidance_strength * uncond_control_output
        elif cond_control_output is not None:
            pred_xstart_control = cond_control_output

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            raise NotImplementedError("Previous x model mean type not implemented")
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
            if pred_xstart_control is not None:
                pred_xstart_control = process_xstart(pred_xstart_control)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "pred_xstart_control": pred_xstart_control,
        }

    # ============================================================
    # Section 2: Single-step samplers
    # ============================================================
    def p_sample(
        self,
        model,
        x,
        t,
        self_condition=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        guidance_strength=0.0,
        nw=0.5,
        start_guide_steps=500,
        prev_pred: Optional[th.Tensor] = None,
        prev_pred_control: Optional[th.Tensor] = None,
        sample_unperturbed: bool = False,
        sample_kwargs=None,
    ):
        """
        Draw one stochastic DDPM reverse step.

        :param model: Denoiser model.
        :param x: Current noisy sample `x_t`.
        :param t: Diffusion timestep tensor.
        :param self_condition: Conditioning inputs for the model.
        :param clip_denoised: Whether to apply output clipping.
        :param denoised_fn: Optional post-process function for predicted x0.
        :param cond_fn: Optional gradient guidance function.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param guidance_strength: Classifier-free guidance scale.
        :param nw: Log-variance scaling factor for sampling noise.
        :param start_guide_steps: Guidance is applied only below this timestep.
        :param prev_pred: Previous x0 prediction used for self-conditioning.
        :param prev_pred_control: Previous control prediction for compatibility.
        :param sample_unperturbed: Unsupported branch flag (kept for API).
        :param sample_kwargs: Extra sampling options (reserved).
        :return: Dict with sampled `x_{t-1}` and predicted x0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            self_condition=self_condition,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            prev_pred=prev_pred,
            prev_pred_control=prev_pred_control,
            sample_unperturbed=sample_unperturbed,
            guidance_strength=guidance_strength,
            sample_kwargs=sample_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        if cond_fn is not None and t[0] < start_guide_steps:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(nw * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "pred_xstart_control": out["pred_xstart_control"]}

    def ddim_sample(
        self,
        model,
        x,
        t,
        self_condition=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        guidance_strength=0.0,
        prev_pred: Optional[th.Tensor] = None,
        prev_pred_control: Optional[th.Tensor] = None,
        sample_kwargs=None,
    ):
        """
        Draw one deterministic/stochastic DDIM reverse step.

        :param model: Denoiser model.
        :param x: Current noisy sample `x_t`.
        :param t: Diffusion timestep tensor.
        :param self_condition: Conditioning inputs for the model.
        :param clip_denoised: Whether to apply output clipping.
        :param denoised_fn: Optional post-process function for predicted x0.
        :param cond_fn: Optional score guidance function.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param eta: DDIM noise scale (0 for deterministic).
        :param guidance_strength: Classifier-free guidance scale.
        :param prev_pred: Previous x0 prediction used for self-conditioning.
        :param prev_pred_control: Previous control prediction for compatibility.
        :param sample_kwargs: Extra sampling options (reserved).
        :return: Dict with sampled `x_{t-1}` and predicted x0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            self_condition=self_condition,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            guidance_strength=guidance_strength,
            prev_pred=prev_pred,
            prev_pred_control=prev_pred_control,
            sample_kwargs=sample_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = th.randn_like(x)
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        self_condition=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        guidance_strength=0.0,
        sample_kwargs=None,
    ):
        """
        Advance one reverse-ODE DDIM step (deterministic path).

        :param model: Denoiser model.
        :param x: Current noisy sample `x_t`.
        :param t: Diffusion timestep tensor.
        :param self_condition: Conditioning inputs for the model.
        :param clip_denoised: Whether to apply output clipping.
        :param denoised_fn: Optional post-process function for predicted x0.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param eta: Must be 0.0 for reverse ODE mode.
        :param guidance_strength: Classifier-free guidance scale.
        :param sample_kwargs: Extra sampling options (reserved).
        :return: Dict with next-state sample and predicted x0.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            self_condition=self_condition,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            guidance_strength=guidance_strength,
            sample_kwargs=sample_kwargs,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    # ============================================================
    # Section 3: Multi-step loops
    # ============================================================
    def p_sample_loop(
        self,
        model,
        shape,
        self_condition=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        start_guide_steps=500,
        start_time=1000,
        nw=0.5,
        guidance_strength=0.0,
        sample_unperturbed: bool = False,
        sample_kwargs=None,
    ):
        """
        Run the full DDPM reverse loop and return the final sample.

        :param model: Denoiser model.
        :param shape: Output sample shape `(B, C, ...)`.
        :param self_condition: Conditioning inputs for the model.
        :param noise: Optional initial noise; random noise if None.
        :param clip_denoised: Whether to apply output clipping.
        :param denoised_fn: Optional post-process function for predicted x0.
        :param cond_fn: Optional gradient guidance function.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param device: Device to allocate sampling tensors on.
        :param progress: Whether to display a progress bar.
        :param start_guide_steps: Guidance is applied only below this timestep.
        :param start_time: Number of reverse steps to run.
        :param nw: Log-variance scaling factor for sampling noise.
        :param guidance_strength: Classifier-free guidance scale.
        :param sample_unperturbed: Unsupported branch flag (kept for API).
        :param sample_kwargs: Extra sampling options (reserved).
        :return: Tuple `(final_sample, traj)`.
        """
        final = None
        traj = []
        for _, sample in enumerate(
            self.p_sample_loop_progressive(
                model,
                shape,
                self_condition=self_condition,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                start_time=start_time,
                nw=nw,
                start_guide_steps=start_guide_steps,
                guidance_strength=guidance_strength,
                sample_unperturbed=sample_unperturbed,
                sample_kwargs=sample_kwargs,
            )
        ):
            final = sample
        return final["sample"], traj

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        self_condition=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        start_time=1000,
        nw=0.5,
        start_guide_steps=500,
        guidance_strength=0.0,
        sample_unperturbed: bool = False,
        sample_kwargs=None,
    ):
        """
        Yield DDPM reverse-step outputs from `start_time-1` down to 0.

        :param model: Denoiser model.
        :param shape: Output sample shape `(B, C, ...)`.
        :param self_condition: Conditioning inputs for the model.
        :param noise: Optional initial noise; random noise if None.
        :param clip_denoised: Whether to apply output clipping.
        :param denoised_fn: Optional post-process function for predicted x0.
        :param cond_fn: Optional gradient guidance function.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param device: Device to allocate sampling tensors on.
        :param progress: Whether to display a progress bar.
        :param start_time: Number of reverse steps to run.
        :param nw: Log-variance scaling factor for sampling noise.
        :param start_guide_steps: Guidance is applied only below this timestep.
        :param guidance_strength: Classifier-free guidance scale.
        :param sample_unperturbed: Unsupported branch flag (kept for API).
        :param sample_kwargs: Extra sampling options (reserved).
        :return: Generator yielding per-step sampling dictionaries.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else th.randn(*shape, device=device)
        prev_pred = th.zeros_like(img)
        prev_pred_control = th.zeros_like(img)
        indices = list(range(start_time))[::-1]

        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    self_condition=self_condition,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    guidance_strength=guidance_strength,
                    nw=nw,
                    start_guide_steps=start_guide_steps,
                    prev_pred=prev_pred,
                    prev_pred_control=prev_pred_control,
                    sample_unperturbed=sample_unperturbed,
                    sample_kwargs=sample_kwargs,
                )
                yield out
                img = out["sample"]
                prev_pred = out["pred_xstart"]
                prev_pred_control = out["pred_xstart_control"]

    def ddim_sample_loop(
        self,
        model,
        shape,
        self_condition=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        start_time=1000,
        eta=0.0,
        guidance_strength=0.0,
        sample_kwargs=None,
    ):
        """
        Run the full DDIM reverse loop and return the final sample.

        :param model: Denoiser model.
        :param shape: Output sample shape `(B, C, ...)`.
        :param self_condition: Conditioning inputs for the model.
        :param noise: Optional initial noise; random noise if None.
        :param clip_denoised: Whether to apply output clipping.
        :param denoised_fn: Optional post-process function for predicted x0.
        :param cond_fn: Optional score guidance function.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param device: Device to allocate sampling tensors on.
        :param progress: Whether to display a progress bar.
        :param start_time: Number of reverse steps to run.
        :param eta: DDIM noise scale (0 for deterministic).
        :param guidance_strength: Classifier-free guidance scale.
        :param sample_kwargs: Extra sampling options (reserved).
        :return: Tuple `(final_sample, traj)`.
        """
        final = None
        traj = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            self_condition=self_condition,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            start_time=start_time,
            eta=eta,
            guidance_strength=guidance_strength,
            sample_kwargs=sample_kwargs,
        ):
            final = sample
        return final["sample"], traj

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        self_condition=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        start_time=1000,
        eta=0.0,
        guidance_strength=0.0,
        sample_kwargs=None,
    ):
        """
        Yield DDIM reverse-step outputs from `start_time-1` down to 0.

        :param model: Denoiser model.
        :param shape: Output sample shape `(B, C, ...)`.
        :param self_condition: Conditioning inputs for the model.
        :param noise: Optional initial noise; random noise if None.
        :param clip_denoised: Whether to apply output clipping.
        :param denoised_fn: Optional post-process function for predicted x0.
        :param cond_fn: Optional score guidance function.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param device: Device to allocate sampling tensors on.
        :param progress: Whether to display a progress bar.
        :param start_time: Number of reverse steps to run.
        :param eta: DDIM noise scale (0 for deterministic).
        :param guidance_strength: Classifier-free guidance scale.
        :param sample_kwargs: Extra sampling options (reserved).
        :return: Generator yielding per-step sampling dictionaries.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else th.randn(*shape, device=device)

        prev_pred = th.zeros_like(img)
        prev_pred_control = None
        indices = list(range(start_time))[::-1]

        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    self_condition=self_condition,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    guidance_strength=guidance_strength,
                    prev_pred=prev_pred,
                    prev_pred_control=prev_pred_control,
                    sample_kwargs=sample_kwargs,
                )
                yield out
                img = out["sample"]
                prev_pred = out["pred_xstart"]


__all__ = ["GaussianDiffusionSamplingMixin"]
