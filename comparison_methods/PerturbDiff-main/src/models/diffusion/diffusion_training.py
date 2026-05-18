"""Training/loss methods for GaussianDiffusion."""

import torch as th
from geomloss import SamplesLoss

from src.models.diffusion.diffusion_core import LossType, ModelMeanType, ModelVarType


def mean_flat(tensor):
    """
    Compute the mean across all non-batch dimensions.

    :param tensor: Input tensor with batch dimension at index 0.
    :return: Tensor reduced to per-sample means.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# ============================================================
# Internal helpers
# ============================================================
def _ensure_supported_loss_type(loss_type):
    """Execute `_ensure_supported_loss_type` and return values used by downstream logic."""
    if loss_type not in (LossType.MSE, LossType.RESCALED_MSE):
        raise NotImplementedError(f"Unsupported loss type in src: {loss_type}")


def _ensure_supported_var_type(model_var_type):
    """Execute `_ensure_supported_var_type` and return values used by downstream logic."""
    assert model_var_type in (ModelVarType.FIXED_SMALL, ModelVarType.FIXED_LARGE), (
        "Only fixed variance model types are supported in src."
    )


def _build_training_target(model_mean_type, x_start, noise):
    """Build the main-branch supervision target."""
    assert model_mean_type != ModelMeanType.PREVIOUS_X, "Previous x model mean type not implemented"
    return {
        ModelMeanType.START_X: x_start,
        ModelMeanType.EPSILON: noise,
    }[model_mean_type]

class GaussianDiffusionTrainingMixin:
    """Gaussiandiffusiontrainingmixin implementation used by the PerturbDiff pipeline."""
    def get_model_output(
        self,
        model,
        x_t,
        control_input_t,
        t,
        self_condition=None,
        model_kwargs={},
    ):
        """
        Run Cross_DiT once at timestep `t` and return branch outputs.

        :param model: Denoiser model.
        :param x_t: Main-branch noisy input tensor.
        :param control_input_t: Control-branch input tensor.
        :param t: Diffusion timestep tensor.
        :param self_condition: Conditioning inputs for the model.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :return: Model output dictionary (main and control branches).
        """
        assert model.model_name == "Cross_DiT"
        return model(
            x_t,
            control_input_t,
            self._scale_timesteps(t).unsqueeze(1),
            self_condition=self_condition,
            **model_kwargs,
        )

    def diffusion_loss(
        self,
        model,
        x_start,
        x_t,
        control_input_t,
        t,
        self_condition=None,
        model_kwargs={},
        noise=None,
        x_0=None,
        control_0=None,
        MMD_loss_fn=None,
        return_model_output=False,
    ):
        """
        Compute per-branch MSE/MMD loss terms for one diffusion step.

        :param model: Denoiser model.
        :param x_start: Clean main-branch target tensor.
        :param x_t: Noisy main-branch tensor at timestep `t`.
        :param control_input_t: Control-branch tensor at timestep `t`.
        :param t: Diffusion timestep tensor.
        :param self_condition: Conditioning inputs for the model.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param noise: Main-branch Gaussian noise.
        :param x_0: Self-conditioning estimate for the main branch.
        :param control_0: Self-conditioning estimate for the control branch.
        :param MMD_loss_fn: MMD loss function.
        :param return_model_output: Whether to also return raw model outputs.
        :return: Loss term dictionary, and optional model output dict.
        """
        terms = {}
        if MMD_loss_fn is None:
            MMD_loss_fn = SamplesLoss(loss="energy", blur=0.05)

        _ensure_supported_loss_type(self.loss_type)
        _ensure_supported_var_type(self.model_var_type)

        model_output = self.get_model_output(
            model=model,
            x_t=th.concat([x_t, x_0], dim=-1),
            control_input_t=th.concat([control_input_t, control_0], dim=-1),
            t=t,
            self_condition=self_condition,
            model_kwargs=model_kwargs,
        )
        model_output1 = model_output["x"]
        target1 = _build_training_target(
            self.model_mean_type,
            x_start,
            noise,
        )

        terms["mse1"] = mean_flat((target1 - model_output1) ** 2)
        terms["mmd1"] = MMD_loss_fn(target1.type_as(model_output1), model_output1).nanmean()
        terms["mmd1_list"] = MMD_loss_fn(target1.type_as(model_output1).detach(), model_output1.detach())

        if "vb1" in terms:
            terms["loss1"] = terms["mse1"] + terms["vb1"]
        else:
            terms["loss1"] = terms["mse1"]

        if return_model_output:
            return terms, model_output
        return terms

    def training_losses(
        self,
        model,
        x_start,
        t,
        self_condition=None,
        model_kwargs=None,
        noise=None,
        p_drop_cond=0.0,
        MMD_loss_fn=None,
        return_model_output=False,
    ):
        """
        Prepare noisy inputs and compute training losses for one batch.

        :param model: Denoiser model.
        :param x_start: Clean main-branch input tensor.
        :param t: Diffusion timestep tensor.
        :param self_condition: Conditioning inputs for the model.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param noise: Optional pre-generated main-branch noise.
        :param p_drop_cond: Probability of dropping conditional batch embedding.
        :param MMD_loss_fn: Optional MMD loss function override.
        :param return_model_output: Kept for compatibility (unused in return).
        :return: Loss term dictionary for optimization/logging.
        """
        if model_kwargs is None:
            model_kwargs = {}
        assert model.model_name == "Cross_DiT"
        control_input_start = self_condition["cont_emb"]

        if noise is None:
            noise = th.randn_like(x_start, dtype=th.float64)
        x_t = self.q_sample(
            x_start,
            t,
            noise=noise,
            x_control=None,
        )

        control_input_t = control_input_start

        if model.model_cfg.p_drop_control > 0.0 and th.rand(1) < model.model_cfg.p_drop_control:
            control_input_t = th.zeros_like(control_input_t)

        self_condition["cont_emb_copy"] = self_condition["cont_emb"]
        if p_drop_cond > 0.0 and th.rand(1) < p_drop_cond:
            self_condition["batch_emb"] = None

        x_0 = th.zeros_like(x_t)
        control_0 = th.zeros_like(control_input_t)
        use_selfcond_now = bool((th.rand(1) > 0.5).item())

        if use_selfcond_now:
            with th.no_grad():
                out = self.get_model_output(
                    model=model,
                    x_t=th.concat([x_t, x_0], dim=-1),
                    control_input_t=th.concat([control_input_t, control_0], dim=-1),
                    t=t,
                    self_condition=self_condition,
                    model_kwargs=model_kwargs,
                )
            x_0 = out["x"]
            control_0 = th.zeros_like(out["x_control"])

        terms = self.diffusion_loss(
            model=model,
            x_start=x_start,
            x_t=x_t,
            control_input_t=control_input_t,
            t=t,
            self_condition=self_condition,
            model_kwargs=model_kwargs,
            noise=noise,
            x_0=x_0,
            control_0=control_0,
            MMD_loss_fn=MMD_loss_fn,
        )

        if model.model_cfg.no_mse_loss:
            terms["mse1"] = th.zeros_like(terms["mse1"])
        return terms


__all__ = ["GaussianDiffusionTrainingMixin"]
