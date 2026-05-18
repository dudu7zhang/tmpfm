"""Cross_DiT main implementation."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.common.utils import get_short_dsname
from src.models.cross_dit.cross_dit_component import (
    ClassEmbedding,
    FinalLayer,
    GeneEmbedding,
    TimestepEmbedder,
)
from src.models.cross_dit.cross_dit_blocks import Cross_DiTBlock
from src.models.cross_dit.cross_dit_init import initialize_weights, reinitialize_adaln, replace_2kgene_layer


# ============================================================
# Cross_DiT: main module assembly
# - builds embedders, transformer blocks, and output heads
# - contains conditioning and forward logic in one file
# ============================================================
class Cross_DiT(nn.Module):
    """Cross Dit class."""
    def __init__(
        self,
        model_cfg,
        mlp_ratio=4.0,
        learn_sigma=False,
        **kwargs,
    ):
        """
        Initialize Cross_DiT modules and fixed runtime flags.

        :param model_cfg: Model configuration.
        :param mlp_ratio: MLP width multiplier for transformer blocks.
        :param learn_sigma: Compatibility flag; must remain False.
        :param **kwargs: Unused compatibility kwargs.
        :return: None.
        """
        super().__init__()
        assert not learn_sigma, "learn_sigma=True is no longer supported for Cross_DiT."

        # ---------------------------------------------------------------------
        # Core config/state
        # ---------------------------------------------------------------------
        self.model_cfg = model_cfg
        self.use_orig_gene_count_as_emb = True
        self.concat_control_as_token = False
        self.model_name = "Cross_DiT"
        self.learn_sigma = False
        self.num_heads = self.model_cfg.dit_num_heads
        self.separate_embedder = model_cfg.separate_embedder
        self.separate_final_layer = False
        self._setup_dims(model_cfg)
        self._validate_config(model_cfg)

        # ---------------------------------------------------------------------
        # Module construction
        # ---------------------------------------------------------------------
        self._build_x_embedders(model_cfg)
        self._build_conditioning_modules(model_cfg)
        self._build_backbone_and_heads(model_cfg, mlp_ratio)

        self.output_fn = F.relu

        # ---------------------------------------------------------------------
        # Parameter initialization
        # ---------------------------------------------------------------------
        self.initialize_weights()

    def _setup_dims(self, model_cfg):
        """Set key model dimensions used across embedders/heads."""
        self.hidden_size = model_cfg.hidden_num[1]
        # hidden_num[0] is the base gene dimension before concat with self-conditioning.
        self.input_size = model_cfg.hidden_num[0]
        self.output_size = self.input_size
        # x-embedder input concatenates current input and self-conditioning.
        self.input_size *= 2

    def _validate_config(self, model_cfg):
        """Validate assumptions enforced by this refactor."""
        assert isinstance(model_cfg.dit_depth, int), "model_cfg.dit_depth must be an integer."

    def _build_x_embedders(self, model_cfg):
        """Build x/control input embedders (shared or dataset-specific)."""
        if model_cfg.separate_embedder:
            x_embedder = {}
            x_control_embedder = {}
            for ds_name in model_cfg.dataset_dict:
                if model_cfg.separate_embedder == "by_name":
                    ds_name = get_short_dsname(ds_name)
                x_embedder[ds_name] = nn.Linear(self.input_size, self.hidden_size, bias=True)
                x_control_embedder[ds_name] = nn.Linear(self.input_size, self.hidden_size, bias=True)
            self.x_embedder = nn.ModuleDict(x_embedder)
            self.x_control_embedder = nn.ModuleDict(x_control_embedder)
            return

        self.x_embedder = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.x_control_embedder = nn.Linear(self.input_size, self.hidden_size, bias=True)

    def _build_conditioning_modules(self, model_cfg):
        """Build timestep/gene/class conditioning modules."""
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.pre_gene_embedding = model_cfg.use_gene_embedding
        gene_embedding_dim = 1 if self.pre_gene_embedding else self.hidden_size
        self.gene_embedding = GeneEmbedding(
            gene_embedding_dim,
            gene_embedding_type=model_cfg.gene_embedding_type,
        )
        self.class_embedding = ClassEmbedding(model_cfg.hidden_num[0], self.hidden_size)

    def _build_backbone_and_heads(self, model_cfg, mlp_ratio):
        """Build transformer backbone blocks and output heads."""
        self.blocks = nn.ModuleList(
            [
                Cross_DiTBlock(
                    self.hidden_size,
                    self.model_cfg.dit_num_heads,
                    mlp_ratio=mlp_ratio,
                    layerf_id=layer_id,
                    Block_type=model_cfg.Block_type,
                )
                for layer_id in range(model_cfg.dit_depth)
            ]
        )
        self.final_layer = FinalLayer(self.hidden_size, output_size=self.output_size)
        self.control_final_layer = FinalLayer(self.hidden_size, output_size=self.output_size)

    def initialize_weights(self):
        # Delegate to shared initialization helper.
        """Execute `initialize_weights` and return values used by downstream logic."""
        initialize_weights(self)
    
    def reinitialize_adaln(self):
        # Re-zero adaLN modulation layers for finetuning/reset scenarios.
        """Execute `reinitialize_adaln` and return values used by downstream logic."""
        reinitialize_adaln(self)

    def replace_2kgene_layer(self, new_input_size):
        # Replace in/out projection layers when gene input dimension changes.
        """Execute `replace_2kgene_layer` and return values used by downstream logic."""
        replace_2kgene_layer(self, new_input_size)

    @staticmethod
    def _standardize_ds_name(separate_mode, ds_name):
        """Execute `_standardize_ds_name` and return values used by downstream logic."""
        if separate_mode == "by_name":
            return get_short_dsname(ds_name)
        return ds_name

    @staticmethod
    def _as_np_name_array(ds_name_list):
        """Execute `_as_np_name_array` and return values used by downstream logic."""
        return np.array([name for name in ds_name_list])

    @staticmethod
    def _allocate_output_tensor(x_input, first_out):
        """Execute `_allocate_output_tensor` and return values used by downstream logic."""
        return torch.zeros_like(
            first_out.new_empty(x_input.shape[0], first_out.shape[1], first_out.shape[2])
        )

    def _apply_embedder_layer(self, x_embed_layer, ds_name_list, x_input):
        """
        Apply a shared or dataset-specific embedder to `x_input`.

        :param x_embed_layer: Shared layer or per-dataset layer dict.
        :param ds_name_list: Dataset name per sample in the batch.
        :param x_input: Input tensor to embed.
        :return: Embedded tensor in the same batch order.
        """
        if not self.separate_embedder:
            return x_embed_layer(x_input)

        ds_name_array = self._as_np_name_array(ds_name_list)
        outs = []
        for raw_ds_name in self.model_cfg.dataset_dict:
            index = ds_name_array == raw_ds_name
            ds_key = self._standardize_ds_name(self.model_cfg.separate_embedder, raw_ds_name)
            if index.any():
                outs.append((index, x_embed_layer[ds_key](x_input[index])))

        ret = self._allocate_output_tensor(x_input, outs[0][1])
        for index, out in outs:
            ret[index] = out
        return ret

    def _get_embedded_x(
        self,
        x_input,
        x_control_input,
        gene_emb,
        x_embed_layer,
        x_control_embed_layer,
        ds_name_list,
    ):
        """
        Build embedded main/control inputs with optional gene pre-embedding.

        :param x_input: Main branch input tensor.
        :param x_control_input: Control branch input tensor.
        :param gene_emb: Gene embedding tensor.
        :param x_embed_layer: Embedder for the main branch.
        :param x_control_embed_layer: Embedder for the control branch.
        :param ds_name_list: Dataset name per sample in the batch.
        :return: Tuple `(x, x_control)` in hidden feature space.
        """
        if self.pre_gene_embedding:
            gene_emb = gene_emb.squeeze(2).unsqueeze(1)  # [bsz, L, 1] -> [bsz, 1, L]
            gene_emb = gene_emb.repeat(1, 1, 2)  # [bsz, 1, L*N]

            # Keep this step for strict behavior parity.
            x = gene_emb + x_input
            x_control = gene_emb + x_control_input

            x = self._apply_embedder_layer(x_embed_layer, ds_name_list, x)  # [bsz, S, dim]
            x_control = self._apply_embedder_layer(x_control_embed_layer, ds_name_list, x_control)  # [bsz, S, dim]
        else:
            x = self._apply_embedder_layer(x_embed_layer, ds_name_list, x_input) + gene_emb  # [bsz, S, dim]
            x_control = (
                self._apply_embedder_layer(x_control_embed_layer, ds_name_list, x_control_input) + gene_emb
            )  # [bsz, S, dim]

        return x, x_control

    def forward(self, x_input, x_control_input, t, self_condition=None):
        """
        Run one forward pass for main/control branches.

        :param x_input: Main branch input tensor.
        :param x_control_input: Control branch input tensor.
        :param t: Diffusion timestep tensor.
        :param self_condition: Conditioning dict with covariates/gene embeddings.
        :return: Dict with denoised outputs and intermediate embeddings.
        """
        cov_emb = self_condition.get("batch_emb", None)

        assert "ds_name" in self_condition
        if self.separate_embedder:
            tmp_ds_name = self_condition["ds_name"][0]
            if self.separate_embedder == "by_name":
                tmp_ds_name = get_short_dsname(tmp_ds_name)

        x_input = x_input.type_as(
            self.x_embedder.weight if not self.separate_embedder else self.x_embedder[tmp_ds_name].weight
        )
        x_control_input = x_control_input.type_as(
            self.x_control_embedder.weight
            if not self.separate_embedder
            else self.x_control_embedder[tmp_ds_name].weight
        )

        if self_condition is not None:
            gene_emb = self_condition.get("gene_emb", None)
            gene_emb = self.gene_embedding(gene_emb, batch_size=x_input.shape[0], x_len=x_input.shape[1])
            class_emb = cov_emb
            class_emb = self.class_embedding(class_emb, batch_size=x_input.shape[0])
        else:
            class_emb = self.class_embedding(None, batch_size=x_input.shape[0])
            gene_emb = self.gene_embedding(None, batch_size=x_input.shape[0], x_len=x_input.shape[1])

        t = self.t_embedder(t)
        c = F.silu(t + class_emb) if self.model_cfg.use_class_silu else t + class_emb

        x, x_control = self._get_embedded_x(
            x_input,
            x_control_input,
            gene_emb,
            self.x_embedder,
            self.x_control_embedder,
            self_condition["ds_name"],
        )
        for layer_idx, block in enumerate(self.blocks):
            x, x_control = block(x, c, x_control, layer_idx)

        x_count = self.final_layer(x, c)
        x_control_count = self.control_final_layer(x_control, c)

        x_count = self.output_fn(x_count)
        x_control_count = self.output_fn(x_control_count)

        return {
            "x": x_count,
            "x_control": x_control_count,
            #"cell_type_emb": None,
            "pert_intermediate_emb": x,
            "control_intermediate_emb": x_control,
        }

__all__ = ["Cross_DiT"]
