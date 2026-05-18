"""Initialization and layer-replacement helpers for Cross_DiT."""

import torch
import torch.nn as nn

from src.common.utils import get_short_dsname
from src.models.cross_dit.cross_dit_component import FinalLayer


def initialize_weights(self):
    # Initialize transformer layers:
    """
    Initialize weights.

    :return: None.
    """
    def _basic_init(module):
        """Execute `_basic_init` and return values used by downstream logic."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    self.apply(_basic_init)

    # Initialize timestep embedding MLP:
    nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
    nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    # Initialize gene embedding:
    nn.init.normal_(self.gene_embedding.gene_embedding.weight, std=0.02)
    nn.init.constant_(self.gene_embedding.gene_embedding.bias, 0)

    # Zero-out adaLN modulation layers in DiT blocks:
    for block_ in self.blocks:
        nn.init.constant_(block_.block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block_.block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(block_.block.control_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block_.block.control_adaLN_modulation[-1].bias, 0)

    # Zero-out output layers:
    nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(self.control_final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.control_final_layer.adaLN_modulation[-1].bias, 0)

    if (self.model_cfg.predict_xstart) and self.output_fn != nn.Identity():
        pass
    else:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.control_final_layer.linear.weight, 0)
        nn.init.constant_(self.control_final_layer.linear.bias, 0)


def reinitialize_adaln(self):
    # Zero-out adaLN modulation layers in DiT blocks:
    """Execute `reinitialize_adaln` and return values used by downstream logic."""
    for block_ in self.blocks:
        nn.init.constant_(block_.block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block_.block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(block_.block.control_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block_.block.control_adaLN_modulation[-1].bias, 0)


def replace_2kgene_layer(self, new_input_size):
    """
    Replace 2kgene layer.

    :param new_input_size: New gene/input width after remapping.
    :return: Computed output(s) for this function.
    """
    new_output_size = new_input_size
    new_input_size *= 2

    # embedder layer
    if self.model_cfg.separate_embedder:
        x_embedder, x_control_embedder = {}, {}
        for ds_name in self.model_cfg.dataset_dict:
            if self.model_cfg.separate_embedder == "by_name":
                ds_name = get_short_dsname(ds_name)
            x_embedder[ds_name] = nn.Linear(new_input_size, self.hidden_size, bias=True)
            x_control_embedder[ds_name] = nn.Linear(new_input_size, self.hidden_size, bias=True)
        self.x_embedder = nn.ModuleDict(x_embedder)
        self.x_control_embedder = nn.ModuleDict(x_control_embedder)
    else:
        self.x_embedder = nn.Linear(new_input_size, self.hidden_size, bias=True)
        self.x_control_embedder = nn.Linear(new_input_size, self.hidden_size, bias=True)

    def _basic_init(module):
        """Execute `_basic_init` and return values used by downstream logic."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    self.x_embedder.apply(_basic_init)
    self.x_control_embedder.apply(_basic_init)

    # final layer
    self.final_layer = FinalLayer(self.hidden_size, output_size=new_output_size)
    self.control_final_layer = FinalLayer(self.hidden_size, output_size=new_output_size)

    nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(self.control_final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.control_final_layer.adaLN_modulation[-1].bias, 0)
