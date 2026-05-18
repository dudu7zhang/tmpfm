"""Cross-DiT block definitions split from DiT_cross_attention.py (logic-preserving)."""

import torch
import torch.nn as nn
from timm.layers.create_norm import get_norm_layer
from timm.models.vision_transformer import Attention, Mlp

from src.models.cross_dit.cross_dit_component import modulate
class MM_DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, norm_layer=None, **block_kwargs):
        """
        Initialize the class instance.

        :param hidden_size: Hidden feature dimension.
        :param num_heads: Count used to control loop/shape behavior.
        :param mlp_ratio: MLP width multiplier used in transformer blocks.
        :param norm_layer: Input `norm_layer` value.
        :param **block_kwargs: Additional keyword arguments forwarded downstream.
        :return: None.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if norm_layer is None:
            norm_layer = get_norm_layer("rmsnorm")
        self.attn = Attention(hidden_size*2, num_heads=num_heads, qkv_bias=True, 
                              qk_norm=True, norm_layer=norm_layer, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.control_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.control_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.control_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.control_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_control, layer_index: int):
        """
        Run the module forward pass.

        :param x: Input tensor.
        :param c: Input `c` value.
        :param x_control: Control-branch tensor.
        :param layer_index: Index value used for lookup or slicing.
        :return: Model output tensor(s) for the given inputs.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        control_shift_msa, control_scale_msa, control_gate_msa, control_shift_mlp, control_scale_mlp, control_gate_mlp = self.control_adaLN_modulation(c).chunk(6, dim=2)

        ln_scale = 1.0
        norm1_out = self.norm1(x) * ln_scale
        norm2_out = self.norm2(x) * ln_scale
        cnorm1_out = self.control_norm1(x_control) * ln_scale
        cnorm2_out = self.control_norm2(x_control) * ln_scale


        dx = modulate(norm1_out, shift_msa, scale_msa)
        dx_control = modulate(cnorm1_out, control_shift_msa, control_scale_msa)

        concat_x = torch.cat([dx, dx_control], dim=-1) # concatenate along feature dimension
        concat_x = self.attn(concat_x)
        dx, dx_control = torch.chunk(concat_x, 2, dim=-1)

        x = x + gate_msa * dx
        x = x + gate_mlp * self.mlp(modulate(norm2_out, shift_mlp, scale_mlp))

        x_control = x_control + control_gate_msa * dx_control
        x_control = x_control + control_gate_mlp * self.control_mlp(modulate(cnorm2_out, control_shift_mlp, control_scale_mlp))
        return x, x_control
    
class Cross_DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, layerf_id=0, Block_type="Cross_DiT", **block_kwargs):
        """Special method `__init__`."""
        super().__init__()
        self.layerf_id = layerf_id
        self.Block_type = Block_type
        if self.Block_type == "MM_DiT":
            block_kwargs.pop("use_control", None)
            self.block = MM_DiTBlock(
                hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                **block_kwargs,
            )
        else:
            raise NotImplementedError(f"Block_type {Block_type} not implemented.")

    def forward(self, x, c, x_control, layer_index: int):
        """Execute `forward` and return values used by downstream logic."""
        x, x_control = self.block(x, c, x_control, layer_index)
        return x, x_control


__all__ = ["MM_DiTBlock", "Cross_DiTBlock"]
