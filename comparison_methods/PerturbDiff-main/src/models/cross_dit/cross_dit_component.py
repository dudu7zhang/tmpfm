"""Cross-DiT shared components (merged from DiT/dit_embeddings/dit_blocks)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """Execute `modulate` and return values used by downstream logic."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """Special method `__init__`."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                 These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """Execute `forward` and return values used by downstream logic."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ContextEmbedding(nn.Module):
    """Contextembedding implementation used by the PerturbDiff pipeline."""
    def __init__(self, input_dim, hidden_dim):
        """Special method `__init__`."""
        super(ContextEmbedding, self).__init__()
        self.emb = nn.Linear(input_dim, hidden_dim, bias=True)
        self.null_emb = nn.Embedding(1, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, emb, batch_size):
        """Execute `forward` and return values used by downstream logic."""
        if emb is None:
            return self.null_emb.weight.expand(batch_size, -1, -1)
        if len(emb.shape) == 2:
            emb = emb.unsqueeze(1)
        emb = emb.type_as(self.null_emb.weight)
        return self.emb(emb)


class ClassEmbedding(nn.Module):
    """Classembedding implementation used by the PerturbDiff pipeline."""
    def __init__(self, input_dim, hidden_dim):
        """Special method `__init__`."""
        super(ClassEmbedding, self).__init__()
        self.emb = nn.Linear(input_dim, hidden_dim, bias=True)
        self.null_emb = nn.Embedding(1, hidden_dim)
        self.activation = nn.SiLU()
        self.hidden_dim = hidden_dim

    def forward(self, emb, batch_size):
        """Execute `forward` and return values used by downstream logic."""
        if emb is None:
            return self.null_emb.weight.expand(batch_size, -1, -1)
        emb = self.activation(emb)
        return self.emb(emb).type_as(self.null_emb.weight)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        """Special method `__init__`."""
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """Execute `forward` and return values used by downstream logic."""
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class GeneEmbedding(nn.Module):
    """Geneembedding implementation used by the PerturbDiff pipeline."""
    def __init__(self, hidden_size, gene_embedding_type="linear"):
        """Special method `__init__`."""
        super().__init__()
        self.gene_embedding_type = gene_embedding_type
        self.gene_embedding = (
            nn.Linear(5120, hidden_size, bias=True)
            if gene_embedding_type == "linear"
            else nn.Linear(1, hidden_size, bias=True)
        )
        self.hidden_dim = hidden_size
        self.null_emb = nn.Embedding(1, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size) if hidden_size > 1 else nn.Identity()

    def forward(self, x, batch_size, x_len):
        """Execute `forward` and return values used by downstream logic."""
        if x is None or torch.all(x == 0).item():
            return self.null_emb.weight.expand(batch_size, x_len, -1)
        if self.gene_embedding_type == "mean":
            x = x.mean(dim=-1, keepdim=True)
        x = self.gene_embedding(x)
        x = self.layer_norm(x)
        return x.type_as(self.gene_embedding.weight)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_size=1):
        """Special method `__init__`."""
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c, weight=None):
        """Execute `forward` and return values used by downstream logic."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        if weight is not None:
            x = F.linear(x, weight.T, self.linear.bias)
        else:
            x = self.linear(x)
        return x


__all__ = [
    "modulate",
    "TimestepEmbedder",
    "ContextEmbedding",
    "ClassEmbedding",
    "LabelEmbedder",
    "GeneEmbedding",
    "FinalLayer",
]
