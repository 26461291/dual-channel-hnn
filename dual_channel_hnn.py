"""Dual-channel cis/trans eQTL HNN model.

Inputs are batched standardized cis/trans genotype tensors. The trans channel
also receives signed positions and one-hot module labels. The module predicts
cis additive effects, low-rank cis/trans factors, the interaction matrix, and
the phenotype. It requires PyTorch; see README.md for usage and training notes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    """Fixed-adjacency message passing with a separate self-feature path."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.neighbor = nn.Linear(width, width, bias=False)
        self.self_connection = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        neighbors = torch.einsum("uv,bvh->buh", adjacency, x)
        return self.norm(F.relu(self.neighbor(neighbors) + self.self_connection(x)))


class DualChannelHNN(nn.Module):
    """Bidirectional cis/trans HNN with low-rank interaction decoding.

    Args:
        hidden_dim: Hidden size of each direction in the cis BiGRU. The shared
            representation width is 2 * hidden_dim.
        rank: Rank of the factorized interaction gamma = z @ w.T.
        adjacency: Symmetrically normalized trans-SNP adjacency, shape
            (n_trans, n_trans).
        trans_positions: Signed trans-SNP positions relative to the cis center.
        module_ids: Integer module label for each trans SNP.
        target_gamma_std: Target standard deviation used to initialize factor
            gains to a controlled interaction scale.
        beta_initial_std: Target standard deviation used to initialize the
            cis additive-effect gain.
        attention_heads: Number of heads in each cross-attention direction.
        n_gnn_layers: Number of fixed-adjacency graph-convolution layers.

    Forward inputs have shapes x_cis=(B, n_cis), x_trans=(B, n_trans).
    Returns (y_hat, beta, z, w, gamma) with shapes (B,), (B,n_cis),
    (B,n_cis,rank), (B,n_trans,rank), (B,n_cis,n_trans).
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        adjacency: Tensor,
        trans_positions: Tensor,
        module_ids: Tensor,
        target_gamma_std: float,
        beta_initial_std: float,
        attention_heads: int = 4,
        n_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        width = 2 * hidden_dim
        if width % attention_heads:
            raise ValueError("2 * hidden_dim must be divisible by attention_heads")
        if trans_positions.ndim != 1 or module_ids.shape != trans_positions.shape:
            raise ValueError("trans_positions and module_ids must be 1D with matching shapes")
        if adjacency.shape != (trans_positions.numel(), trans_positions.numel()):
            raise ValueError("adjacency shape must match the number of trans SNPs")

        self.register_buffer("adjacency", adjacency)
        module_labels = F.one_hot(
            module_ids.long(), num_classes=int(module_ids.max().item()) + 1
        ).to(trans_positions.dtype)
        max_abs_position = trans_positions.abs().max().clamp_min(1e-8)
        self.register_buffer(
            "trans_position_features",
            torch.cat(
                ((trans_positions / max_abs_position).unsqueeze(-1), module_labels),
                dim=-1,
            ),
        )
        self.cis_gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        trans_feature_count = 1 + self.trans_position_features.shape[-1]
        self.trans_embedding = nn.Linear(trans_feature_count, width)
        self.gnn_layers = nn.ModuleList(
            GraphConvolution(width) for _ in range(n_gnn_layers)
        )
        self.cis_to_trans = nn.MultiheadAttention(
            width, attention_heads, batch_first=True
        )
        self.trans_to_cis = nn.MultiheadAttention(
            width, attention_heads, batch_first=True
        )

        self.cis_cross_norm = nn.LayerNorm(width)
        self.trans_cross_norm = nn.LayerNorm(width)
        self.cis_ffn = self._make_ffn(width)
        self.trans_ffn = self._make_ffn(width)
        self.cis_ffn_norm = nn.LayerNorm(width)
        self.trans_ffn_norm = nn.LayerNorm(width)

        self.cis_beta_head = nn.Linear(width, 1)
        self.cis_z_head = nn.Linear(width, rank)
        self.trans_w_head = nn.Linear(width, rank)
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.beta_gain = nn.Parameter(torch.tensor(1.0))
        self.z_gain = nn.Parameter(torch.tensor(1.0))
        self.w_gain = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("beta_gain_target", torch.tensor(beta_initial_std))
        self.register_buffer(
            "gamma_gain_target", torch.tensor(math.sqrt(target_gamma_std / rank))
        )

        for head in (self.cis_beta_head, self.cis_z_head, self.trans_w_head):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _make_ffn(width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * width, width),
        )

    def raw_factors(
        self, x_cis: Tensor, x_trans: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        cis_input = x_cis.unsqueeze(-1)
        position_features = self.trans_position_features.unsqueeze(0).expand(
            x_trans.shape[0], -1, -1
        )
        trans_input = torch.cat((x_trans.unsqueeze(-1), position_features), dim=-1)

        cis_out, _ = self.cis_gru(cis_input)
        trans_out = F.relu(self.trans_embedding(trans_input))
        for layer in self.gnn_layers:
            trans_out = layer(trans_out, self.adjacency)

        cis_context, _ = self.cis_to_trans(cis_out, trans_out, trans_out)
        trans_context, _ = self.trans_to_cis(trans_out, cis_out, cis_out)
        cis_repr = self.cis_cross_norm(cis_out + cis_context)
        trans_repr = self.trans_cross_norm(trans_out + trans_context)
        cis_repr = self.cis_ffn_norm(cis_repr + self.cis_ffn(cis_repr))
        trans_repr = self.trans_ffn_norm(trans_repr + self.trans_ffn(trans_repr))

        beta = self.cis_beta_head(cis_repr).squeeze(-1) * self.beta_gain
        z = self.cis_z_head(cis_repr)
        w = self.trans_w_head(trans_repr)
        return beta, z, w

    @torch.no_grad()
    def calibrate_factor_gains(self, x_cis: Tensor, x_trans: Tensor) -> None:
        """Calibrate output gains from a representative training batch."""
        was_training = self.training
        self.eval()
        beta_raw, z_raw, w_raw = self.raw_factors(x_cis, x_trans)
        self.beta_gain.copy_(self.beta_gain_target / beta_raw.std().clamp_min(1e-8))
        self.z_gain.copy_(self.gamma_gain_target / z_raw.std().clamp_min(1e-8))
        self.w_gain.copy_(self.gamma_gain_target / w_raw.std().clamp_min(1e-8))
        self.train(was_training)

    def forward(
        self, x_cis: Tensor, x_trans: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        beta, z_raw, w_raw = self.raw_factors(x_cis, x_trans)
        z = z_raw * self.z_gain
        w = w_raw * self.w_gain
        gamma = torch.einsum("btr,bnr->btn", z, w)
        y_hat = (
            self.mu
            + torch.einsum("bt,bt->b", beta, x_cis)
            + torch.einsum("btn,bt,bn->b", gamma, x_cis, x_trans)
        )
        return y_hat, beta, z, w, gamma
