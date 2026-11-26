"""Lightweight clip-level classification heads for transition feature sequences."""

from __future__ import annotations

import torch
from torch import nn

HEAD_ARCHITECTURES = ('direct_linear', 'mean_linear', 'mean_mlp', 'attention')


class AttentionPool(nn.Module):
    """Single-query masked temporal attention pooling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(features).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum('bt,btd->bd', weights, features)


class DirectCameraHead(nn.Module):
    """Per-transition LayerNorm, masked clip mean, and linear projection.

    Used across all four adaptation modes for strictly matched comparisons.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.input_norm(features)
        if mask is None:
            pooled = features.mean(dim=1)
        else:
            weights = mask.to(features.dtype).unsqueeze(-1)
            pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


class MeanPoolCameraHead(nn.Module):
    """Masked mean pooling followed by a linear layer or one-hidden-layer MLP."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            pooled = features.mean(dim=1)
        else:
            weights = mask.to(features.dtype).unsqueeze(-1)
            pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


class CameraMotionHead(nn.Module):
    """Per-transition LayerNorm, masked attention pooling, and one-hidden-layer MLP."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.pool = AttentionPool(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.pool(self.input_norm(features), mask))


def build_head(
    architecture: str,
    input_dim: int,
    num_classes: int,
    *,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> nn.Module:
    """Construct a clip classification head by name."""
    if architecture == 'direct_linear':
        return DirectCameraHead(input_dim, num_classes)
    if architecture == 'mean_linear':
        return MeanPoolCameraHead(input_dim, num_classes, hidden_dim=0, dropout=0.0)
    if architecture == 'mean_mlp':
        return MeanPoolCameraHead(input_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    if architecture == 'attention':
        return CameraMotionHead(input_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f'Unknown head architecture {architecture!r}; expected one of {HEAD_ARCHITECTURES}.')
