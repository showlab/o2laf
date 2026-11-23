"""Diagonal-Gaussian posterior over latent actions, and its KL to N(0, I)."""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianActionHead(nn.Module):
    """Project features to ``(mu, logvar)`` of a diagonal Gaussian action posterior.

    Moments and the reparameterisation run in fp32 regardless of the surrounding
    autocast dtype, because ``exp(0.5 * logvar)`` overflows easily in bf16. The
    projection is near-zero-initialised, so at step 0 the posterior is the prior
    and KL is ~0. In eval mode the returned ``action`` is the posterior mean.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if int(latent_dim) < 1:
            raise ValueError(f'latent_dim must be >= 1, got {latent_dim}')
        if float(logvar_min) >= float(logvar_max):
            raise ValueError(f'logvar_min={logvar_min} must be < logvar_max={logvar_max}')
        self.latent_dim = int(latent_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.to_moments = nn.Linear(input_dim, 2 * self.latent_dim)
        nn.init.normal_(self.to_moments.weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.to_moments.bias)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        moments = self.to_moments(hidden_states)
        moments = torch.nan_to_num(moments, nan=0.0, posinf=0.0, neginf=0.0)
        action_mu, action_logvar = moments.float().chunk(2, dim=-1)
        action_logvar = torch.nan_to_num(action_logvar, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            self.logvar_min, self.logvar_max
        )
        if self.training:
            deviation = torch.exp(0.5 * action_logvar)
            action = action_mu + torch.randn_like(deviation) * deviation
        else:
            action = action_mu
        return {
            'action': action.to(hidden_states.dtype),
            'action_mu': action_mu,
            'action_logvar': action_logvar,
        }


def kl_divergence(action_mu: torch.Tensor, action_logvar: torch.Tensor) -> torch.Tensor:
    """KL to a standard normal in nats per action: summed over the code, averaged elsewhere."""
    action_mu = action_mu.float()
    action_logvar = action_logvar.float()
    return -0.5 * (1 + action_logvar - action_mu.square() - action_logvar.exp()).sum(dim=-1).mean()


def kl_per_dimension(action_mu: torch.Tensor, action_logvar: torch.Tensor) -> torch.Tensor:
    """Per-code-dimension KL ``[latent_dim]``; a dimension near 0 carries nothing."""
    action_mu = action_mu.float()
    action_logvar = action_logvar.float()
    per_dimension = -0.5 * (1 + action_logvar - action_mu.square() - action_logvar.exp())
    return per_dimension.reshape(-1, per_dimension.shape[-1]).mean(dim=0)
