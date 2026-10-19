from typing import NamedTuple

import torch
from torch import nn


# Five proprioceptive steps is the encoder's context. Sixteen is the width of the latent the actor reads.
HISTORY_STEPS = 5
LATENT_DIM = 16
_LOG_VARIANCE_LIMIT = 8.0


class ContextBelief(NamedTuple):
    velocity: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor


class CENet(nn.Module):
    """Estimate base velocity and a latent, and reconstruct the next proprioceptive observation.

    One encoder reads the flattened proprioceptive history. A linear head regresses the base linear velocity. A second
    head is a variational encoder whose latent prior is the standard normal, and the decoder rebuilds the next
    observation from that latent alone.
    """

    def __init__(self, proprio_dim, history_steps=HISTORY_STEPS, latent_dim=LATENT_DIM):
        super().__init__()
        self.history_steps = history_steps
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(proprio_dim * history_steps, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.velocity_head = nn.Linear(128, 3)
        self.mean_head = nn.Linear(128, latent_dim)
        self.log_variance_head = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, proprio_dim),
        )

    def encode(self, history):
        """Map a proprioceptive history to a velocity and a diagonal normal."""
        feature = self.encoder(history.reshape(history.shape[0], -1))
        log_variance = self.log_variance_head(feature).clamp(-_LOG_VARIANCE_LIMIT, _LOG_VARIANCE_LIMIT)
        return ContextBelief(self.velocity_head(feature), self.mean_head(feature), log_variance)

    def reparameterize(self, belief):
        """Draw a latent from the encoded normal."""
        std = torch.exp(0.5 * belief.log_variance)
        return belief.mean + std * torch.randn_like(std)

    def decode(self, latent):
        """Reconstruct the next proprioceptive observation from the latent."""
        return self.decoder(latent)

    def loss(self, history, velocity, next_proprio):
        """Return velocity error plus reconstruction error plus the latent's divergence from the standard normal."""
        belief = self.encode(history)
        reconstructed = self.decode(self.reparameterize(belief))
        velocity_loss = torch.mean(torch.square(belief.velocity - velocity))
        reconstruction_loss = torch.mean(torch.square(reconstructed - next_proprio))
        # Sum over the latent and average over the batch, which is the KL of a diagonal normal against N(0, I).
        divergence = belief.log_variance - belief.mean.square() - belief.log_variance.exp()
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + divergence, dim=-1))
        return velocity_loss + reconstruction_loss + kl_loss


def adaboot_truth_probability(returns):
    """Map completed episode returns to the probability of training on ground truth.

    The probability is the hyperbolic tangent of the coefficient of variation, so a batch whose returns agree stays on
    the estimator and a batch whose returns scatter falls back to ground truth. An empty batch, or a batch whose mean
    return vanishes, is treated as ground truth.
    """
    if returns.numel() == 0:
        return torch.ones((), dtype=returns.dtype, device=returns.device)
    mean_return = returns.mean()
    if not bool(mean_return.abs() > 1.0e-6):
        return torch.ones((), dtype=returns.dtype, device=returns.device)
    coefficient = returns.std(unbiased=False) / mean_return.abs()
    return torch.tanh(coefficient)


def adaboot_uses_truth(returns, draw):
    """Choose ground truth for a whole batch from one draw against the return coefficient of variation."""
    if returns.numel() == 0:
        return True
    return bool(draw < adaboot_truth_probability(returns))
