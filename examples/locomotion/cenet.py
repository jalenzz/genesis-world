from typing import NamedTuple

import torch
from torch import nn

from rsl_rl.modules import EmpiricalNormalization


# The encoder reads five proprioceptive steps (H = 5). The actor reads a sixteen-dimensional context vector.
HISTORY_STEPS = 5
LATENT_DIM = 16
_LOG_VARIANCE_LIMIT = 8.0


class ContextBelief(NamedTuple):
    velocity: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor


class CENet(nn.Module):
    """Estimate the base velocity and a context vector, and reconstruct the next proprioceptive observation.

    One encoder (128 x 64) reads the flattened proprioceptive history. A linear head regresses the body linear velocity
    and a second one the diagonal normal of the context vector, whose prior is the standard normal. The decoder
    (64 x 128) rebuilds the next observation from the context vector alone.

    Every proprioceptive frame, input and reconstruction target alike, goes through a running normalization to zero mean
    and unit variance, which is what makes the standard normal a sound prior. The normalization statistics are updated
    from the reconstruction targets of each training batch.

    `beta` weights the Kullback-Leibler (KL) divergence of the context vector against its prior. Both mean squared
    errors average over their components while the divergence sums over the latent components, so the default
    `2 / proprio_dim` weights the divergence as the evidence lower bound of a unit-variance Gaussian decoder does.
    """

    def __init__(self, proprio_dim, history_steps=HISTORY_STEPS, latent_dim=LATENT_DIM, beta=None):
        super().__init__()
        self.history_steps = history_steps
        self.latent_dim = latent_dim
        self.beta = 2.0 / proprio_dim if beta is None else beta
        self.normalizer = EmpiricalNormalization(proprio_dim)
        self.encoder = nn.Sequential(
            nn.Linear(proprio_dim * history_steps, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.velocity_head = nn.Linear(64, 3)
        self.mean_head = nn.Linear(64, latent_dim)
        self.log_variance_head = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, proprio_dim),
        )

    def encode(self, history):
        """Map a proprioceptive history to a body velocity and the diagonal normal of the context vector."""
        feature = self.encoder(self.normalizer(history).reshape(history.shape[0], -1))
        log_variance = self.log_variance_head(feature).clamp(-_LOG_VARIANCE_LIMIT, _LOG_VARIANCE_LIMIT)
        return ContextBelief(self.velocity_head(feature), self.mean_head(feature), log_variance)

    def reparameterize(self, belief):
        """Draw a context vector from the encoded normal."""
        std = torch.exp(0.5 * belief.log_variance)
        return belief.mean + std * torch.randn_like(std)

    def decode(self, latent):
        """Reconstruct the next normalized proprioceptive observation from the context vector."""
        return self.decoder(latent)

    def loss(self, history, velocity, next_proprio):
        """Return the hybrid loss L_CE = MSE(v~, v) + MSE(o~_{t+1}, o_{t+1}) + beta * KL(q(z | o^H) || N(0, I)).

        `history` is the proprioceptive history, `velocity` the measured body linear velocity and `next_proprio` the
        observation that follows the history. In training mode the normalization statistics first absorb
        `next_proprio`.
        """
        self.normalizer.update(next_proprio)
        belief = self.encode(history)
        reconstructed = self.decode(self.reparameterize(belief))
        velocity_loss = torch.mean(torch.square(belief.velocity - velocity))
        reconstruction_loss = torch.mean(torch.square(reconstructed - self.normalizer(next_proprio)))
        divergence = belief.log_variance - belief.mean.square() - belief.log_variance.exp()
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + divergence, dim=-1))
        return velocity_loss + reconstruction_loss + self.beta * kl_loss


def adaboot_probability(mean_return, std_return):
    """Return the adaptive bootstrapping probability p_boot = 1 - tanh(CV(R)) of the next learning iteration.

    `mean_return` and `std_return` are the mean and the standard deviation of the episodic returns R finished during
    the last rollout. The coefficient of variation CV divides the deviation by the magnitude of the mean, which keeps it
    positive when penalties make the mean negative. Returns that agree (small CV) give a high probability of feeding the
    actor the estimated velocity, and scattered returns give a high probability of feeding it the measured one.
    """
    return 1.0 - torch.tanh(std_return / mean_return.abs().clamp(min=1.0e-6))
