import math

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_class, resolve_obs_groups
from tensordict import TensorDict
from torch import nn

import genesis as gs

EMBED_DIM = 128
TRANSFORMER_LAYERS = 2
ATTENTION_HEADS = 4
# Rollout bookkeeping stored beside the observations, so the minibatch shuffle keeps it aligned with each transition.
EXECUTED_TEACHER_KEY = "ult_is_teacher"


def ult_tokens(obs: TensorDict) -> torch.Tensor:
    """Return the state-action tokens z_k = (o_k, a_{k-1}) of the history window, oldest first."""
    return torch.cat((obs["proprio_history"], obs["action_history"]), dim=-1)


def ult_causal_attention_mask(history_steps, device):
    """Return the boolean attention mask of a history window followed by one privilege token, True where blocked.

    Every token attends to itself and the tokens before it. The privilege token comes last, so it attends to the whole
    history while no history token attends to it.
    """
    return ~torch.ones((history_steps + 1, history_steps + 1), dtype=torch.bool, device=device).tril()


class ULTActorModel(nn.Module):
    """Causal transformer actor of the Unified Locomotion Transformer (ULT), with a student and a teacher action.

    The history tokens share one linear embedding and the privilege goes through a three-layer encoder appended as the
    last token. A shared linear decoder maps each history output to the next state-action token. The action part of
    the prediction from the newest token is the student action. A three-layer MLP on the privilege output gives the
    teacher action. Both actions are the mean of the same Gaussian distribution.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        history_steps: int,
        distribution_cfg: dict,
        embed_dim: int = EMBED_DIM,
        n_layers: int = TRANSFORMER_LAYERS,
        n_heads: int = ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        dist_class, dist_cfg = resolve_class(distribution_cfg)
        self.distribution = dist_class(output_dim, **dist_cfg)
        # The student mean is a slice of the predicted token, so the distribution must read a bare mean.
        assert self.distribution.input_dim == output_dim
        self.proprio_dim = obs["proprio_history"].shape[-1]
        token_dim = self.proprio_dim + obs["action_history"].shape[-1]

        self.token_embed = nn.Linear(token_dim, embed_dim)
        self.privilege_encoder = nn.Sequential(
            nn.Linear(obs["privilege"].shape[-1], 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.next_token = nn.Linear(embed_dim, token_dim)
        self.teacher_action = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

        # Sinusoidal position encoding of the original transformer, over the history positions and the privilege one.
        position = torch.arange(history_steps + 1, dtype=gs.tc_float).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, embed_dim, 2, dtype=gs.tc_float) * (-math.log(10000.0) / embed_dim))
        positional_encoding = torch.zeros((history_steps + 1, embed_dim), dtype=gs.tc_float)
        positional_encoding[:, 0::2] = torch.sin(position * frequency)
        positional_encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("positional_encoding", positional_encoding, persistent=False)
        self.register_buffer("attn_mask", ult_causal_attention_mask(history_steps, "cpu"), persistent=False)

    def forward_heads(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full sequence and return the next-token predictions, the student action and the teacher action.

        The prediction at history position k is the token z_{k+1}. The last one carries the student action in its
        action part.
        """
        privilege = self.privilege_encoder(obs["privilege"]).unsqueeze(1)
        sequence = torch.cat((self.token_embed(ult_tokens(obs)), privilege), dim=1) + self.positional_encoding
        encoded = self.transformer(sequence, mask=self.attn_mask)
        next_tokens = self.next_token(encoded[:, :-1])
        return next_tokens, next_tokens[:, -1, self.proprio_dim :], self.teacher_action(encoded[:, -1])

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return the student action, computed from the history window alone.

        The privilege token comes after every history token and none of them attends to it, so dropping it leaves the
        student action unchanged. Deployment therefore needs no privilege input.
        """
        sequence = self.token_embed(ult_tokens(obs)) + self.positional_encoding[:-1]
        encoded = self.transformer(sequence, mask=self.attn_mask[:-1, :-1])
        mean = self.next_token(encoded[:, -1])[:, self.proprio_dim :]
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        pass


class ULTPPO(PPO):
    """PPO with the ULT action mixer, next state-action prediction and action imitation.

    Each environment plays the teacher action with probability ``ult_mix_alpha`` and the student action otherwise. The
    choice is drawn again for every rollout and stored with each transition, so the update evaluates the
    log-probability under the head that acted. The loss adds ``ult_aux_lambda * (L_n + ult_beta * L_a)`` to the PPO
    loss, where ``L_n`` is the squared error of the next state-action predictions over the history window and ``L_a``
    the squared distance between the teacher and student actions. The learning rate follows a cosine decay over
    ``lr_cosine_iterations`` updates.
    """

    def __init__(
        self,
        actor: ULTActorModel,
        critic: MLPModel,
        storage: RolloutStorage,
        lr_cosine_iterations: int,
        ult_mix_alpha: float = 0.6,
        ult_aux_lambda: float = 1.0,
        ult_beta: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.ult_mix_alpha = ult_mix_alpha
        self.ult_aux_lambda = ult_aux_lambda
        self.ult_beta = ult_beta
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=lr_cosine_iterations)
        self.is_teacher = torch.rand((storage.num_envs, 1), device=self.device) < self.ult_mix_alpha

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        _, student_action, teacher_action = self.actor.forward_heads(obs)
        self.actor.distribution.update(torch.where(self.is_teacher, teacher_action, student_action))
        self.transition.actions = self.actor.distribution.sample().detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs.copy()
        self.transition.observations[EXECUTED_TEACHER_KEY] = self.is_teacher
        return self.transition.actions

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_next_state_action_loss = 0.0
        mean_imitation_loss = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            obs = batch.observations
            next_tokens, student_action, teacher_action = self.actor.forward_heads(obs)
            self.actor.distribution.update(torch.where(obs[EXECUTED_TEACHER_KEY], teacher_action, student_action))
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            entropy = self.actor.output_entropy
            values = self.critic(obs)
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            next_state_action_loss = (ult_tokens(obs)[:, 1:] - next_tokens[:, :-1]).square().sum(dim=-1).mean()
            imitation_loss = (teacher_action - student_action).square().sum(dim=-1).mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.ult_aux_lambda * (next_state_action_loss + self.ult_beta * imitation_loss)
            )
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_next_state_action_loss += next_state_action_loss.item()
            mean_imitation_loss += imitation_loss.item()
        self.lr_scheduler.step()
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        self.is_teacher = torch.rand((self.storage.num_envs, 1), device=self.device) < self.ult_mix_alpha
        self.storage.clear()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "ult_next_state_action": mean_next_state_action_loss / num_updates,
            "ult_imitation": mean_imitation_loss / num_updates,
        }

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        self.lr_scheduler.load_state_dict(loaded_dict["lr_scheduler_state_dict"])
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "ULTPPO":
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        actor: ULTActorModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        storage_obs = obs.copy()
        storage_obs[EXECUTED_TEACHER_KEY] = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=obs.device)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], storage_obs, [env.num_actions], device)
        alg: ULTPPO = alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
