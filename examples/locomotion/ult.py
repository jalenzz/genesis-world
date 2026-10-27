import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_class, resolve_obs_groups

import genesis as gs

from go2_env import SPEED_VX_CAPS, format_speed_curriculum

EMBED_DIM = 128
TRANSFORMER_LAYERS = 2
ATTENTION_HEADS = 4
NSA_OUTPUT_DIM = 49


def ult_causal_attention_mask(history_steps, device):
    """Return a transformer mask that blocks disallowed attention positions.

    The sequence is ``history_steps`` proprio-action tokens followed by one privilege token. Proprio queries keep a
    causal mask over proprio keys and cannot attend the privilege key. The privilege query may attend all proprio keys.
    """
    seq_len = history_steps + 1
    priv_idx = history_steps
    allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for query in range(history_steps):
        for key in range(history_steps):
            allowed[query, key] = key <= query
    for key in range(history_steps):
        allowed[priv_idx, key] = True
    allowed[priv_idx, priv_idx] = True
    return ~allowed


class ULTActorModel(nn.Module):
    """Causal-transformer actor with student and teacher action heads."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        history_steps: int,
        proprio_dim: int,
        token_dim: int,
        embed_dim: int = EMBED_DIM,
        n_layers: int = TRANSFORMER_LAYERS,
        n_heads: int = ATTENTION_HEADS,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if distribution_cfg is None:
            raise ValueError("ULTActorModel requires a Gaussian distribution configuration.")
        dist_class, dist_cfg = resolve_class(distribution_cfg)
        self.distribution = dist_class(output_dim, **dist_cfg)
        self.history_steps = history_steps
        self.proprio_dim = proprio_dim
        self.token_dim = token_dim
        self.obs_groups = obs_groups[obs_set]
        self._student_only = False
        self._execution_teacher = None

        self.token_embed = nn.Linear(token_dim, embed_dim)
        privilege_dim = obs["privilege"].shape[-1]
        self.privilege_encoder = nn.Sequential(
            nn.Linear(privilege_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.student_action = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, self.distribution.input_dim),
        )
        self.teacher_action = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, self.distribution.input_dim),
        )
        self.nsa_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, NSA_OUTPUT_DIM),
        )
        self._attn_mask = ult_causal_attention_mask(history_steps, device=obs.device)

    def _build_tokens(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = obs["proprio_history"]
        actions = obs["action_history"]
        tokens = torch.cat((proprio, actions), dim=-1)
        embedded = self.token_embed(tokens)
        privilege = self.privilege_encoder(obs["privilege"]).unsqueeze(1)
        return torch.cat((embedded, privilege), dim=1), privilege

    def _encode(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, privilege_token = self._build_tokens(obs)
        if self._student_only:
            sequence = sequence[:, : self.history_steps]
            student_mask = torch.triu(
                torch.ones((self.history_steps, self.history_steps), dtype=torch.bool, device=sequence.device),
                diagonal=1,
            )
            encoded = self.transformer(sequence, mask=student_mask)
            student_state = encoded[:, -1]
            return student_state, student_state
        encoded = self.transformer(sequence, mask=self._attn_mask)
        student_state = encoded[:, self.history_steps - 1]
        teacher_state = encoded[:, self.history_steps]
        return student_state, teacher_state

    def forward_heads(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return student and teacher distribution inputs and the next-state-action prediction."""
        student_state, teacher_state = self._encode(obs)
        student_out = self.student_action(student_state)
        teacher_out = self.teacher_action(teacher_state)
        nsa = self.nsa_head(student_state)
        return student_out, teacher_out, nsa

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        student_only: bool = False,
    ) -> torch.Tensor:
        previous_student_only = self._student_only
        self._student_only = student_only
        student_out, teacher_out, _ = self.forward_heads(obs)
        self._student_only = previous_student_only
        if student_only or self._execution_teacher is None:
            mlp_output = student_out
        else:
            mlp_output = torch.where(self._execution_teacher.unsqueeze(-1), teacher_out, student_out)
        if stochastic_output:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_output)

    def set_execution_teacher(self, use_teacher: torch.Tensor | None) -> None:
        """Record which action head supplied the rollout action."""
        self._execution_teacher = use_teacher

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None):
        student_state, _ = self._encode(obs)
        return student_state

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

    def imitation_loss(self, obs: TensorDict) -> torch.Tensor:
        student_out, teacher_out, _ = self.forward_heads(obs)
        student_mean = self.distribution.deterministic_output(student_out)
        teacher_mean = self.distribution.deterministic_output(teacher_out)
        return (student_mean - teacher_mean).pow(2).mean()

    def next_state_action_loss(self, obs: TensorDict) -> torch.Tensor:
        _, _, nsa = self.forward_heads(obs)
        return (nsa - obs["nsa_target"]).pow(2).mean()


class ULTPPO(PPO):
    """PPO with teacher-student action mixing and ULT auxiliary losses."""

    def __init__(
        self,
        actor: ULTActorModel,
        critic: MLPModel,
        storage: RolloutStorage,
        ult_mix_alpha: float = 0.6,
        ult_aux_lambda: float = 1.0,
        ult_beta: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.ult_mix_alpha = ult_mix_alpha
        self.ult_aux_lambda = ult_aux_lambda
        self.ult_beta = ult_beta

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        student_out, teacher_out, _ = self.actor.forward_heads(obs)
        batch_n = student_out.shape[0]
        use_teacher = torch.rand((batch_n,), device=self.device) < self.ult_mix_alpha
        self.actor.set_execution_teacher(use_teacher)
        mlp_output = torch.where(use_teacher.unsqueeze(-1), teacher_out, student_out)
        self.actor.distribution.update(mlp_output)
        self.transition.actions = self.actor.distribution.sample().detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def _ult_auxiliary_loss(
        self, batch_obs: TensorDict, batch_dones: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_dones is None:
            alive = torch.ones((batch_obs.batch_size[0],), dtype=torch.bool, device=self.device)
        else:
            alive = batch_dones.squeeze(-1) == 0
        if not bool(alive.any()):
            zero = torch.zeros((), device=self.device)
            return zero, zero
        obs_alive = batch_obs[alive]
        imitation = self.actor.imitation_loss(obs_alive)
        nsa = self.actor.next_state_action_loss(obs_alive)
        return nsa, imitation

    def update(self) -> dict[str, float]:
        mean_nsa = 0.0
        mean_imitation = 0.0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]
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
            batch_dones = batch.dones[:original_batch_size] if batch.dones is not None else None
            nsa_loss, imitation_loss = self._ult_auxiliary_loss(
                batch.observations[:original_batch_size],
                batch_dones,
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.ult_aux_lambda * (nsa_loss + self.ult_beta * imitation_loss)
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
            mean_nsa += float(nsa_loss.detach())
            mean_imitation += float(imitation_loss.detach())
        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "ult_nsa": mean_nsa / num_updates,
            "ult_imitation": mean_imitation / num_updates,
        }

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "ULTPPO":
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        actor: ULTActorModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg: ULTPPO = alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


class ULTRunner(OnPolicyRunner):
    """On-policy runner for the ULT transformer policy."""

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if not self.env.is_ult_policy:
            raise gs.GenesisException("ULTRunner requires env_cfg['policy'] == 'ult'.")
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.alg.optimizer,
            T_max=num_learning_iterations,
            eta_min=self.alg.learning_rate * 0.1,
        )
        next_it = self.current_learning_iteration
        if self.env._speed_level is not None:
            print(format_speed_curriculum(SPEED_VX_CAPS[self.env._speed_level], next_it, None), flush=True)

        def step_scheduler(loss_dict):
            nonlocal next_it
            cosine.step()
            promoted = self.env.advance_speed_cap()
            if promoted is not None:
                vx_max, tracking = promoted
                print(format_speed_curriculum(vx_max, next_it, tracking), flush=True)
            next_it += 1
            return loss_dict

        update = self.alg.update
        self.alg.update = lambda: step_scheduler(update())
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def get_inference_policy(self, device=None):
        device = device or self.device
        self.alg.eval_mode()

        def policy(obs):
            obs = obs.to(device)
            with torch.inference_mode():
                return self.alg.actor(obs, stochastic_output=False, student_only=True)

        return policy
