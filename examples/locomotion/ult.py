import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_class, resolve_obs_groups
from tensordict import TensorDict

import genesis as gs

from go2_env import SPEED_VX_CAPS, ULT_TOKEN_DIM, format_speed_curriculum

EMBED_DIM = 128
TRANSFORMER_LAYERS = 2
ATTENTION_HEADS = 4
# Rollout bookkeeping stored beside the observations, so the minibatch shuffle keeps it aligned with each transition.
EXECUTED_TEACHER_KEY = "ult_is_teacher"
NSA_TARGET_KEY = "ult_nsa_target"
NSA_VALID_KEY = "ult_is_nsa_valid"


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
            dropout=0.0,
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
            nn.Linear(embed_dim, ULT_TOKEN_DIM),
        )
        self._attn_mask = ult_causal_attention_mask(history_steps, device=obs.device)

    def _embed_history(self, obs: TensorDict) -> torch.Tensor:
        return self.token_embed(torch.cat((obs["proprio_history"], obs["action_history"]), dim=-1))

    def _encode_student(self, obs: TensorDict) -> torch.Tensor:
        # The privilege token sits after every proprio token and is masked out of their queries, so dropping it leaves
        # the student state unchanged and spares the privilege input at deployment.
        encoded = self.transformer(self._embed_history(obs), mask=self._attn_mask[:-1, :-1])
        return encoded[:, -1]

    def forward_heads(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return student and teacher distribution inputs and the next-state-action prediction."""
        privilege = self.privilege_encoder(obs["privilege"]).unsqueeze(1)
        sequence = torch.cat((self._embed_history(obs), privilege), dim=1)
        encoded = self.transformer(sequence, mask=self._attn_mask)
        student_state = encoded[:, self.history_steps - 1]
        teacher_state = encoded[:, self.history_steps]
        return self.student_action(student_state), self.teacher_action(teacher_state), self.nsa_head(student_state)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return the student action, which reads the proprio and action history alone."""
        mlp_output = self.student_action(self._encode_student(obs))
        if stochastic_output:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_output)

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None):
        return self._encode_student(obs)

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
    """PPO with teacher-student action mixing and ULT auxiliary losses.

    Every rollout transition records which action head produced its action (see ``EXECUTED_TEACHER_KEY``), so the
    update evaluates the log-probability under that same head. The next-state-action target of a transition is the
    newest proprio and action token of the following observation, valid only when the episode continues.
    """

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
        is_teacher = torch.rand((student_out.shape[0], 1), device=self.device) < self.ult_mix_alpha
        self.actor.distribution.update(torch.where(is_teacher, teacher_out, student_out))
        self.transition.actions = self.actor.distribution.sample().detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs.copy()
        self.transition.observations[EXECUTED_TEACHER_KEY] = is_teacher
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # The environment writes the executed action into the newest action token, beside the proprio it produced.
        nsa_target = torch.cat((obs["proprio_history"][:, -1], obs["action_history"][:, -1]), dim=-1)
        self.transition.observations[NSA_TARGET_KEY] = nsa_target
        self.transition.observations[NSA_VALID_KEY] = dones.reshape(-1, 1) == 0
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        mean_nsa = 0.0
        mean_imitation = 0.0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            obs = batch.observations
            student_out, teacher_out, nsa = self.actor.forward_heads(obs)
            self.actor.distribution.update(torch.where(obs[EXECUTED_TEACHER_KEY], teacher_out, student_out))
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
            is_nsa_valid = obs[NSA_VALID_KEY]
            nsa_error = (nsa - obs[NSA_TARGET_KEY]).pow(2).mean(dim=-1, keepdim=True)
            nsa_loss = (nsa_error * is_nsa_valid).sum() / is_nsa_valid.sum().clamp(min=1)
            student_mean = self.actor.distribution.deterministic_output(student_out)
            teacher_mean = self.actor.distribution.deterministic_output(teacher_out)
            imitation_loss = (student_mean - teacher_mean).pow(2).mean()
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
            mean_nsa += nsa_loss.item()
            mean_imitation += imitation_loss.item()
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
        storage_obs = obs.copy()
        storage_obs[EXECUTED_TEACHER_KEY] = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=obs.device)
        storage_obs[NSA_TARGET_KEY] = torch.zeros((env.num_envs, ULT_TOKEN_DIM), dtype=gs.tc_float, device=obs.device)
        storage_obs[NSA_VALID_KEY] = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=obs.device)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], storage_obs, [env.num_actions], device)
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
                return self.alg.actor(obs, stochastic_output=False)

        return policy
