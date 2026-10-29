import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

from go2_env import format_speed_curriculum


class CurriculumPPO(PPO):
    """PPO that also trains the context estimator and advances the curricula of the Go2 environment.

    The estimator reads the rollout before the policy update clears it. After the update, the forward speed cap may
    rise, and adaptive bootstrapping draws whether the next rollout feeds the actor the measured or the estimated
    velocity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = None
        self.iteration = 0

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        alg.env = env
        return alg

    def update(self):
        estimator_loss = self.env.update_estimator(self.storage)
        loss_dict = super().update()
        if self.env.is_dreamwaq_policy:
            loss_dict["cenet"] = estimator_loss
            self.env.set_estimator_ground_truth(self.env.adaboot_uses_truth())
        # The rate is read after the update, which is where an adaptive schedule would have changed it.
        if self.schedule == "fixed":
            print(
                f"Learning iteration {self.iteration} learning_rate={self.learning_rate!r} schedule={self.schedule}",
                flush=True,
            )
        promoted = self.env.advance_speed_cap()
        if promoted is not None:
            vx_max, tracking = promoted
            print(format_speed_curriculum(vx_max, self.iteration, tracking), flush=True)
        self.iteration += 1
        return loss_dict


class CurriculumRunner(OnPolicyRunner):
    """Train a Go2 walking policy with `CurriculumPPO`, and checkpoint the curricula and the context estimator.

    `load` reads the rsl-rl entries of `load_cfg` and three more: "curriculum" restores the speed and terrain curricula,
    "speed_curriculum" the forward speed cap alone, and "estimator" the context estimator of the DreamWaQ policy. A
    `load_cfg` of None loads everything.
    """

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.alg.iteration = self.current_learning_iteration
        if self.env.speed_vx_cap is not None:
            print(format_speed_curriculum(self.env.speed_vx_cap, self.current_learning_iteration, None), flush=True)
        if self.env.is_dreamwaq_policy:
            # The first rollout has no finished episodes, so the actor starts on measured velocity.
            self.env.set_estimator_ground_truth(True)
            self.env.set_latent_sampling(True)
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path, infos=None):
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["curriculum_state"] = self.env.curriculum_state
        if self.env.is_dreamwaq_policy:
            saved_dict["cenet_state_dict"] = self.env.cenet_state_dict()
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        is_terrain_loaded = load_cfg is None or load_cfg.get("curriculum", False)
        if is_terrain_loaded or load_cfg.get("speed_curriculum"):
            self.env.load_curriculum_state(loaded_dict["curriculum_state"], load_terrain=is_terrain_loaded)
        if self.env.is_dreamwaq_policy and (load_cfg is None or load_cfg.get("estimator")):
            self.env.load_cenet_state_dict(loaded_dict["cenet_state_dict"])
        return infos
