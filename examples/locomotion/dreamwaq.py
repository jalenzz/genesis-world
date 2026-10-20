import torch
from rsl_rl.runners import OnPolicyRunner


class DreamWaQRunner(OnPolicyRunner):
    """Train the walking policy and the context estimator on the same rollout."""

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # The first rollout has no finished episodes, so the actor starts on measured velocity.
        self.env.set_estimator_ground_truth(True)
        self.env.set_latent_sampling(True)
        update = self.alg.update
        next_it = self.current_learning_iteration

        def update_and_estimator():
            # The policy update clears the rollout, so the estimator has to read it first.
            nonlocal next_it
            estimator_loss = self.env.update_estimator(self.alg.storage)
            loss_dict = update()
            loss_dict["cenet"] = estimator_loss
            self.env.set_estimator_ground_truth(self.env.adaboot_uses_truth())
            # The rate is read after the update, which is where an adaptive schedule would have changed it.
            if self.alg.schedule == "fixed":
                print(
                    f"Learning iteration {next_it} learning_rate={self.alg.learning_rate!r} "
                    f"schedule={self.alg.schedule}",
                    flush=True,
                )
            next_it += 1
            return loss_dict

        self.alg.update = update_and_estimator
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path, infos=None):
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["cenet_state_dict"] = self.env.cenet_state_dict()
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        if "cenet_state_dict" in loaded_dict:
            self.env.load_cenet_state_dict(loaded_dict["cenet_state_dict"])
        return infos
