import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from dreamwaq import DreamWaQRunner
from go2_env import (
    POLICY_DREAMWAQ,
    POLICY_MLP,
    Go2Env,
    apply_curriculum,
    locomotion_cfgs,
    is_dreamwaq_policy,
    policy_obs_groups,
)


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def configure_train_cfg(exp_name, env_cfg):
    """Return the runner config, with DreamWaQ observation groups when that policy is selected."""
    train_cfg = get_train_cfg(exp_name)
    train_cfg["obs_groups"] = policy_obs_groups(env_cfg)
    return train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=101, help="Number of learning iterations")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to load. Learning continues after it.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="PPO learning rate. When resuming, this replaces the checkpoint rate and the schedule stays fixed.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--terrain",
        choices=("plane", "curriculum"),
        default="curriculum",
        help=(
            "Ground under the robot. 'curriculum' starts vx in [-1, 1] and raises the upper end with tracking, "
            "draws vy from [-0.5, 0.5] and yaw from [-1, 1], and moves each terrain row up or down from tracking. "
            "'plane' keeps the flat walker and its fixed forward command."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=(POLICY_MLP, POLICY_DREAMWAQ),
        default=None,
        help="Actor policy. Defaults to dreamwaq with --terrain curriculum and mlp with plane.",
    )
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = locomotion_cfgs()
    if args.terrain == "curriculum":
        apply_curriculum(env_cfg, reward_cfg, command_cfg)
    if args.policy is not None:
        env_cfg["policy"] = args.policy
    elif args.terrain == "curriculum":
        env_cfg["policy"] = POLICY_DREAMWAQ
    else:
        env_cfg["policy"] = POLICY_MLP
    train_cfg = configure_train_cfg(args.exp_name, env_cfg)
    if args.learning_rate is not None:
        train_cfg["algorithm"]["learning_rate"] = args.learning_rate
        train_cfg["algorithm"]["schedule"] = "fixed"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Training throughput comes from the 4096 parallel environments, which need a GPU.
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner_cls = DreamWaQRunner if is_dreamwaq_policy(env_cfg) else OnPolicyRunner
    runner = runner_cls(env, train_cfg, log_dir, device=gs.device)
    if args.resume:
        runner.load(args.resume)
        # The file stores the iteration just finished. The learning loop starts at that index.
        runner.current_learning_iteration += 1
        # Loading restores the adaptive rate from the checkpoint. The requested rate replaces that value,
        # and a fixed schedule keeps later updates from changing it.
        if args.learning_rate is not None:
            runner.alg.schedule = "fixed"
            runner.alg.learning_rate = args.learning_rate
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = args.learning_rate

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training. Forward speed starts in [-1, 1] and its upper end rises with tracking.
# Lateral speed is sampled from [-0.5, 0.5], yaw from [-1, 1]. Terrain rows move with tracking.
python examples/locomotion/go2_train.py -e go2-cmd-terrain

# training, flat ground and a fixed forward command
python examples/locomotion/go2_train.py -e go2-plane --terrain plane
"""
