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
from go2_env import Go2Env, apply_curriculum, locomotion_cfgs, policy_obs_groups


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
    """Return the runner config, with estimator observation groups when the estimator is on."""
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
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--terrain",
        choices=("plane", "curriculum"),
        default="curriculum",
        help=(
            "Ground under the robot. 'curriculum' draws vx, vy and yaw from [-1, 1] and moves each terrain row up or "
            "down from tracking. 'plane' keeps the flat walker and its fixed forward command."
        ),
    )
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = locomotion_cfgs()
    if args.terrain == "curriculum":
        apply_curriculum(env_cfg, reward_cfg, command_cfg)
    train_cfg = configure_train_cfg(args.exp_name, env_cfg)

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

    runner_cls = DreamWaQRunner if env_cfg.get("estimator") == "cenet" else OnPolicyRunner
    runner = runner_cls(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training. Commands are sampled from [-1, 1]. Terrain rows move with tracking.
python examples/locomotion/go2_train.py -e go2-cmd-terrain

# training, flat ground and a fixed forward command
python examples/locomotion/go2_train.py -e go2-plane --terrain plane
"""
