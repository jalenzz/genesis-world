import argparse
import math
import os
import pickle
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

import genesis as gs

from dreamwaq import CurriculumRunner
from go2_env import TERRAIN_KINDS, Go2Env


def make_eval_env(
    exp_name,
    num_envs,
    show_viewer,
    terrain,
    terrain_level=None,
    pin_easiest=False,
    randomize=False,
    rendered_envs_idx=None,
    add_camera=False,
):
    """Build the environment of a saved run for evaluation, and return it with its saved configs.

    `terrain` is "config" to keep the saved ground, or "plane" or "curriculum". `terrain_level` builds the curriculum
    rows 0 through that index at their saved slopes and spawns every environment on it. `pin_easiest` spawns on row 0.
    The terrain rows stay where they spawn. Falls are recorded in `env.is_fallen` without ending the episode, the
    episode never times out and the commands stay as `env.set_commands` sets them. `randomize` keeps the training
    domain randomization.
    """
    with open(f"logs/{exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    if terrain != "config":
        env_cfg["terrain"] = terrain
    terrain_kind = env_cfg.get("terrain", "plane")
    n_levels = env_cfg.get("terrain_levels", 10)
    spawn_level = None
    if terrain_level is not None:
        if terrain_kind != "curriculum":
            raise SystemExit("--terrain-level requires curriculum terrain")
        if terrain_level < 0 or terrain_level >= n_levels:
            raise SystemExit(f"--terrain-level {terrain_level} is outside 0..{n_levels - 1}")
        # Row i keeps its slope from build_curriculum_height_field. The mesh stops after the requested row.
        env_cfg["terrain_difficulty_levels"] = env_cfg.get("terrain_difficulty_levels", n_levels)
        env_cfg["terrain_levels"] = terrain_level + 1
        spawn_level = terrain_level
    elif pin_easiest and terrain_kind == "curriculum":
        spawn_level = 0
    if spawn_level is not None:
        env_cfg["initial_terrain_level"] = spawn_level
    env_cfg["terrain_curriculum"] = False
    env_cfg["terminate_on_fall"] = False
    env_cfg["resampling_time_s"] = None
    env_cfg["domain_randomization"] = randomize
    if rendered_envs_idx is not None:
        env_cfg["rendered_envs_idx"] = rendered_envs_idx
    env = Go2Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        add_camera=add_camera,
    )
    env.max_episode_length = math.inf
    return env, env_cfg, reward_cfg, command_cfg, train_cfg


def load_policy(env, env_cfg, train_cfg, exp_name, ckpt):
    """Load checkpoint `ckpt` of a run into `env` and return the inference policy and the checkpoint path.

    The actor, the context estimator of the DreamWaQ policy, and the forward speed cap the run reached are loaded.
    """
    log_dir = f"logs/{exp_name}"
    ckpt_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner = CurriculumRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(
        ckpt_path,
        load_cfg={"actor": True, "estimator": True, "speed_curriculum": env_cfg.get("speed_curriculum")},
    )
    return runner.get_inference_policy(device=gs.device), ckpt_path


def _group_rows(env, split_vx, terrain_level=None):
    """Label each environment with the terrain cell and the forward command it holds."""
    n = env.batch_size
    if env.terrain is None:
        levels = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        kinds = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        kind_names = ["plane"]
    else:
        levels = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        kinds = torch.arange(n, device=gs.device) % len(TERRAIN_KINDS)
        half = n // 2
        if terrain_level is None:
            levels[half:] = env.n_levels // 2
        else:
            levels.fill_(terrain_level)
        kinds[half:] = torch.arange(n - half, device=gs.device) % len(TERRAIN_KINDS)
        env.terrain_levels.copy_(levels)
        env.terrain_types.copy_(kinds)
        kind_names = list(TERRAIN_KINDS)
    commands = torch.zeros((n, 3), dtype=gs.tc_float, device=gs.device)
    commands[:, 0] = split_vx[0]
    if split_vx[1] is not None:
        commands[n // 2 :, 0] = split_vx[1]
    return levels, kinds, kind_names, commands


def _mean(values, mask):
    if not bool(mask.any()):
        return float("nan")
    return float(values[mask].mean())


def run_metrics(args):
    if args.also_vx is not None and args.terrain != "plane":
        # Curriculum metrics split the envs between the easiest and the middle row, which leaves no half for vx.
        with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
            saved_terrain = pickle.load(f)[0].get("terrain", "plane")
        if args.terrain == "curriculum" or saved_terrain == "curriculum":
            raise SystemExit("--also-vx requires flat ground")
    rendered_envs_idx = None
    if args.video and args.terrain != "plane":
        if args.num_envs < 2 * len(TERRAIN_KINDS):
            raise SystemExit(f"--video on curriculum terrain needs --num-envs >= {2 * len(TERRAIN_KINDS)}")
        # Obstacles and stairs are adjacent columns in the second half of the batch.
        half = args.num_envs // 2
        rendered_envs_idx = [
            half + TERRAIN_KINDS.index("discrete_obstacles"),
            half + TERRAIN_KINDS.index("stairs"),
        ]

    gs.init(backend=gs.cpu)
    env, env_cfg, reward_cfg, command_cfg, train_cfg = make_eval_env(
        args.exp_name,
        args.num_envs,
        show_viewer=False,
        terrain=args.terrain,
        terrain_level=args.terrain_level,
        randomize=args.randomize,
        rendered_envs_idx=rendered_envs_idx,
        add_camera=bool(args.video),
    )
    policy, ckpt_path = load_policy(env, env_cfg, train_cfg, args.exp_name, args.ckpt)
    split_vx = (args.vx, args.also_vx)
    levels, kinds, kind_names, commands = _group_rows(env, split_vx, terrain_level=args.terrain_level)

    env.reset()
    obs = env.set_commands(commands)
    camera = env.camera
    if camera is not None:
        if env.terrain is None:
            origin = env.base_pos[0]
            camera.set_pose(
                pos=(origin[0] - 1.5, origin[1] - 1.6, origin[2] + 0.55),
                lookat=(origin[0], origin[1], origin[2] - 0.05),
            )
            camera.follow_entity(env.robot, fix_orientation=True)
        else:
            row = env.n_levels // 2 if args.terrain_level is None else args.terrain_level
            centers = env.cell_origins[
                [row, row],
                [TERRAIN_KINDS.index("discrete_obstacles"), TERRAIN_KINDS.index("stairs")],
            ]
            look = centers.mean(dim=0)
            # Both columns stay in frame. Following one base would leave the other column.
            camera.set_pose(
                pos=(look[0] - 18.0, look[1], look[2] + 9.0),
                lookat=(look[0], look[1], look[2] - 0.2),
            )
        camera.start_recording(save_to_filename=args.video, fps=50)

    steps_4s = int(round(4.0 / env.dt))
    steps_total = int(round(args.episode_seconds / env.dt))
    video_steps = int(round(args.video_seconds / env.dt)) if args.video else 0
    sigma = reward_cfg["tracking_sigma"]
    origin_x = env.base_pos[:, 0].clone()
    fell = torch.zeros((env.batch_size,), dtype=torch.bool, device=gs.device)
    height_fell = torch.zeros_like(fell)
    dx_4s = torch.zeros((env.batch_size,), dtype=gs.tc_float, device=gs.device)
    vx_sum = torch.zeros((env.batch_size,), dtype=gs.tc_float, device=gs.device)
    track_sum = torch.zeros((env.batch_size,), dtype=gs.tc_float, device=gs.device)
    action_abs = 0.0
    action_count = 0

    with torch.no_grad():
        for step in range(steps_total):
            actions = policy(obs)
            action_abs += float(torch.abs(actions).sum())
            action_count += actions.numel()
            obs, _rews, _dones, _infos = env.step(actions)
            vx = env.base_lin_vel[:, 0]
            vy = env.base_lin_vel[:, 1]
            vel_error = (vx - commands[:, 0]) ** 2 + (vy - commands[:, 1]) ** 2
            if step < steps_4s:
                vx_sum += vx
                track_sum += torch.exp(-vel_error / sigma)
                dx_4s = env.base_pos[:, 0] - origin_x
            height = env.base_pos[:, 2] - env.terrain_height(env.base_pos[:, :2])
            fell |= env.is_fallen
            height_fell |= height < 0.15
            if camera is not None and step + 1 == video_steps:
                camera.stop_recording()
                camera = None

    if camera is not None:
        camera.stop_recording()

    collision_pairs = env.scene.rigid_solver.max_collision_pairs
    mean_abs_action = action_abs / action_count
    print(f"ckpt={ckpt_path}")
    print(f"terrain={env.terrain_kind} collision_pairs={collision_pairs} randomize={args.randomize}")
    print(
        f"command_cfg_vx={command_cfg['lin_vel_x_range']} vy={command_cfg['lin_vel_y_range']} "
        f"yaw={command_cfg['ang_vel_range']}"
    )
    print(f"dt={env.dt} steps_4s={steps_4s} steps_total={steps_total}")
    print(f"mean_abs_action={mean_abs_action:.6f}")
    print("fall=base_contact_or_tilt_limit height_collapse_m=0.150")

    if mean_abs_action < 1.0e-4 and float(torch.max(torch.abs(dx_4s))) < 0.01:
        raise SystemExit("robot is frozen: actions and displacement are both ~0")

    vx_values = commands[:, 0]
    reported = []
    for vx_value in torch.unique(vx_values).tolist():
        vx_mask = torch.abs(vx_values - vx_value) < 1.0e-6
        if env.terrain is None:
            level_kind = [(0, 0)]
        else:
            level_kind = []
            for level in torch.unique(levels[vx_mask]).tolist():
                for kind in range(len(kind_names)):
                    level_kind.append((level, kind))
        for level, kind in level_kind:
            mask = vx_mask & (levels == level) & (kinds == kind)
            if not bool(mask.any()):
                continue
            n_eps = int(mask.sum())
            n_fall = int((fell & mask).sum())
            n_height = int((height_fell & mask).sum())
            n_either = int(((fell | height_fell) & mask).sum())
            name = kind_names[kind]
            dx = _mean(dx_4s, mask)
            body_vx = _mean(vx_sum / steps_4s, mask)
            tracking = _mean(track_sum / steps_4s, mask)
            line = (
                f"result terrain={name} level={level} vx={vx_value:.3f} vy=0.000 yaw=0.000 "
                f"n={n_eps} dx_4s={dx:.3f} mean_body_vx={body_vx:.3f} tracking={tracking:.3f} "
                f"falls={n_fall}/{n_eps} fall_rate={n_fall / n_eps:.3f} "
                f"height_falls={n_height}/{n_eps} either_falls={n_either}/{n_eps} "
                f"either_rate={n_either / n_eps:.3f}"
            )
            print(line)
            reported.append(line)
    if args.video:
        size = os.path.getsize(args.video) if os.path.exists(args.video) else 0
        print(f"video={os.path.abspath(args.video)} bytes={size}")
    return reported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--terrain", choices=("config", "plane", "curriculum"), default="config")
    parser.add_argument(
        "--terrain-level",
        type=int,
        default=None,
        help=(
            "Curriculum row to spawn on, 0-based, 0 easiest. The mesh keeps rows 0 through this index at the saved "
            "curriculum slopes, and every env spawns on this row. Requires curriculum terrain."
        ),
    )
    parser.add_argument("--vx", type=float, default=0.5, help="Forward command in m/s, held for the whole run")
    parser.add_argument(
        "--also-vx", type=float, default=None, help="Forward command of the second half of the envs, flat ground only"
    )
    parser.add_argument(
        "--randomize", action="store_true", help="Keep the training domain randomization (mass, friction, gains)"
    )
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--video-seconds", type=float, default=8.0)
    run_metrics(parser.parse_args())


if __name__ == "__main__":
    main()

"""
# Metrics of logs/go2-walking/model_100.pt. The saved config selects flat ground or the curriculum terrain.
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100

# Curriculum row 2 only, 16 envs across the four columns
python examples/locomotion/go2_eval.py -e go2-cmd-terrain --ckpt 1000 --num-envs 16 --terrain-level 2
"""
