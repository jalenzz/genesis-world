import argparse
import os
import pickle
import time
from importlib import metadata

# com.apple.gamecontroller.driver.XboxGamepad owns the Microsoft pad (vid 1118, pid 2834), so the
# HIDAPI backend stays at 0. SDL then exposes the IOKit device, and only while a Cocoa window exists.
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
os.environ["SDL_JOYSTICK_HIDAPI"] = "0"
os.environ["SDL_JOYSTICK_HIDAPI_XBOX"] = "0"
os.environ["SDL_JOYSTICK_HIDAPI_XBOX_ONE"] = "0"
os.environ["SDL_JOYSTICK_MFI"] = "1"

import pygame
import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

from dreamwaq import DreamWaQRunner
from go2_env import TERRAIN_KINDS, Go2Env

# Resting Xbox sticks sit a little off center. Inside this band the command is 0.
STICK_DEADZONE = 0.15
# IOKit axis order: 0 left X, 1 left Y, 2 left trigger, 3 right X, 4 right Y, 5 right trigger.
LEFT_STICK_X = 0
LEFT_STICK_Y = 1
RIGHT_STICK_X = 3
# Held for the teleop run. The IOKit pad enumerates only while this Cocoa window is open.
_pad_window = None


def _load_cfgs(exp_name):
    with open(f"logs/{exp_name}/cfgs.pkl", "rb") as f:
        return pickle.load(f)


def _make_env(exp_name, num_envs, show_viewer, terrain):
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = _load_cfgs(exp_name)
    reward_cfg["reward_scales"] = {}
    if terrain == "curriculum":
        env_cfg["terrain"] = "curriculum"
        # A height field overflows the flat budget of 20 and the overflow reset leaves the robot standing still.
        env_cfg["max_collision_pairs"] = 256
    elif terrain == "plane":
        env_cfg["terrain"] = "plane"
    env = Go2Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )
    return env, env_cfg, reward_cfg, command_cfg, train_cfg


def _pin_commands(env, commands):
    env.commands.copy_(commands)
    env._update_observation()
    return env.get_observations()


def _group_rows(env, split_vx):
    """Label each environment with the terrain cell and the forward command it holds."""
    n = env._batch_n
    if env.terrain is None:
        levels = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        kinds = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        kind_names = ["plane"]
    else:
        mid = env._n_levels // 2
        levels = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        kinds = torch.arange(n, device=gs.device) % len(TERRAIN_KINDS)
        half = n // 2
        levels[half:] = mid
        kinds[half:] = torch.arange(n - half, device=gs.device) % len(TERRAIN_KINDS)
        env.terrain_levels.copy_(levels)
        env.terrain_types.copy_(kinds)
        kind_names = list(TERRAIN_KINDS)
    commands = torch.zeros((n, 3), dtype=gs.tc_float, device=gs.device)
    commands[:, 0] = split_vx[0]
    if split_vx[1] is not None:
        commands[n // 2 :, 0] = split_vx[1]
        if env.terrain is not None:
            # The second command replaces the mid-level half, so terrain stays on the first command.
            commands[:, 0] = split_vx[0]
    return levels, kinds, kind_names, commands


def _mean(values, mask):
    if not bool(mask.any()):
        return float("nan")
    return float(values[mask].mean())


def run_metrics(args):
    gs.init(backend=gs.cpu)

    cameras = []
    if args.video:
        original_build = gs.Scene.build

        def build_with_camera(scene, *build_args, **build_kwargs):
            cameras.append(
                scene.add_camera(
                    res=(640, 480),
                    pos=(1.6, -1.6, 1.0),
                    lookat=(0.0, 0.0, 0.3),
                    fov=50,
                    GUI=False,
                    near=0.05,
                    far=30.0,
                )
            )
            return original_build(scene, *build_args, **build_kwargs)

        gs.Scene.build = build_with_camera

    env, env_cfg, reward_cfg, command_cfg, train_cfg = _make_env(
        args.exp_name, args.num_envs, show_viewer=False, terrain=args.terrain
    )
    split_vx = (args.vx, args.also_vx)
    levels, kinds, kind_names, commands = _group_rows(env, split_vx)
    # Roll and pitch limits are the env fall flag. Widening them keeps a fallen base in place so displacement is the
    # motion over the window, and the same thresholds are applied to the recorded euler afterwards.
    roll_limit = env_cfg["termination_if_roll_greater_than"]
    pitch_limit = env_cfg["termination_if_pitch_greater_than"]
    env.env_cfg["termination_if_roll_greater_than"] = 180.0
    env.env_cfg["termination_if_pitch_greater_than"] = 180.0

    log_dir = f"logs/{args.exp_name}"
    runner_cls = DreamWaQRunner if env_cfg.get("estimator") == "cenet" else OnPolicyRunner
    runner = runner_cls(env, train_cfg, log_dir, device=gs.device)
    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    obs = _pin_commands(env, commands)
    camera = None
    if args.video:
        camera = cameras[0]
        origin = env.base_pos[0]
        camera.set_pose(
            pos=(origin[0] - 1.5, origin[1] - 1.6, origin[2] + 0.55),
            lookat=(origin[0], origin[1], origin[2] - 0.05),
        )
        camera.follow_entity(env.robot, fix_orientation=True)
        camera.start_recording(save_to_filename=args.video, fps=50)

    steps_4s = int(round(4.0 / env.dt))
    steps_total = int(round(args.episode_seconds / env.dt))
    video_steps = int(round(args.video_seconds / env.dt)) if args.video else 0
    sigma = reward_cfg["tracking_sigma"]
    origin_x = env.base_pos[:, 0].clone()
    fell = torch.zeros((env._batch_n,), dtype=torch.bool, device=gs.device)
    height_fell = torch.zeros_like(fell)
    dx_4s = torch.zeros((env._batch_n,), dtype=gs.tc_float, device=gs.device)
    vx_sum = torch.zeros((env._batch_n,), dtype=gs.tc_float, device=gs.device)
    track_sum = torch.zeros((env._batch_n,), dtype=gs.tc_float, device=gs.device)
    action_abs = 0.0
    action_count = 0

    with torch.no_grad():
        for step in range(steps_total):
            actions = policy(obs)
            action_abs += float(torch.abs(actions).sum())
            action_count += actions.numel()
            obs, _rews, _dones, _infos = env.step(actions)
            obs = _pin_commands(env, commands)
            vx = env.base_lin_vel[:, 0]
            vy = env.base_lin_vel[:, 1]
            vel_error = (vx - commands[:, 0]) ** 2 + (vy - commands[:, 1]) ** 2
            if step < steps_4s:
                vx_sum += vx
                track_sum += torch.exp(-vel_error / sigma)
                dx_4s = env.base_pos[:, 0] - origin_x
            roll = torch.abs(env.base_euler[:, 0])
            pitch = torch.abs(env.base_euler[:, 1])
            height = env.base_pos[:, 2] - env._terrain_z(env.base_pos[:, :2])
            fell |= (roll > roll_limit) | (pitch > pitch_limit)
            height_fell |= height < 0.15
            if camera is not None and step + 1 == video_steps:
                camera.stop_recording()
                camera = None

    if camera is not None:
        camera.stop_recording()

    collision_pairs = env.scene.rigid_solver.max_collision_pairs
    mean_abs_action = action_abs / action_count
    print(f"ckpt={ckpt_path}")
    print(f"terrain={env._terrain_kind} collision_pairs={collision_pairs}")
    print(
        f"command_cfg_vx={command_cfg['lin_vel_x_range']} vy={command_cfg['lin_vel_y_range']} yaw={command_cfg['ang_vel_range']}"
    )
    print(f"dt={env.dt} steps_4s={steps_4s} steps_total={steps_total}")
    print(f"mean_abs_action={mean_abs_action:.6f}")
    print(f"roll_pitch_limits_deg={roll_limit},{pitch_limit} height_collapse_m=0.150")

    if env._terrain_kind == "curriculum" and collision_pairs < 256:
        raise SystemExit(f"curriculum collision budget {collision_pairs} is below the training budget 256")
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


def _open_pad():
    """Open the Xbox pad through a Cocoa window, or return None when no pad is connected."""
    global _pad_window
    pygame.init()
    _pad_window = pygame.display.set_mode((640, 200))
    pygame.display.set_caption("xbox stick")
    pygame.joystick.init()
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline and pygame.joystick.get_count() < 1:
        pygame.event.pump()
        time.sleep(0.05)
    if pygame.joystick.get_count() < 1:
        _pad_window = None
        pygame.quit()
        return None
    pad = pygame.joystick.Joystick(0)
    pad.init()
    return pad


def _stick_command(pad):
    """Return vx, vy, and yaw in [-1, 1]. A stick inside the deadzone contributes 0."""
    pygame.event.get()
    # SDL reports X to the right and Y downward. vx is forward, vy is left, and yaw is left.
    vx = -pad.get_axis(LEFT_STICK_Y)
    vy = -pad.get_axis(LEFT_STICK_X)
    yaw = -pad.get_axis(RIGHT_STICK_X)
    return tuple(value if abs(value) >= STICK_DEADZONE else 0.0 for value in (vx, vy, yaw))


def _publish_command(env, command):
    env.commands[:, 0] = command[0]
    env.commands[:, 1] = command[1]
    env.commands[:, 2] = command[2]
    env._update_observation()
    return env.get_observations()


def run_teleop(args):
    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = _load_cfgs(args.exp_name)
    reward_cfg["reward_scales"] = {}
    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    pad = _open_pad()
    held_keys = set()

    def hold_key(name):
        held_keys.add(name)

    def release_key(name):
        held_keys.discard(name)

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    def axis_keybinds(name, key, held_name):
        return (
            Keybind(f"{name}_press", key, KeyAction.PRESS, callback=hold_key, args=(held_name,)),
            Keybind(f"{name}_release", key, KeyAction.RELEASE, callback=release_key, args=(held_name,)),
        )

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    env.scene.viewer.register_keybinds(
        *axis_keybinds("forward", Key.W, "w"),
        *axis_keybinds("backward", Key.S, "s"),
        *axis_keybinds("strafe_left", Key.Q, "q"),
        *axis_keybinds("strafe_right", Key.E, "e"),
        *axis_keybinds("yaw_left", Key.A, "a"),
        *axis_keybinds("yaw_right", Key.D, "d"),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    runner_cls = DreamWaQRunner if env_cfg.get("estimator") == "cenet" else OnPolicyRunner
    runner = runner_cls(env, train_cfg, log_dir, device=gs.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)

    pad_name = "none" if pad is None else pad.get_name()
    print(
        f"ckpt={ckpt_path} pad={pad_name} deadzone={STICK_DEADZONE} "
        f"vx=-axis{LEFT_STICK_Y} vy=-axis{LEFT_STICK_X} yaw=-axis{RIGHT_STICK_X} "
        "keys W/S vx, Q/E vy, A/D yaw, Esc quit"
    )

    obs_dict = env.reset()

    def command():
        vx = (1.0 if "w" in held_keys else 0.0) - (1.0 if "s" in held_keys else 0.0)
        vy = (1.0 if "q" in held_keys else 0.0) - (1.0 if "e" in held_keys else 0.0)
        yaw = (1.0 if "a" in held_keys else 0.0) - (1.0 if "d" in held_keys else 0.0)
        keys = (vx, vy, yaw)
        if pad is None:
            return keys
        stick = _stick_command(pad)
        return tuple(stick[i] if abs(stick[i]) >= STICK_DEADZONE else keys[i] for i in range(3))

    obs_dict = _publish_command(env, command())
    with torch.no_grad():
        while is_running and env.scene.viewer.is_alive():
            actions = policy(obs_dict)
            obs_dict, _rews, _dones, _infos = env.step(actions)
            obs_dict = _publish_command(env, command())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp-name",
        type=str,
        default="go2-smooth-slope",
        help="Experiment name; also the log directory under logs/",
    )
    parser.add_argument("--ckpt", type=int, default=2999, help="Checkpoint iteration to load")
    parser.add_argument(
        "--metrics", action="store_true", help="Print displacement, tracking and fall counts, then exit"
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--terrain", choices=("config", "plane", "curriculum"), default="config")
    parser.add_argument("--vx", type=float, default=0.5, help="Forward command in m/s, held for the whole run")
    parser.add_argument(
        "--also-vx", type=float, default=None, help="Forward command used by the second half of the envs"
    )
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--video-seconds", type=float, default=8.0)
    args = parser.parse_args()

    if args.metrics:
        run_metrics(args)
        return

    run_teleop(args)


if __name__ == "__main__":
    main()

"""
# Teleop on logs/go2-smooth-slope/model_2999.pt. Left stick is vx/vy, right stick x is yaw, each in [-1, 1].
# W/S, Q/E, and A/D command the same axes when the pad is absent or that stick is inside the deadzone.
uv run python examples/locomotion/go2_eval.py

# Metrics. The log's saved config selects flat ground or the curriculum terrain.
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100 --metrics
"""
