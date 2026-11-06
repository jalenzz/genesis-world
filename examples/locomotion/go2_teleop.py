import argparse
import os
import time

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

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, xyz_to_quat
from genesis.vis.keybindings import Key, KeyAction, Keybind

from go2_eval import load_policy, make_eval_env


# Resting Xbox sticks sit a little off center. Inside this band the command is 0.
STICK_DEADZONE = 0.15
# Chase camera in the base heading frame (yaw only, world z up). +x is forward, so this sits behind and above.
CHASE_OFFSET = (-1.5, 0.0, 0.55)
# IOKit axis order: 0 left X, 1 left Y, 2 left trigger, 3 right X, 4 right Y, 5 right trigger.
LEFT_STICK_X = 0
LEFT_STICK_Y = 1
RIGHT_STICK_X = 3
_HELD_AXES = (
    ("forward", Key.W, "w"),
    ("backward", Key.S, "s"),
    ("strafe_left", Key.Q, "q"),
    ("strafe_right", Key.E, "e"),
    ("yaw_left", Key.A, "a"),
    ("yaw_right", Key.D, "d"),
)


class TeleopInput:
    """Keyboard and Xbox-pad commands for one teleop session.

    The pad window stays open only while a pad is connected. Keys are the viewer's own bindings.
    """

    def __init__(self):
        self.held_keys = set()
        self.pad = None
        self._window = None
        self.is_running = True
        self.is_reset_requested = False

    def open_pad(self):
        """Open the Xbox pad through a Cocoa window. Leave the pad unset when none is connected."""
        pygame.init()
        self._window = pygame.display.set_mode((640, 200))
        pygame.display.set_caption("xbox stick")
        pygame.joystick.init()
        deadline = time.perf_counter() + 3
        while time.perf_counter() < deadline and pygame.joystick.get_count() < 1:
            pygame.event.pump()
            time.sleep(0.05)
        if pygame.joystick.get_count() < 1:
            self.close_pad()
            return
        self.pad = pygame.joystick.Joystick(0)
        self.pad.init()

    def close_pad(self):
        self._window = None
        self.pad = None
        pygame.quit()

    def hold(self, name):
        self.held_keys.add(name)

    def release(self, name):
        self.held_keys.discard(name)

    def stop(self):
        self.is_running = False

    def request_reset(self):
        self.is_reset_requested = True

    def keybinds(self):
        """Return the viewer bindings: W/S vx, Q/E vy, A/D yaw, R reset, Esc quit."""
        binds = []
        for name, key, held in _HELD_AXES:
            binds.append(Keybind(f"{name}_press", key, KeyAction.PRESS, callback=self.hold, args=(held,)))
            binds.append(Keybind(f"{name}_release", key, KeyAction.RELEASE, callback=self.release, args=(held,)))
        binds.append(Keybind("reset", Key.R, KeyAction.RELEASE, callback=self.request_reset))
        binds.append(Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=self.stop))
        return binds

    def command(self, limits):
        """Return vx, vy and yaw rate inside `limits`.

        Each axis is in [-1, 1] before scaling: the positive half maps onto the upper limit and the negative half onto
        the lower one, so a centered stick commands zero. A stick outside the deadzone replaces that axis's key.
        """
        vx = (1.0 if "w" in self.held_keys else 0.0) - (1.0 if "s" in self.held_keys else 0.0)
        vy = (1.0 if "q" in self.held_keys else 0.0) - (1.0 if "e" in self.held_keys else 0.0)
        yaw = (1.0 if "a" in self.held_keys else 0.0) - (1.0 if "d" in self.held_keys else 0.0)
        axes = [vx, vy, yaw]
        if self.pad is not None:
            pygame.event.get()
            # SDL reports X to the right and Y downward. vx is forward, vy is left, and yaw is left.
            stick = (
                -self.pad.get_axis(LEFT_STICK_Y),
                -self.pad.get_axis(LEFT_STICK_X),
                -self.pad.get_axis(RIGHT_STICK_X),
            )
            for i, value in enumerate(stick):
                if abs(value) >= STICK_DEADZONE:
                    axes[i] = value
        axes = torch.tensor(axes, dtype=gs.tc_float, device=gs.device)
        lower, upper = limits
        span = torch.where(axes >= 0.0, upper, -lower)
        return axes * span


def chase_base(env, offset, world_up):
    """Put the viewer behind the base along its heading and aim it at the base, with world up as screen up."""
    pos = env.robot.get_pos()
    quat = env.robot.get_quat()
    if pos.ndim > 1:
        pos = pos[0]
        quat = quat[0]
    # Base pitch and roll would swing the camera off the heading and tilt the horizon, so only the yaw is kept.
    euler = quat_to_xyz(quat, rpy=True).clone()
    euler[:2] = 0.0
    eye = pos + transform_by_quat(offset, xyz_to_quat(euler, rpy=True))
    env.scene.viewer.set_camera_pose(pos=eye, lookat=pos, up=world_up)


def run_teleop(args):
    gs.init(backend=gs.cpu)
    teleop = TeleopInput()
    teleop.open_pad()
    env, env_cfg, _, _, train_cfg = make_eval_env(
        args.exp_name,
        num_envs=1,
        show_viewer=True,
        terrain=args.terrain,
        terrain_level=args.terrain_level,
        pin_easiest=True,
    )
    env.scene.viewer.register_keybinds(*teleop.keybinds(), overwrite=True)
    policy, ckpt_path = load_policy(env, env_cfg, train_cfg, args.exp_name, args.ckpt)
    pad_name = "none" if teleop.pad is None else teleop.pad.get_name()
    print(
        f"ckpt={ckpt_path} pad={pad_name} deadzone={STICK_DEADZONE} "
        f"vx=-axis{LEFT_STICK_Y} vy=-axis{LEFT_STICK_X} yaw=-axis{RIGHT_STICK_X} "
        "keys W/S vx, Q/E vy, A/D yaw, R reset, Esc quit"
    )
    offset = torch.tensor(CHASE_OFFSET, dtype=gs.tc_float, device=gs.device)
    world_up = torch.tensor((0.0, 0.0, 1.0), dtype=gs.tc_float, device=gs.device)
    env.reset()
    obs = env.set_commands(teleop.command(env.commands_limits))
    with torch.no_grad():
        while teleop.is_running and env.scene.viewer.is_alive():
            actions = policy(obs)
            env.step(actions)
            if teleop.is_reset_requested:
                env.reset()
                teleop.is_reset_requested = False
            obs = env.set_commands(teleop.command(env.commands_limits))
            chase_base(env, offset, world_up)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    parser.add_argument("--terrain", choices=("config", "plane", "curriculum"), default="config")
    parser.add_argument(
        "--terrain-level",
        type=int,
        default=None,
        help=(
            "Curriculum row to spawn on, 0-based, 0 easiest. The mesh keeps rows 0 through this index at the saved "
            "curriculum slopes, and the robot spawns on this row. Requires curriculum terrain."
        ),
    )
    run_teleop(parser.parse_args())


if __name__ == "__main__":
    main()

"""
# Teleop on logs/go2-walking/model_100.pt. Left stick is vx/vy, right stick x is yaw, each scaled to the saved
# command range (and to the forward cap the checkpoint reached). W/S, Q/E and A/D command the same axes when the pad
# is absent or that stick is inside the deadzone. R respawns the robot without redrawing the domain.
uv run --extra examples python examples/locomotion/go2_teleop.py -e go2-walking --ckpt 100
"""
