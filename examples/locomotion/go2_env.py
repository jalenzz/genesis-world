import math
from typing import NamedTuple

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
from genesis.ext.isaacgym.terrain_utils import (
    SubTerrain,
    discrete_obstacles_terrain,
    pyramid_sloped_terrain,
    pyramid_stairs_terrain,
    random_uniform_terrain,
)
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

from cenet import CENet, HISTORY_STEPS, LATENT_DIM, adaboot_uses_truth as adaboot_batch_uses_truth


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


# Columns of the curriculum grid, in order. Rows are the difficulty levels.
TERRAIN_KINDS = ("smooth_slope", "rough_slope", "discrete_obstacles", "stairs")


class DomainRandomization(NamedTuple):
    friction: tuple[float, float]
    mass_scale: tuple[float, float]
    com_offset: float
    inertia_scale: tuple[float, float]
    gain_scale: tuple[float, float]
    force_scale: tuple[float, float]
    gravity_scale: tuple[float, float]
    push_xy: float
    push_z: float


DEFAULT_RANDOMIZATION = DomainRandomization(
    friction=(0.5, 1.5),
    mass_scale=(0.75, 1.25),
    com_offset=0.05,
    inertia_scale=(0.75, 1.25),
    gain_scale=(0.8, 1.2),
    force_scale=(0.8, 1.2),
    gravity_scale=(0.9, 1.1),
    push_xy=20.0,
    push_z=5.0,
)

FOOT_LINK_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


def locomotion_cfgs():
    """Return the flat-ground Go2 walking config used by the training entry."""
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def curriculum_reward_scales():
    """Return reward weights for velocity tracking on the terrain curriculum.

    Tracking stays the dominant term. Joint acceleration and motor power are large signals, so their weights stay
    small. Height is measured above the terrain, and a heavy height penalty fights the slope, so that weight is light.
    """
    return {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -0.2,
        "dof_acc": -2.5e-7,
        "joint_power": -2.0e-5,
        "base_height": -1.0,
        "feet_air_time": 1.0,
        "feet_clearance": -0.01,
        "action_rate": -0.01,
        "action_smoothness": -0.01,
        "power_distribution": -1.0e-5,
    }


def estimator_obs_groups():
    """Return the actor and critic observation groups used with the context estimator.

    The actor reads the current proprioception, the estimated base linear velocity and the latent. The critic reads the
    proprioception together with the measured base linear velocity and the disturbance.
    """
    return {
        "actor": ["proprio", "est_vel", "latent"],
        "critic": ["proprio", "lin_vel", "disturbance"],
    }


def policy_obs_groups(env_cfg):
    """Return the actor and critic groups saved with a walking config.

    The flat walker keeps the single proprioceptive group. The curriculum config selects the estimator groups.
    """
    if env_cfg.get("estimator") == "cenet":
        return estimator_obs_groups()
    return {"actor": ["policy"], "critic": ["policy"]}


def apply_curriculum(env_cfg, reward_cfg):
    """Point a walking config at the terrain curriculum and its reward weights.

    The flat-ground entry leaves these fields unset. A height field produces more foot contacts than a plane, so the
    collision budget is raised with the terrain. The curriculum actor reads an estimated base velocity, so the context
    estimator is enabled here and stays off for the flat walker.
    """
    env_cfg["terrain"] = "curriculum"
    env_cfg["domain_randomization"] = True
    env_cfg["max_collision_pairs"] = 256
    env_cfg["estimator"] = "cenet"
    reward_cfg["reward_scales"] = curriculum_reward_scales()


def build_curriculum_height_field(n_levels, subterrain_size, horizontal_scale, vertical_scale, max_slope_deg):
    """Build the height field of the tracking curriculum.

    Each row is one difficulty, with slope angle `row / (n_levels - 1) * max_slope_deg`. The columns are a smooth
    pyramid slope, the same slope plus height noise, discrete obstacles, and pyramid stairs. The value at each grid
    point is the discrete height the terrain morph scales by `vertical_scale`.
    """
    n_types = len(TERRAIN_KINDS)
    subterrain_rows = int(subterrain_size[0] / horizontal_scale + gs.EPS) + 1
    subterrain_cols = int(subterrain_size[1] / horizontal_scale + gs.EPS) + 1
    height_field = np.zeros(
        (
            n_levels * (subterrain_rows - 1) + 1,
            n_types * (subterrain_cols - 1) + 1,
        ),
        dtype=gs.np_float,
    )
    platform = min(2.0, 0.5 * min(subterrain_size))
    step_width = horizontal_scale * 2.0
    min_obstacle = horizontal_scale * 4.0
    max_obstacle = max(min_obstacle, min(subterrain_size[0] * 0.45, horizontal_scale * 12.0))
    saved_state = np.random.get_state()
    try:
        for level in range(n_levels):
            difficulty = level / (n_levels - 1)
            slope = math.tan(math.radians(max_slope_deg * difficulty))
            for kind in range(n_types):
                np.random.seed(level * n_types + kind)
                tile = SubTerrain(
                    width=subterrain_rows,
                    length=subterrain_cols,
                    vertical_scale=vertical_scale,
                    horizontal_scale=horizontal_scale,
                )
                if kind == 0:
                    pyramid_sloped_terrain(tile, slope=slope, platform_size=platform)
                elif kind == 1:
                    pyramid_sloped_terrain(tile, slope=slope, platform_size=platform)
                    noise = 0.08 * difficulty
                    if noise > vertical_scale:
                        random_uniform_terrain(
                            tile,
                            min_height=-noise,
                            max_height=noise,
                            step=vertical_scale,
                            downsampled_scale=horizontal_scale * 2.0,
                        )
                elif kind == 2:
                    discrete_obstacles_terrain(
                        tile,
                        max_height=0.02 + 0.18 * difficulty,
                        min_size=min_obstacle,
                        max_size=max_obstacle,
                        num_rects=8,
                        platform_size=platform,
                    )
                else:
                    pyramid_stairs_terrain(
                        tile,
                        step_width=step_width,
                        step_height=slope * step_width,
                        platform_size=platform,
                    )
                row0 = level * (subterrain_rows - 1)
                col0 = kind * (subterrain_cols - 1)
                patch = height_field[row0 : row0 + subterrain_rows, col0 : col0 + subterrain_cols]
                patch[:] = np.maximum(patch, tile.height_field_raw)
    finally:
        np.random.set_state(saved_state)
    return height_field


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs: int = num_envs
        self._batch_n = num_envs if num_envs > 0 else 1
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]
        self._terrain_kind = env_cfg.get("terrain", "plane")
        self._has_domain_randomization = bool(env_cfg.get("domain_randomization", False))
        randomization = env_cfg.get("randomization")
        self._randomization = (
            DEFAULT_RANDOMIZATION if randomization is None else DEFAULT_RANDOMIZATION._replace(**randomization)
        )
        self._curriculum_up = env_cfg.get("curriculum_up", 0.8)
        self._curriculum_down = env_cfg.get("curriculum_down", 0.25)
        self._n_levels = env_cfg.get("terrain_levels", 10)
        self._subterrain_size = env_cfg.get("terrain_subterrain_size", (8.0, 8.0))
        self._horizontal_scale = env_cfg.get("terrain_horizontal_scale", 0.25)
        self._vertical_scale = env_cfg.get("terrain_vertical_scale", 0.005)
        self._max_slope_deg = env_cfg.get("terrain_max_slope_deg", 22.0)

        if self._terrain_kind == "curriculum":
            span_x = self._n_levels * self._subterrain_size[0]
            span_y = len(TERRAIN_KINDS) * self._subterrain_size[1]
            camera_pos = (span_x * 0.5, -span_y, max(span_x * 0.35, 2.5))
            camera_lookat = (span_x * 0.5, span_y * 0.5, 0.0)
        else:
            camera_pos = (2.0, 0.0, 2.5)
            camera_lookat = (0.0, 0.0, 0.5)

        # A flat ground fits in 20 pairs. The height field's box covers the whole robot, so every collision geom is a
        # pair with it and 20 overflows. A pose reset clears that overflow, which hides it.
        max_collision_pairs = env_cfg.get("max_collision_pairs", 20)
        if self._terrain_kind == "curriculum" and max_collision_pairs == 20:
            max_collision_pairs = 256

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=max_collision_pairs,
                tolerance=1e-5,
                # Per-environment mass, center of mass and actuator gains need their own copy of those parameters.
                batch_links_info=self._has_domain_randomization,
                batch_dofs_info=self._has_domain_randomization,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        self.terrain = None
        if self._terrain_kind == "curriculum":
            self.terrain = self.scene.add_entity(
                gs.morphs.Terrain(
                    height_field=build_curriculum_height_field(
                        self._n_levels,
                        self._subterrain_size,
                        self._horizontal_scale,
                        self._vertical_scale,
                        self._max_slope_deg,
                    ),
                    horizontal_scale=self._horizontal_scale,
                    vertical_scale=self._vertical_scale,
                ),
            )
        else:
            self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/plane/plane.urdf",
                    fixed=True,
                )
            )

        # The foot is a fixed child of the calf, so the loader folds it into the calf. Clearance and foot contact
        # read that sphere as its own link, which means naming it here. The flat-ground rewards never read it.
        needs_feet = "feet_air_time" in self.reward_scales or "feet_clearance" in self.reward_scales
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
                links_to_keep=FOOT_LINK_NAMES if needs_feet else (),
            ),
        )

        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        self._feet_idx = [self.robot.get_link(name).idx_local for name in FOOT_LINK_NAMES] if needs_feet else []
        self._base_link = self.robot.get_link("base")

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self._nominal_gravity = torch.tensor([0.0, 0.0, -9.81], dtype=gs.tc_float, device=gs.device)

        # Initial state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)
        self._cell_origins = None
        if self.terrain is not None:
            self._cell_origins = self._curriculum_origins()

        if self._has_domain_randomization:
            self._nominal_mass = self._shared_nominal(self.robot.get_links_mass())
            self._nominal_com = self._shared_nominal(self.robot.get_links_COM())
            self._nominal_inertia = self._shared_nominal(self.robot.get_links_inertia())
            force_lower, force_upper = self.robot.get_dofs_force_range(self.motors_dof_idx)
            self._nominal_force_lower = self._shared_nominal(force_lower)
            self._nominal_force_upper = self._shared_nominal(force_upper)

        # initialize buffers
        self.base_lin_vel = torch.empty((self._batch_n, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self._batch_n, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self._batch_n, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self._batch_n,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self._batch_n,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((self._batch_n,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self._batch_n, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((self._batch_n, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self._batch_n, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self._batch_n, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self._batch_n, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = dict()  # extra information for logging
        self.terrain_levels = torch.zeros((self._batch_n,), dtype=gs.tc_int, device=gs.device)
        self.terrain_types = torch.arange(self._batch_n, device=gs.device) % len(TERRAIN_KINDS)
        self.episode_tracking_lin = torch.zeros((self._batch_n,), dtype=gs.tc_float, device=gs.device)
        self._feet_air_time = torch.zeros((self._batch_n, len(FOOT_LINK_NAMES)), dtype=gs.tc_float, device=gs.device)
        self._push_force = torch.zeros((self._batch_n, 3), dtype=gs.tc_float, device=gs.device)
        self._cenet = None
        self._cenet_optimizer = None
        self._finished_returns = []
        self._is_ground_truth = False
        self._is_latent_sampled = False
        self._published_observations = None
        self._proprio_dim = 9 + self.num_actions * 3
        if env_cfg.get("estimator") == "cenet":
            self._cenet = CENet(self._proprio_dim).to(device=gs.device, dtype=gs.tc_float)
            self._cenet_optimizer = torch.optim.Adam(self._cenet.parameters(), lr=1.0e-3)
            self._history = torch.zeros(
                (self._batch_n, HISTORY_STEPS, self._proprio_dim), dtype=gs.tc_float, device=gs.device
            )
            self._episode_return = torch.zeros((self._batch_n,), dtype=gs.tc_float, device=gs.device)
        self._feet_pos = None
        self._feet_vel = None
        self._feet_contact = None
        self._dof_acc = None
        self._motor_power = None

        reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "orientation": self._reward_orientation,
            "dof_acc": self._reward_dof_acc,
            "joint_power": self._reward_joint_power,
            "base_height": self._reward_base_height,
            "feet_air_time": self._reward_feet_air_time,
            "feet_clearance": self._reward_feet_clearance,
            "action_rate": self._reward_action_rate,
            "action_smoothness": self._reward_action_smoothness,
            "power_distribution": self._reward_power_distribution,
            "similar_to_default": self._reward_similar_to_default,
        }
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = reward_fns[name]
            self.episode_sums[name] = torch.zeros((self._batch_n,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    def _shared_nominal(self, value):
        # Batched link and dof info repeats the authored value once per environment. The randomizer scales that value.
        if self.num_envs > 0 and value.ndim >= 1 and value.shape[0] == self.num_envs:
            return value[0]
        return value

    def _batch(self, tensor):
        if self.num_envs > 0 or tensor.ndim == 0:
            return tensor
        return tensor.unsqueeze(0)

    def _curriculum_origins(self):
        n_types = len(TERRAIN_KINDS)
        xs = (torch.arange(self._n_levels, dtype=gs.tc_float, device=gs.device) + 0.5) * self._subterrain_size[0]
        ys = (torch.arange(n_types, dtype=gs.tc_float, device=gs.device) + 0.5) * self._subterrain_size[1]
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        xy = torch.stack((grid_x, grid_y), dim=-1).reshape(self._n_levels * n_types, 2)
        heights = self.terrain.get_terrain_height(xy)
        if self.num_envs > 0:
            heights = heights[0]
        heights = heights.reshape(self._n_levels, n_types)
        return torch.stack((grid_x, grid_y, heights + self.init_base_pos[2]), dim=-1)

    def _terrain_z(self, xy):
        if self.terrain is None:
            return torch.zeros(xy.shape[:-1], dtype=gs.tc_float, device=gs.device)
        if self.num_envs == 0:
            return self.terrain.get_terrain_height(xy[0]).unsqueeze(0)
        if xy.ndim == 2:
            return self.terrain.get_terrain_height(xy[:, None, :])[:, 0]
        return self.terrain.get_terrain_height(xy)

    def _reset_mask(self, envs_idx):
        if envs_idx is None:
            return torch.ones((self._batch_n,), dtype=torch.bool, device=gs.device)
        return envs_idx

    def _sample_interval(self, low, high, shape):
        return (high - low) * torch.rand(shape, dtype=gs.tc_float, device=gs.device) + low

    def _update_terrain_curriculum(self, envs_idx):
        if self.terrain is None:
            return
        # The tracking reward saturates at 1. The level rises above the upper fraction and falls below the lower one.
        finished = self._reset_mask(envs_idx) & (self.episode_length_buf > 0)
        mean_tracking = self.episode_tracking_lin / self.episode_length_buf.clamp(min=1)
        delta = (mean_tracking > self._curriculum_up).to(dtype=gs.tc_int)
        delta = delta - (mean_tracking < self._curriculum_down).to(dtype=gs.tc_int)
        self.terrain_levels += torch.where(finished, delta, torch.zeros_like(delta))
        self.terrain_levels.clamp_(0, self._n_levels - 1)

    def _randomize_domain(self, envs_idx):
        if not self._has_domain_randomization:
            return
        # Link and dof setters take environment indices, while the reset decision is a mask, so the indices are read
        # back on the steps an episode actually ends.
        if envs_idx is None:
            env_ids = None
            count = self._batch_n
        else:
            if self.num_envs == 0:
                if not bool(envs_idx[0]):
                    return
                env_ids = None
                count = 1
            else:
                if not bool(envs_idx.any()):
                    return
                env_ids = envs_idx.nonzero(as_tuple=True)[0]
                count = len(env_ids)
        n_links = self.robot.n_links
        ranges = self._randomization
        link_shape = (n_links,) if self.num_envs == 0 else (count, n_links)
        dof_shape = (self.num_actions,) if self.num_envs == 0 else (count, self.num_actions)
        friction = self._sample_interval(*ranges.friction, link_shape)
        self.robot.set_friction_ratio(friction, envs_idx=env_ids)
        mass_scale = self._sample_interval(*ranges.mass_scale, link_shape)
        self.robot.set_links_mass(self._nominal_mass * mass_scale, envs_idx=env_ids)
        if self.num_envs == 0:
            com_offset = (torch.rand((n_links, 3), dtype=gs.tc_float, device=gs.device) * 2.0 - 1.0) * ranges.com_offset
            inertia_scale = self._sample_interval(*ranges.inertia_scale, (n_links, 1, 1))
            gravity_scale = self._sample_interval(*ranges.gravity_scale, ())
        else:
            com_offset = (
                torch.rand((count, n_links, 3), dtype=gs.tc_float, device=gs.device) * 2.0 - 1.0
            ) * ranges.com_offset
            inertia_scale = self._sample_interval(*ranges.inertia_scale, (count, n_links, 1, 1))
            gravity_scale = self._sample_interval(*ranges.gravity_scale, (count, 1))
        self.robot.set_links_COM(self._nominal_com + com_offset, envs_idx=env_ids)
        self.robot.set_links_inertia(self._nominal_inertia * inertia_scale, envs_idx=env_ids)
        gain_scale = self._sample_interval(*ranges.gain_scale, dof_shape)
        self.robot.set_dofs_kp(self.env_cfg["kp"] * gain_scale, self.motors_dof_idx, env_ids)
        self.robot.set_dofs_kv(self.env_cfg["kd"] * gain_scale, self.motors_dof_idx, env_ids)
        force_scale = self._sample_interval(*ranges.force_scale, dof_shape)
        self.robot.set_dofs_force_range(
            self._nominal_force_lower * force_scale,
            self._nominal_force_upper * force_scale,
            self.motors_dof_idx,
            env_ids,
        )
        self.scene.sim.set_gravity(self._nominal_gravity * gravity_scale, envs_idx=env_ids)
        push = torch.rand((count, 3), dtype=gs.tc_float, device=gs.device) * 2.0 - 1.0
        push[:, 0] *= ranges.push_xy
        push[:, 1] *= ranges.push_xy
        push[:, 2] *= ranges.push_z
        if env_ids is None:
            self._push_force.copy_(push)
        else:
            self._push_force[env_ids] = push

    def _apply_push(self):
        if not self._has_domain_randomization:
            return
        force = self._push_force if self.num_envs > 0 else self._push_force[0]
        self._base_link.apply_external_wrench(force=force)

    def _spawn_qpos(self):
        if self.terrain is None:
            return self.init_qpos
        qpos = self.init_qpos.unsqueeze(0).repeat(self._batch_n, 1)
        qpos[:, :3] = self._cell_origins[self.terrain_levels, self.terrain_types]
        if self.num_envs == 0:
            return qpos[0]
        return qpos

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self._batch_n,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def _read_feet_and_motors(self):
        needs_feet = "feet_air_time" in self.reward_functions or "feet_clearance" in self.reward_functions
        if needs_feet:
            self._feet_pos = self._batch(self.robot.get_links_pos(self._feet_idx))
            self._feet_vel = self._batch(self.robot.get_links_vel(self._feet_idx))
            contact_force = self._batch(self.robot.get_links_net_contact_force())
            # A foot is on the ground once the contact force exceeds a newton, above numerical noise.
            self._feet_contact = torch.linalg.norm(contact_force[:, self._feet_idx], dim=-1) > 1.0
        if "dof_acc" in self.reward_functions:
            self._dof_acc = self._batch(self.robot.get_dofs_acc(self.motors_dof_idx))
        if "joint_power" in self.reward_functions or "power_distribution" in self.reward_functions:
            torque = self._batch(self.robot.get_dofs_control_force(self.motors_dof_idx))
            self._motor_power = torch.abs(torque) * torch.abs(self.dof_vel)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        ordered = target_dof_pos[:, self.actions_dof_idx]
        self._apply_push()
        self.robot.control_dofs_position(ordered if self.num_envs > 0 else ordered[0], slice(6, 18))
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos = self._batch(self.robot.get_pos())
        self.base_quat = self._batch(self.robot.get_quat())
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self._batch(self.robot.get_vel()), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self._batch(self.robot.get_ang()), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self._batch(self.robot.get_dofs_position(self.motors_dof_idx))
        self.dof_vel = self._batch(self.robot.get_dofs_velocity(self.motors_dof_idx))
        self._read_feet_and_motors()
        if self.terrain is not None:
            self.episode_tracking_lin += self._reward_tracking_lin_vel()

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self._cenet is not None:
            self._episode_return += self.rew_buf

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # A pose reset clears errno. Checking it here is what makes an overflow halt instead of respawning forever.
        error_mask = self.scene.rigid_solver.get_error_envs_mask()
        if error_mask.ndim == 0:
            error_mask = error_mask.reshape(1)
        if bool(error_mask.any()):
            self.scene.rigid_solver.check_errno()

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_last_actions.copy_(self.last_actions)
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        if self._cenet is None:
            self._published_observations = TensorDict({"policy": self.obs_buf}, batch_size=[self._batch_n])
            return self._published_observations
        velocity, latent = self._velocity_and_latent()
        self._published_observations = TensorDict(
            {
                "proprio": self.obs_buf,
                "est_vel": velocity,
                "latent": latent,
                "lin_vel": self.base_lin_vel,
                "disturbance": self._push_force,
                "history": self._history,
            },
            batch_size=[self._batch_n],
        )
        return self._published_observations

    def _velocity_and_latent(self):
        """Return the base velocity and latent the actor reads on this step."""
        if self._is_ground_truth:
            # The latent has no measured value. Zero is the mean of its standard normal prior.
            latent = torch.zeros((self._batch_n, LATENT_DIM), dtype=gs.tc_float, device=gs.device)
            return self.base_lin_vel, latent
        # The history can be an inference tensor from the rollout. Autograd has to save a normal tensor.
        history = self._history.detach().clone()
        belief = self._cenet.encode(history)
        latent = self._cenet.reparameterize(belief) if self._is_latent_sampled else belief.mean
        return belief.velocity.detach(), latent.detach()

    def set_estimator_ground_truth(self, use_ground_truth):
        """Choose measured velocity or the estimator for the actor, including the observation already published."""
        self._is_ground_truth = use_ground_truth
        if self._cenet is None or self._published_observations is None:
            return
        velocity, latent = self._velocity_and_latent()
        self._published_observations["est_vel"] = velocity
        self._published_observations["latent"] = latent

    def set_latent_sampling(self, sample_latent):
        """Sample the latent during a rollout, or publish its mean."""
        self._is_latent_sampled = sample_latent

    def adaboot_uses_truth(self):
        """Return whether the next batch should read ground truth, from the returns finished in this batch."""
        if self._cenet is None or len(self._finished_returns) == 0:
            return True
        returns = torch.cat(self._finished_returns)
        self._finished_returns.clear()
        return adaboot_batch_uses_truth(returns, torch.rand((), dtype=gs.tc_float, device=gs.device))

    def update_estimator(self, storage):
        """Fit the context estimator on one rollout and return its loss.

        The reconstruction target is the next stored proprioception. A finished step is left out, because the following
        observation belongs to the next episode. A rollout with no continuing step returns zero.
        """
        if self._cenet is None:
            return 0.0
        # Clones leave inference tensors from the rollout. The loss gradient stays on the estimator parameters.
        history = storage.observations["history"][:-1].detach().clone()
        next_proprio = storage.observations["proprio"][1:].detach().clone()
        velocity = storage.observations["lin_vel"][:-1].detach().clone()
        alive = storage.dones[:-1].squeeze(-1) == 0
        if not bool(alive.any()):
            return 0.0
        self._cenet.train()
        loss = self._cenet.loss(history[alive], velocity[alive], next_proprio[alive])
        self._cenet_optimizer.zero_grad()
        loss.backward()
        self._cenet_optimizer.step()
        return float(loss.detach())

    def cenet_state_dict(self):
        """Return the estimator weights and optimizer state."""
        return {"model": self._cenet.state_dict(), "optimizer": self._cenet_optimizer.state_dict()}

    def load_cenet_state_dict(self, state):
        """Load estimator weights and optimizer state saved beside the policy."""
        self._cenet.load_state_dict(state["model"])
        self._cenet_optimizer.load_state_dict(state["optimizer"])

    def _reset_idx(self, envs_idx=None):
        self._update_terrain_curriculum(envs_idx)
        self._randomize_domain(envs_idx)
        write_pose = self.num_envs > 0 or envs_idx is None or bool(envs_idx[0])
        if write_pose:
            pose_envs = None if self.num_envs == 0 else envs_idx
            self.robot.set_qpos(self._spawn_qpos(), envs_idx=pose_envs, zero_velocity=True, skip_forward=True)

        # reset buffers
        if self._cenet is not None:
            finished_mask = self._reset_mask(envs_idx) & (self.episode_length_buf > 0)
            if bool(finished_mask.any()):
                self._finished_returns.append(self._episode_return[finished_mask].detach().clone())
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_last_actions.zero_()
            self.episode_length_buf.zero_()
            self.episode_tracking_lin.zero_()
            self._feet_air_time.zero_()
            self.reset_buf.fill_(True)
            if self._cenet is not None:
                self._episode_return.zero_()
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.episode_tracking_lin.masked_fill_(envs_idx, 0.0)
            self._feet_air_time.masked_fill_(envs_idx[:, None], 0.0)
            self.reset_buf.masked_fill_(envs_idx, True)
            if self._cenet is not None:
                self._episode_return.masked_fill_(envs_idx, 0.0)

        if self.terrain is not None:
            origin = self._cell_origins[self.terrain_levels, self.terrain_types]
            if envs_idx is None:
                self.base_pos.copy_(origin)
            else:
                torch.where(envs_idx[:, None], origin, self.base_pos, out=self.base_pos)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self._batch_n
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )
        if self._cenet is None:
            return
        assert self.obs_buf.shape[-1] == self._proprio_dim
        rolled = torch.roll(self._history, shifts=-1, dims=1)
        rolled[:, -1] = self.obs_buf
        filled = self.obs_buf[:, None, :].expand_as(self._history)
        # Filling the whole window drops the previous episode. A roll would leave it in the encoder's context.
        # The reset mask is allocated as gs.tc_bool, an integer on Metal, and where requires a boolean condition.
        is_resetting = self.reset_buf[:, None, None] != 0
        updated = torch.where(is_resetting, filled, rolled)
        # Rollout steps run in inference mode and mark this result. The clone below is a normal tensor.
        with torch.inference_mode(False):
            self._history = updated.detach().clone()

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # Reward functions.
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize roll and pitch angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize torso tilt away from upright
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_dof_acc(self):
        # Penalize joint acceleration
        return torch.sum(torch.square(self._dof_acc), dim=1)

    def _reward_joint_power(self):
        # Penalize mechanical power at the motors
        return torch.sum(self._motor_power, dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        # Penalize the second difference of the action, which is a jerk in the position target
        return torch.sum(torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target, measured above the ground under the base
        height = self.base_pos[:, 2] - self._terrain_z(self.base_pos[:, :2])
        return torch.square(height - self.reward_cfg["base_height_target"])

    def _reward_feet_air_time(self):
        # Pay the swing duration once, on the landing step. A foot that stays down has a cleared timer.
        contact = self._feet_contact
        first_contact = contact & (self._feet_air_time > 0)
        self._feet_air_time += self.dt
        rew = torch.sum((self._feet_air_time - 0.5) * first_contact, dim=1)
        self._feet_air_time *= ~contact
        moving = torch.linalg.norm(self.commands[:, :2], dim=1) > 0.1
        return torch.where(moving, rew, torch.zeros_like(rew))

    def _reward_feet_clearance(self):
        # Penalize swing feet whose height above the terrain misses the clearance target. Stance speed is ~0.
        clearance = self._feet_pos[..., 2] - self._terrain_z(self._feet_pos[..., :2])
        speed = torch.linalg.norm(self._feet_vel[..., :2], dim=-1)
        return torch.sum(torch.square(clearance - self.reward_cfg["feet_height_target"]) * speed, dim=1)

    def _reward_power_distribution(self):
        # Penalize uneven mechanical power across the motors
        return torch.var(self._motor_power, dim=1)
