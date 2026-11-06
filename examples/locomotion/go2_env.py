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

from cenet import CENet, HISTORY_STEPS, adaboot_probability


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


# Columns of the curriculum grid, in order. Rows are the difficulty levels.
TERRAIN_KINDS = ("smooth_slope", "rough_slope", "discrete_obstacles", "stairs")
# Episode share of each column: smooth slope, rough slope, discrete obstacles, stairs.
TERRAIN_KIND_FRACTIONS = (0.48, 0.20, 0.12, 0.20)
# Forward caps, in order. Sampling uses [-1, cap]. A cap rises after 100 iterations when the mean of the last 50
# per-step linear tracking rewards is at least 0.55.
SPEED_VX_CAPS = (1.0, 1.25, 1.5, 1.75, 2.0)
SPEED_PROMOTE_ITERS = 100
SPEED_TRACK_WINDOW = 50
SPEED_TRACK_BAR = 0.55
# Consecutive episodes above the tracking bar before an obstacle or stair row may rise.
SLOW_PROMOTE_EPISODES = 8
# A fresh run spawns uniformly on rows 0 through this one, and the curriculum moves each environment from there.
MAX_INIT_TERRAIN_LEVEL = 5
# A contact force above this on the base link counts as a fall.
BASE_CONTACT_FORCE = 1.0
# A joint or a base faster than this overflows the next integration. The step puts that environment back on its spawn.
MAX_JOINT_SPEED = 100.0
MAX_BASE_LINEAR_SPEED = 20.0
MAX_BASE_ANGULAR_SPEED = 100.0
# Height samples of the critic, on a grid in the base heading frame (forward x, left y), in meters.
HEIGHT_SCAN_X = tuple(0.1 * i for i in range(-8, 9))
HEIGHT_SCAN_Y = tuple(0.1 * i for i in range(-5, 6))


class DomainRandomization(NamedTuple):
    friction: tuple[float, float]
    mass_scale: tuple[float, float]
    # Largest shift of the base center of mass along each axis, in meters.
    com_offset: float
    inertia_scale: tuple[float, float]
    # Range of the proportional and of the derivative gain factor, each drawn independently.
    gain_scale: tuple[float, float]
    force_scale: tuple[float, float]
    gravity_scale: tuple[float, float]
    push_xy: float
    push_z: float
    # Added mass on the base, in kilograms. (0, 0) leaves the mass scale as the only mass change.
    payload: tuple[float, float] = (0.0, 0.0)
    # Extra delay inside the control period, in seconds. (0, 0) keeps the one-step position target.
    system_delay: tuple[float, float] = (0.0, 0.0)
    # Absolute friction coefficient. None keeps `friction` as a ratio of the authored coefficient.
    friction_coefficient: tuple[float, float] | None = None


class CurriculumState(NamedTuple):
    """Speed and terrain curriculum progress saved with a checkpoint."""

    speed_level: int | None
    speed_iters_at_cap: int
    speed_tracking_means: list[float]
    terrain_levels: torch.Tensor
    promote_streak: torch.Tensor


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

POLICY_MLP = "mlp"
POLICY_DREAMWAQ = "dreamwaq"


def is_dreamwaq_policy(env_cfg):
    """Return whether the env config selects the DreamWaQ actor and context estimator.

    The ``policy`` key names the actor, and a config without it selects the flat MLP walker.
    """
    return env_cfg.get("policy", POLICY_MLP) == POLICY_DREAMWAQ


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
    """Return the per-step reward weights of the terrain curriculum.

    Each weight multiplies its term once per control step. Linear and yaw tracking are exponentials of the squared
    velocity error. The remaining terms penalize vertical and roll-pitch velocity, body tilt, joint acceleration,
    motor power, body height, foot clearance, action changes, and uneven motor power.
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
        "feet_clearance": -0.01,
        "action_rate": -0.01,
        "action_smoothness": -0.01,
        "power_distribution": -1.0e-5,
    }


def dreamwaq_obs_groups():
    """Return the actor and critic observation groups for the DreamWaQ policy.

    The actor reads the current proprioception, the base linear velocity and the context vector. The critic reads the
    privileged state: the proprioception, the measured base linear velocity, the disturbance force on the base and the
    height scan around the base.
    """
    return {
        "actor": ["proprio", "est_vel", "latent"],
        "critic": ["proprio", "lin_vel", "disturbance", "height_scan"],
    }


def policy_obs_groups(env_cfg):
    """Return the actor and critic groups saved with a walking config.

    The flat MLP walker keeps the single proprioceptive group. The DreamWaQ policy selects ``dreamwaq_obs_groups``.
    """
    if is_dreamwaq_policy(env_cfg):
        return dreamwaq_obs_groups()
    return {"actor": ["policy"], "critic": ["policy"]}


def raised_speed_cap(level, iters_at_cap, tracking_means):
    """Return the forward-cap index after one iteration, and the tracking mean when it rises.

    `level` selects `SPEED_VX_CAPS`. `iters_at_cap` counts iterations spent on that cap, including the one just
    finished. `tracking_means` holds that cap's per-step linear tracking reward, oldest first. The index rises by one
    after 100 iterations when the mean of the last 50 values is at least 0.55. The last index stays in place.
    """
    is_cap_open = level < len(SPEED_VX_CAPS) - 1
    is_window_ready = iters_at_cap >= SPEED_PROMOTE_ITERS and len(tracking_means) >= SPEED_TRACK_WINDOW
    if not is_cap_open or not is_window_ready:
        return level, None
    window = tracking_means[-SPEED_TRACK_WINDOW:]
    mean = sum(window) / SPEED_TRACK_WINDOW
    if mean + 1.0e-6 < SPEED_TRACK_BAR:
        return level, None
    return level + 1, mean


def format_speed_curriculum(vx_max, iteration, tracking):
    """Format the forward-cap line printed at the start of training and at each promotion."""
    text = f"{vx_max:.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    if tracking is None:
        tracking_text = "nan"
    else:
        tracking_text = f"{tracking:.4f}"
    return f"speed_curriculum vx_max={text} iter={iteration} tracking_lin_vel={tracking_text}"


def apply_curriculum(env_cfg, reward_cfg, command_cfg=None):
    """Point a walking config at the terrain curriculum, its rewards, and its domain randomization.

    The flat-ground entry leaves these fields unset. A fall is a contact on the base, so the roll and pitch limits of
    the flat walker are dropped: a base aligned with a 22 degree slope is upright. The control delay is the randomized
    system delay alone. Set ``env_cfg["policy"]`` separately when the DreamWaQ actor is required. When a command config
    is passed, forward speed starts in [-1, 1] and its upper end rises with tracking. Lateral speed is drawn from
    [-0.5, 0.5] and yaw from [-1, 1].
    """
    env_cfg["terrain"] = "curriculum"
    env_cfg["domain_randomization"] = True
    env_cfg["speed_curriculum"] = True
    env_cfg["simulate_action_latency"] = False
    del env_cfg["termination_if_roll_greater_than"]
    del env_cfg["termination_if_pitch_greater_than"]
    env_cfg["randomization"] = {
        "friction": (1.0, 1.0),
        "mass_scale": (1.0, 1.0),
        "com_offset": 0.05,
        "inertia_scale": (1.0, 1.0),
        "gain_scale": (0.9, 1.1),
        "force_scale": (0.9, 1.1),
        "gravity_scale": (1.0, 1.0),
        "payload": (-1.0, 2.0),
        "system_delay": (0.0, 0.015),
        "friction_coefficient": (0.2, 1.25),
    }
    reward_cfg["reward_scales"] = curriculum_reward_scales()
    if command_cfg is not None:
        command_cfg["lin_vel_x_range"] = [-1.0, SPEED_VX_CAPS[0]]
        command_cfg["lin_vel_y_range"] = [-0.5, 0.5]
        command_cfg["ang_vel_range"] = [-1.0, 1.0]


def build_curriculum_height_field(
    n_levels, subterrain_size, horizontal_scale, vertical_scale, max_slope_deg, difficulty_levels=None
):
    """Build the height field of the tracking curriculum.

    Each row is one difficulty. Smooth and rough slopes use angle `row / (difficulty_levels - 1) * max_slope_deg`.
    `difficulty_levels` defaults to `n_levels`, so the last built row is the full slope. Discrete obstacles and
    stairs use the square of that fraction. The columns are a smooth pyramid slope, the same slope plus height
    noise, discrete obstacles, and pyramid stairs. The value at each grid point is the discrete height the terrain
    morph scales by `vertical_scale`. A field of one row is flat.
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
    difficulty_count = n_levels if difficulty_levels is None else difficulty_levels
    difficulty_span = difficulty_count - 1
    saved_state = np.random.get_state()
    try:
        for level in range(n_levels):
            difficulty = 0.0 if difficulty_span == 0 else level / difficulty_span
            slope = math.tan(math.radians(max_slope_deg * difficulty))
            eased = difficulty * difficulty
            eased_slope = math.tan(math.radians(max_slope_deg * eased))
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
                        max_height=0.20 * eased,
                        min_size=min_obstacle,
                        max_size=max_obstacle,
                        num_rects=8,
                        platform_size=platform,
                    )
                else:
                    pyramid_stairs_terrain(
                        tile,
                        step_width=step_width,
                        step_height=eased_slope * step_width,
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
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, add_camera=False):
        self.num_envs: int = num_envs
        self.batch_size = num_envs if num_envs > 0 else 1
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        resampling_time_s = env_cfg["resampling_time_s"]
        self._resampling_steps = None if resampling_time_s is None else int(resampling_time_s / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self._policy = env_cfg.get("policy", POLICY_MLP)
        if self._policy not in (POLICY_MLP, POLICY_DREAMWAQ):
            raise gs.GenesisException(
                f"Unknown policy '{self._policy}'. Supported policies: '{POLICY_MLP}' and '{POLICY_DREAMWAQ}'."
            )

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]
        self.terrain_kind = env_cfg.get("terrain", "plane")
        self._has_domain_randomization = bool(env_cfg.get("domain_randomization", False))
        randomization = env_cfg.get("randomization")
        self._randomization = (
            DEFAULT_RANDOMIZATION if randomization is None else DEFAULT_RANDOMIZATION._replace(**randomization)
        )
        self._is_terrain_curriculum = env_cfg.get("terrain_curriculum", True)
        self._is_fall_terminal = env_cfg.get("terminate_on_fall", True)
        self._curriculum_up = env_cfg.get("curriculum_up", 0.8)
        self._curriculum_down = env_cfg.get("curriculum_down", 0.25)
        self.n_levels = env_cfg.get("terrain_levels", 10)
        # See difficulty_levels in build_curriculum_height_field. A shorter mesh can keep that slope.
        self._difficulty_levels = env_cfg.get("terrain_difficulty_levels", self.n_levels)
        self._subterrain_size = env_cfg.get("terrain_subterrain_size", (8.0, 8.0))
        self._horizontal_scale = env_cfg.get("terrain_horizontal_scale", 0.25)
        self._vertical_scale = env_cfg.get("terrain_vertical_scale", 0.005)
        self._max_slope_deg = env_cfg.get("terrain_max_slope_deg", 22.0)
        self._terrain_span_x = self.n_levels * self._subterrain_size[0]
        self._terrain_span_y = len(TERRAIN_KINDS) * self._subterrain_size[1]

        if self.terrain_kind == "curriculum":
            camera_pos = (self._terrain_span_x * 0.5, -self._terrain_span_y, max(self._terrain_span_x * 0.35, 2.5))
            camera_lookat = (self._terrain_span_x * 0.5, self._terrain_span_y * 0.5, 0.0)
        else:
            camera_pos = (2.0, 0.0, 2.5)
            camera_lookat = (0.0, 0.0, 0.5)

        # A flat ground fits in 20 pairs. The height field's box covers the whole robot, so every collision geom is a
        # pair with it and 20 overflows. An overflow halts the step, see the errno check in step.
        max_collision_pairs = env_cfg.get("max_collision_pairs")
        if max_collision_pairs is None:
            max_collision_pairs = 256 if self.terrain_kind == "curriculum" else 20

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
                rendered_envs_idx=env_cfg.get("rendered_envs_idx", [0]),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        self.terrain = None
        if self.terrain_kind == "curriculum":
            self.terrain = self.scene.add_entity(
                gs.morphs.Terrain(
                    height_field=build_curriculum_height_field(
                        self.n_levels,
                        self._subterrain_size,
                        self._horizontal_scale,
                        self._vertical_scale,
                        self._max_slope_deg,
                        difficulty_levels=self._difficulty_levels,
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
        needs_feet = "feet_clearance" in self.reward_scales
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
                links_to_keep=FOOT_LINK_NAMES if needs_feet else (),
            ),
        )

        self.camera = None
        if add_camera:
            self.camera = self.scene.add_camera(
                res=(640, 480),
                pos=(1.6, -1.6, 1.0),
                lookat=(0.0, 0.0, 0.3),
                fov=50,
                GUI=False,
                near=0.05,
                far=30.0,
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
        self._forward_axis = torch.tensor([1.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
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
        self.cell_origins = None
        if self.terrain is not None:
            self.cell_origins = self._curriculum_origins()

        if self._has_domain_randomization:
            self._nominal_mass = self._shared_nominal(self.robot.get_links_mass())
            self._nominal_com = self._shared_nominal(self.robot.get_links_COM())
            self._nominal_inertia = self._shared_nominal(self.robot.get_links_inertia())
            force_lower, force_upper = self.robot.get_dofs_force_range(self.motors_dof_idx)
            self._nominal_force_lower = self._shared_nominal(force_lower)
            self._nominal_force_upper = self._shared_nominal(force_upper)
            if self._randomization.friction_coefficient is not None:
                frictions = []
                for link in self.robot.links:
                    if link.geoms:
                        frictions.append(link.geoms[0].get_friction())
                    else:
                        frictions.append(torch.ones((), dtype=gs.tc_float, device=gs.device))
                self._nominal_friction = torch.stack(frictions)

        # initialize buffers
        self.base_lin_vel = torch.empty((self.batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.batch_size,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.batch_size,), dtype=gs.tc_bool, device=gs.device)
        self.is_fallen = torch.zeros((self.batch_size,), dtype=torch.bool, device=gs.device)
        self.episode_length_buf = torch.zeros((self.batch_size,), dtype=gs.tc_int, device=gs.device)
        # Steps actually simulated in the episode. A learner may overwrite episode_length_buf with a random start.
        self._episode_steps = torch.zeros((self.batch_size,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.batch_size, self.num_commands), dtype=gs.tc_float, device=gs.device)
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
        self._speed_level = 0 if env_cfg.get("speed_curriculum") else None
        self._speed_iters_at_cap = 0
        self._speed_tracking_means = []
        self._speed_track_sum = torch.zeros((), dtype=gs.tc_float, device=gs.device)
        self._speed_track_count = 0
        self.actions = torch.zeros((self.batch_size, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.batch_size, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = dict()  # extra information for logging
        kind_weights = torch.tensor(TERRAIN_KIND_FRACTIONS, dtype=gs.tc_float, device=gs.device)
        self.terrain_types = torch.multinomial(kind_weights, self.batch_size, replacement=True)
        max_init_level = min(MAX_INIT_TERRAIN_LEVEL, self.n_levels - 1)
        self.terrain_levels = torch.randint(0, max_init_level + 1, (self.batch_size,), device=gs.device)
        # A set row replaces the sampled row before the first reset.
        initial_level = env_cfg.get("initial_terrain_level")
        if initial_level is not None:
            self.terrain_levels.fill_(initial_level)
        self._promote_streak = torch.zeros((self.batch_size,), dtype=gs.tc_int, device=gs.device)
        self._obstacle_kind = TERRAIN_KINDS.index("discrete_obstacles")
        self._stair_kind = TERRAIN_KINDS.index("stairs")
        self.episode_tracking_lin = torch.zeros((self.batch_size,), dtype=gs.tc_float, device=gs.device)
        self._push_force = torch.zeros((self.batch_size, 3), dtype=gs.tc_float, device=gs.device)
        self._action_delay = torch.zeros((self.batch_size,), dtype=gs.tc_float, device=gs.device)
        self._cenet = None
        self._cenet_optimizer = None
        # Count, sum and sum of squares of the episodic returns finished since the last bootstrapping draw.
        self._finished_return_moments = torch.zeros((3,), dtype=gs.tc_float, device=gs.device)
        self._is_ground_truth = False
        self._is_latent_sampled = False
        self._published_observations = None
        self._proprio_dim = 9 + self.num_actions * 3
        self._height_scan_offsets = None
        if self.is_dreamwaq_policy:
            self._cenet = CENet(self._proprio_dim).to(device=gs.device, dtype=gs.tc_float)
            self._cenet_optimizer = torch.optim.Adam(self._cenet.parameters(), lr=1.0e-3)
            self._history = torch.zeros(
                (self.batch_size, HISTORY_STEPS, self._proprio_dim), dtype=gs.tc_float, device=gs.device
            )
            self._episode_return = torch.zeros((self.batch_size,), dtype=gs.tc_float, device=gs.device)
            scan_x = torch.tensor(HEIGHT_SCAN_X, dtype=gs.tc_float, device=gs.device)
            scan_y = torch.tensor(HEIGHT_SCAN_Y, dtype=gs.tc_float, device=gs.device)
            grid_x, grid_y = torch.meshgrid(scan_x, scan_y, indexing="ij")
            self._height_scan_offsets = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        self._feet_pos = None
        self._feet_vel = None
        self._dof_acc = None
        self._motor_torque = None
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
            "feet_clearance": self._reward_feet_clearance,
            "action_rate": self._reward_action_rate,
            "action_smoothness": self._reward_action_smoothness,
            "power_distribution": self._reward_power_distribution,
            "similar_to_default": self._reward_similar_to_default,
        }
        # The flat walker folds the control period into each weight. Curriculum weights are the per-step multipliers.
        scale_by_dt = self.terrain_kind != "curriculum"
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            if scale_by_dt:
                self.reward_scales[name] *= self.dt
            self.reward_functions[name] = reward_fns[name]
            self.episode_sums[name] = torch.zeros((self.batch_size,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    @property
    def is_dreamwaq_policy(self):
        return self._policy == POLICY_DREAMWAQ

    @property
    def speed_vx_cap(self):
        """Upper end of the forward command range, or None without the speed curriculum."""
        if self._speed_level is None:
            return None
        return SPEED_VX_CAPS[self._speed_level]

    @property
    def curriculum_state(self):
        """Return a copy of the speed and terrain curriculum progress."""
        return CurriculumState(
            speed_level=self._speed_level,
            speed_iters_at_cap=self._speed_iters_at_cap,
            speed_tracking_means=list(self._speed_tracking_means),
            terrain_levels=self.terrain_levels.clone(),
            promote_streak=self._promote_streak.clone(),
        )

    def load_curriculum_state(self, state, load_terrain):
        """Restore the curriculum progress saved by `curriculum_state`.

        The forward cap and its promotion window always load. `load_terrain` also restores the terrain row and the
        promotion streak of every environment, which requires the same number of environments as the saved run.
        """
        self._speed_level = state.speed_level
        self._speed_iters_at_cap = state.speed_iters_at_cap
        self._speed_tracking_means = list(state.speed_tracking_means)
        if self._speed_level is not None:
            self.commands_limits[1][0] = SPEED_VX_CAPS[self._speed_level]
            self.command_cfg["lin_vel_x_range"][1] = SPEED_VX_CAPS[self._speed_level]
        if not load_terrain:
            return
        if state.terrain_levels.shape != self.terrain_levels.shape:
            raise gs.GenesisException(
                f"The checkpoint holds terrain rows for {state.terrain_levels.shape[0]} environments, "
                f"this run has {self.batch_size}."
            )
        self.terrain_levels.copy_(state.terrain_levels)
        self._promote_streak.copy_(state.promote_streak)

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
        xs = (torch.arange(self.n_levels, dtype=gs.tc_float, device=gs.device) + 0.5) * self._subterrain_size[0]
        ys = (torch.arange(n_types, dtype=gs.tc_float, device=gs.device) + 0.5) * self._subterrain_size[1]
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        xy = torch.stack((grid_x, grid_y), dim=-1).reshape(self.n_levels * n_types, 2)
        heights = self.terrain.get_terrain_height(xy)
        if self.num_envs > 0:
            heights = heights[0]
        heights = heights.reshape(self.n_levels, n_types)
        return torch.stack((grid_x, grid_y, heights + self.init_base_pos[2]), dim=-1)

    def terrain_height(self, xy):
        """Return the ground height under world x-y positions of shape (batch, 2) or (batch, n_points, 2)."""
        if self.terrain is None:
            return torch.zeros(xy.shape[:-1], dtype=gs.tc_float, device=gs.device)
        # A point more than one cell past the height field comes back as NaN and poisons the height rewards.
        xy = torch.stack(
            (
                xy[..., 0].clamp(0.0, self._terrain_span_x),
                xy[..., 1].clamp(0.0, self._terrain_span_y),
            ),
            dim=-1,
        )
        if self.num_envs == 0:
            return self.terrain.get_terrain_height(xy[0]).unsqueeze(0)
        if xy.ndim == 2:
            return self.terrain.get_terrain_height(xy[:, None, :])[:, 0]
        return self.terrain.get_terrain_height(xy)

    def _height_scan(self):
        """Return the base height above each height-scan point, offset and clipped as the critic reads it."""
        heading = transform_by_quat(self._forward_axis, self.base_quat)[:, :2]
        heading = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp(min=gs.EPS)
        offsets_x = self._height_scan_offsets[:, 0]
        offsets_y = self._height_scan_offsets[:, 1]
        points_x = heading[:, None, 0] * offsets_x - heading[:, None, 1] * offsets_y
        points_y = heading[:, None, 1] * offsets_x + heading[:, None, 0] * offsets_y
        points = self.base_pos[:, None, :2] + torch.stack((points_x, points_y), dim=-1)
        return (self.base_pos[:, 2:3] - 0.5 - self.terrain_height(points)).clamp(-1.0, 1.0)

    def _reset_mask(self, envs_idx):
        if envs_idx is None:
            return torch.ones((self.batch_size,), dtype=torch.bool, device=gs.device)
        return envs_idx

    def _sample_interval(self, low, high, shape):
        return (high - low) * torch.rand(shape, dtype=gs.tc_float, device=gs.device) + low

    def _is_episode_complete(self, envs_idx):
        # The first episode after a random start length has no comparable return or tracking, so it is left out.
        return self._reset_mask(envs_idx) & (self._episode_steps > 0) & (self._episode_steps == self.episode_length_buf)

    def _update_terrain_curriculum(self, envs_idx):
        if self.terrain is None or not self._is_terrain_curriculum:
            return
        # A row rises after an episode that lasts to the timeout with mean linear tracking above the upper fraction,
        # and falls after a fall or a mean below the lower fraction. Obstacles and stairs rise only after consecutive
        # rising episodes. A row past the last one stays on the last one.
        is_complete = self._is_episode_complete(envs_idx)
        is_timeout = self.episode_length_buf > self.max_episode_length
        mean_tracking = self.episode_tracking_lin / self._episode_steps.clamp(min=1)
        is_up = is_complete & is_timeout & (mean_tracking > self._curriculum_up)
        is_down = is_complete & (~is_timeout | (mean_tracking < self._curriculum_down))
        is_slow = (self.terrain_types == self._obstacle_kind) | (self.terrain_types == self._stair_kind)
        streak = torch.where(
            is_complete & is_slow,
            torch.where(is_up, self._promote_streak + 1, 0),
            self._promote_streak,
        )
        is_slow_up = is_slow & (streak >= SLOW_PROMOTE_EPISODES)
        self._promote_streak = torch.where(is_slow_up, 0, streak)
        is_level_up = torch.where(is_slow, is_slow_up, is_up)
        self.terrain_levels += torch.where(is_level_up, 1, 0) - torch.where(is_down, 1, 0)
        self.terrain_levels.clamp_(0, self.n_levels - 1)

    def _randomize_domain(self, envs_idx):
        if not self._has_domain_randomization:
            return
        if envs_idx is None:
            env_ids = None
            count = self.batch_size
        else:
            # Link and dof setters take environment indices, so the reset mask is read back once.
            env_ids = torch.nonzero(envs_idx)[:, 0]
            count = len(env_ids)
            if count == 0:
                return
        setter_ids = env_ids if self.num_envs > 0 else None
        n_links = self.robot.n_links
        ranges = self._randomization

        if ranges.friction_coefficient is None:
            friction = self._sample_interval(*ranges.friction, (count, n_links))
        else:
            # The setter scales the authored coefficient, so an absolute coefficient becomes that ratio.
            friction = self._sample_interval(*ranges.friction_coefficient, (count, n_links)) / self._nominal_friction
        mass = self._nominal_mass * self._sample_interval(*ranges.mass_scale, (count, n_links))
        base_local = self._base_link.idx_local
        mass[:, base_local] += self._sample_interval(*ranges.payload, (count,))
        com = self._nominal_com.repeat(count, 1, 1)
        com[:, base_local] += self._sample_interval(-ranges.com_offset, ranges.com_offset, (count, 3))
        inertia = self._nominal_inertia * self._sample_interval(*ranges.inertia_scale, (count, n_links, 1, 1))
        gravity = self._nominal_gravity * self._sample_interval(*ranges.gravity_scale, (count, 1))
        kp = self.env_cfg["kp"] * self._sample_interval(*ranges.gain_scale, (count, self.num_actions))
        kd = self.env_cfg["kd"] * self._sample_interval(*ranges.gain_scale, (count, self.num_actions))
        force_scale = self._sample_interval(*ranges.force_scale, (count, self.num_actions))
        if self.num_envs == 0:
            friction, mass, com, inertia, gravity, kp, kd, force_scale = (
                friction[0],
                mass[0],
                com[0],
                inertia[0],
                gravity[0],
                kp[0],
                kd[0],
                force_scale[0],
            )
        self.robot.set_friction_ratio(friction, envs_idx=setter_ids)
        self.robot.set_links_mass(mass, envs_idx=setter_ids)
        self.robot.set_links_COM(com, envs_idx=setter_ids)
        self.robot.set_links_inertia(inertia, envs_idx=setter_ids)
        self.robot.set_dofs_kp(kp, self.motors_dof_idx, setter_ids)
        self.robot.set_dofs_kv(kd, self.motors_dof_idx, setter_ids)
        self.robot.set_dofs_force_range(
            self._nominal_force_lower * force_scale,
            self._nominal_force_upper * force_scale,
            self.motors_dof_idx,
            setter_ids,
        )
        self.scene.sim.set_gravity(gravity, envs_idx=setter_ids)

        push = torch.rand((count, 3), dtype=gs.tc_float, device=gs.device) * 2.0 - 1.0
        push[:, :2] *= ranges.push_xy
        push[:, 2] *= ranges.push_z
        delay = self._sample_interval(*ranges.system_delay, (count,))
        if env_ids is None:
            self._push_force.copy_(push)
            self._action_delay.copy_(delay)
        else:
            self._push_force[env_ids] = push
            self._action_delay[env_ids] = delay

    def _apply_push(self):
        if not self._has_domain_randomization:
            return
        force = self._push_force if self.num_envs > 0 else self._push_force[0]
        self._base_link.apply_external_wrench(force=force)

    def _spawn_qpos(self):
        if self.terrain is None:
            return self.init_qpos
        qpos = self.init_qpos.unsqueeze(0).repeat(self.batch_size, 1)
        qpos[:, :3] = self.cell_origins[self.terrain_levels, self.terrain_types]
        if self.num_envs == 0:
            return qpos[0]
        return qpos

    def advance_speed_cap(self):
        """Record this rollout's tracking and raise the forward cap when the promotion rule passes.

        The returned pair is the new cap and the mean of the last 50 per-step tracking rewards. A rollout that does
        not raise the cap returns nothing.
        """
        if self._speed_level is None:
            return None
        mean_step = float(self._speed_track_sum) / self._speed_track_count
        self._speed_track_sum.zero_()
        self._speed_track_count = 0
        self._speed_iters_at_cap += 1
        self._speed_tracking_means.append(mean_step)
        level, tracking = raised_speed_cap(self._speed_level, self._speed_iters_at_cap, self._speed_tracking_means)
        if level == self._speed_level:
            return None
        self._speed_level = level
        self._speed_iters_at_cap = 0
        self._speed_tracking_means.clear()
        vx_max = SPEED_VX_CAPS[level]
        self.commands_limits[1][0] = vx_max
        self.command_cfg["lin_vel_x_range"][1] = vx_max
        return vx_max, tracking

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.batch_size,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def set_commands(self, commands):
        """Hold velocity commands and republish the current observation with them.

        `commands` holds vx, vy and yaw rate, shared by every environment or given per environment. The proprioceptive
        history keeps its length: only its newest frame takes the new command.
        """
        commands = torch.as_tensor(commands, dtype=gs.tc_float, device=gs.device)
        if commands.shape != self.commands.shape:
            commands = commands.reshape(1, -1).expand(self.batch_size, -1)
        self.commands.copy_(commands)
        self.obs_buf = self._proprioception()
        if self.is_dreamwaq_policy:
            self._history[:, -1] = self.obs_buf
        return self.get_observations()

    def _read_feet_and_motors(self):
        needs_feet = "feet_clearance" in self.reward_functions
        if needs_feet:
            self._feet_pos = self._batch(self.robot.get_links_pos(self._feet_idx))
            self._feet_vel = self._batch(self.robot.get_links_vel(self._feet_idx))
        if "dof_acc" in self.reward_functions:
            self._dof_acc = self._batch(self.robot.get_dofs_acc(self.motors_dof_idx))
        if "joint_power" in self.reward_functions or "power_distribution" in self.reward_functions:
            self._motor_torque = self._batch(self.robot.get_dofs_control_force(self.motors_dof_idx))
            self._motor_power = torch.abs(self._motor_torque) * torch.abs(self.dof_vel)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        delayed_actions = self.last_actions if self.simulate_action_latency else self.actions
        previous_actions = self.last_last_actions if self.simulate_action_latency else self.last_actions
        # A delay shorter than the control period blends the target toward the one before it. Zero delay keeps it.
        alpha = (self._action_delay / self.dt).clamp(0.0, 1.0).unsqueeze(-1)
        exec_actions = delayed_actions + alpha * (previous_actions - delayed_actions)
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        ordered = target_dof_pos[:, self.actions_dof_idx]
        self._apply_push()
        self.robot.control_dofs_position(ordered if self.num_envs > 0 else ordered[0], slice(6, 18))
        # The cached speeds are the previous step. A faster environment is back on its spawn before integration, and
        # the reset below ends that episode.
        joint_speed = torch.amax(self.dof_vel.abs(), dim=-1)
        linear_speed = torch.linalg.norm(self.base_lin_vel, dim=-1)
        angular_speed = torch.linalg.norm(self.base_ang_vel, dim=-1)
        is_diverged = (
            (joint_speed > MAX_JOINT_SPEED)
            | (linear_speed > MAX_BASE_LINEAR_SPEED)
            | (angular_speed > MAX_BASE_ANGULAR_SPEED)
        )
        if bool(is_diverged.any()):
            pose_envs = None if self.num_envs == 0 else is_diverged
            self.robot.set_qpos(self._spawn_qpos(), envs_idx=pose_envs, zero_velocity=True, skip_forward=True)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self._episode_steps += 1
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
            if self._speed_level is not None and name == "tracking_lin_vel":
                self._speed_track_sum += rew.sum()
                self._speed_track_count += rew.numel()
        # The episode sums keep the raw step. The learner reward is bounded so one diverging environment cannot set
        # the return scale.
        self.rew_buf.clamp_(min=-5.0, max=5.0)
        if self.is_dreamwaq_policy:
            self._episode_return += self.rew_buf

        if self._resampling_steps is not None:
            self._resample_commands(self.episode_length_buf % self._resampling_steps == 0)

        # A fall is a contact on the base. The flat walker also falls past its roll and pitch limits.
        base_force = self._batch(self.robot.get_links_net_contact_force())[:, self._base_link.idx_local]
        self.is_fallen = torch.linalg.norm(base_force, dim=-1) > BASE_CONTACT_FORCE
        if "termination_if_roll_greater_than" in self.env_cfg:
            self.is_fallen |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
            self.is_fallen |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        if self._is_fall_terminal:
            self.reset_buf |= self.is_fallen
        self.reset_buf |= is_diverged
        # A pose reset clears errno. Checking it here is what makes an overflow halt instead of respawning forever.
        error_mask = self.scene.rigid_solver.get_error_envs_mask()
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
        if not self.is_dreamwaq_policy:
            self._published_observations = TensorDict({"policy": self.obs_buf}, batch_size=[self.batch_size])
            return self._published_observations
        velocity, latent = self._velocity_and_latent()
        self._published_observations = TensorDict(
            {
                "proprio": self.obs_buf,
                "est_vel": velocity,
                "latent": latent,
                "lin_vel": self.base_lin_vel,
                "disturbance": self._push_force,
                "height_scan": self._height_scan(),
                "history": self._history,
            },
            batch_size=[self.batch_size],
        )
        return self._published_observations

    def _velocity_and_latent(self):
        """Return the base velocity and context vector the actor reads on this step.

        Adaptive bootstrapping swaps the velocity alone: the measured one without bootstrapping, the estimated one with
        it. The context vector has no measured counterpart and always comes from the estimator.
        """
        # The history can be an inference tensor from the rollout. Autograd has to save a normal tensor.
        history = self._history.detach().clone()
        belief = self._cenet.encode(history)
        latent = self._cenet.reparameterize(belief) if self._is_latent_sampled else belief.mean
        velocity = self.base_lin_vel if self._is_ground_truth else belief.velocity
        return velocity.detach(), latent.detach()

    def set_estimator_ground_truth(self, use_ground_truth):
        """Feed the actor the measured or the estimated velocity, including in the observation already published."""
        self._is_ground_truth = use_ground_truth
        if not self.is_dreamwaq_policy or self._published_observations is None:
            return
        velocity, latent = self._velocity_and_latent()
        self._published_observations["est_vel"] = velocity
        self._published_observations["latent"] = latent

    def set_latent_sampling(self, sample_latent):
        """Sample the context vector during a rollout, or publish its mean."""
        self._is_latent_sampled = sample_latent

    def adaboot_uses_truth(self):
        """Draw whether the next rollout feeds the actor the measured velocity, see adaboot_probability in cenet.py.

        The draw reads the episodic returns finished since the previous draw. Without any, the actor reads the measured
        velocity.
        """
        if not self.is_dreamwaq_policy:
            return True
        count, total, total_square = self._finished_return_moments
        self._finished_return_moments.zero_()
        if count < 0.5:
            return True
        mean = total / count
        std = (total_square / count - mean.square()).clamp(min=0.0).sqrt()
        return bool(torch.rand((), device=gs.device) >= adaboot_probability(mean, std))

    def update_estimator(self, storage):
        """Fit the context estimator on one rollout and return its loss.

        The reconstruction target is the next stored proprioception. A finished step is left out, because the following
        observation belongs to the next episode. A rollout with no continuing step returns zero.
        """
        if not self.is_dreamwaq_policy:
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
        estimator_loss = float(loss.detach())
        is_mps = history.device.type == "mps"
        del history, next_proprio, velocity, alive, loss
        # Freed MPS blocks stay in the caching allocator. After those blocks are paged out, the process footprint
        # still includes them, and the next update allocates further blocks.
        if is_mps:
            torch.mps.empty_cache()
        return estimator_loss

    def cenet_state_dict(self):
        """Return the estimator weights and optimizer state."""
        if not self.is_dreamwaq_policy:
            raise gs.GenesisException("CENet state is only defined for the DreamWaQ policy.")
        return {"model": self._cenet.state_dict(), "optimizer": self._cenet_optimizer.state_dict()}

    def load_cenet_state_dict(self, state):
        """Load estimator weights and optimizer state saved beside the policy."""
        if not self.is_dreamwaq_policy:
            raise gs.GenesisException("CENet state is only defined for the DreamWaQ policy.")
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
        if self.is_dreamwaq_policy:
            # Running moments keep the step free of a device read and the memory bounded between two draws.
            is_complete = self._is_episode_complete(envs_idx)
            returns = torch.where(is_complete, self._episode_return, 0.0)
            self._finished_return_moments += torch.stack((is_complete.sum(), returns.sum(), returns.square().sum()))
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
            self._episode_steps.zero_()
            self.episode_tracking_lin.zero_()
            self.reset_buf.fill_(True)
            if self.is_dreamwaq_policy:
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
            self._episode_steps.masked_fill_(envs_idx, 0)
            self.episode_tracking_lin.masked_fill_(envs_idx, 0.0)
            self.reset_buf.masked_fill_(envs_idx, True)
            if self.is_dreamwaq_policy:
                self._episode_return.masked_fill_(envs_idx, 0.0)

        if self.terrain is not None:
            origin = self.cell_origins[self.terrain_levels, self.terrain_types]
            if envs_idx is None:
                self.base_pos.copy_(origin)
            else:
                torch.where(envs_idx[:, None], origin, self.base_pos, out=self.base_pos)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.batch_size
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

    def _proprioception(self):
        return torch.concatenate(
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

    def _update_observation(self):
        self.obs_buf = self._proprioception()
        if not self.is_dreamwaq_policy:
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
        # Penalize the squared norm of gravity in the body frame
        return torch.sum(torch.square(self.projected_gravity), dim=1)

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
        height = self.base_pos[:, 2] - self.terrain_height(self.base_pos[:, :2])
        return torch.square(height - self.reward_cfg["base_height_target"])

    def _reward_feet_clearance(self):
        # Penalize swing feet whose height above the terrain misses the clearance target. Stance speed is ~0.
        clearance = self._feet_pos[..., 2] - self.terrain_height(self._feet_pos[..., :2])
        speed = torch.linalg.norm(self._feet_vel[..., :2], dim=-1)
        return torch.sum(torch.square(clearance - self.reward_cfg["feet_height_target"]) * speed, dim=1)

    def _reward_power_distribution(self):
        # Penalize the square of the variance of signed joint power across the motors
        signed_power = self._motor_torque * self.dof_vel
        return torch.square(torch.var(signed_power, dim=1, unbiased=False))
