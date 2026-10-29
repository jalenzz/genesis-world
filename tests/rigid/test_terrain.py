import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import igl
import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.terrain as tu
from genesis.utils.misc import get_assets_dir, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal

_LOCOMOTION_DIR = Path(__file__).resolve().parents[2] / "examples" / "locomotion"
sys.path.insert(0, str(_LOCOMOTION_DIR))
from cenet import CENet, HISTORY_STEPS, LATENT_DIM, adaboot_truth_probability, adaboot_uses_truth

_GO2_ENV_PATH = _LOCOMOTION_DIR / "go2_env.py"
_go2_spec = importlib.util.spec_from_file_location("go2_locomotion_env", _GO2_ENV_PATH)
go2_locomotion_env = importlib.util.module_from_spec(_go2_spec)
_go2_spec.loader.exec_module(go2_locomotion_env)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(gs.gpu, marks=pytest.mark.required),
        gs.cpu,  # This test takes too much time of CPU (~1000s)
    ],
)
@pytest.mark.parametrize("is_named", [True, False])
def test_generation(is_named, show_viewer, tol):
    TERRAIN_PATTERN = [
        ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
        ["flat_terrain", "fractal_terrain", "random_uniform_terrain", "sloped_terrain", "flat_terrain"],
        ["flat_terrain", "pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain", "flat_terrain"],
        ["flat_terrain", "stairs_terrain", "pyramid_stairs_terrain", "stepping_stones_terrain", "flat_terrain"],
        ["flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain", "flat_terrain"],
    ]
    TERRAIN_OFFSET = (10.0, -10.0, -1.0)
    TERRAIN_SIZE = 10.0
    SUBTERRAIN_GRID_SIZE = 15
    OBJ_SIZE = 0.1
    OBJ_HEIGHT_INIT = 0.3
    NUM_OBJ_SQRT = 15

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.006,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0 + TERRAIN_OFFSET[0], -5.0 + TERRAIN_OFFSET[1], 10.0 + TERRAIN_OFFSET[2]),
            camera_lookat=(5.0 + TERRAIN_OFFSET[0], 5.0 + TERRAIN_OFFSET[1], 0.0 + TERRAIN_OFFSET[2]),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    terrain_kwargs = dict(
        pos=TERRAIN_OFFSET,
        n_subterrains=(len(TERRAIN_PATTERN),) * 2,
        subterrain_size=(TERRAIN_SIZE / len(TERRAIN_PATTERN),) * 2,
        horizontal_scale=TERRAIN_SIZE / len(TERRAIN_PATTERN) / SUBTERRAIN_GRID_SIZE,
        vertical_scale=0.05,
        subterrain_types=TERRAIN_PATTERN,
        randomize=False,
        name="my_terrain" if is_named else None,
    )
    # FIXME: Collision detection is very unstable for 'stepping_stones' pattern.
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(**terrain_kwargs),
    )
    obj = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(1.0, 1.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    scene.build(n_envs=NUM_OBJ_SQRT**2)

    # Spread objects across the entire field
    obj_pos_1d = torch.linspace(OBJ_SIZE / 2, TERRAIN_SIZE - OBJ_SIZE / 2, NUM_OBJ_SQRT)
    obj_pos_init_rel = torch.cartesian_prod(*(obj_pos_1d,) * 2, torch.tensor((OBJ_HEIGHT_INIT,)))
    obj.set_pos(obj_pos_init_rel + torch.tensor(TERRAIN_OFFSET))

    # Drop the objects and simulate for a while.
    for _ in range(600):
        scene.step()

    # Check that objects are not moving anymore
    assert_allclose(obj.get_vel(), 0.0, tol=0.1)

    # Check the the terrain is not entirely flat and has the expected size
    terrain_min_corner, terrain_max_corner = tensor_to_array(terrain.geoms[0].get_AABB()) - TERRAIN_OFFSET
    assert_allclose(terrain_min_corner[:2], 0.0, tol=gs.EPS)
    assert_allclose(terrain_max_corner[:2], TERRAIN_SIZE, tol=gs.EPS)
    assert terrain_min_corner[2] < -1.0  # Stepping stone depth
    assert terrain_max_corner[2] > 0.01  # FIXME: It should not be larger than 'vertical_scale'

    # Check that all objects are in contact with the terrain
    obj_pos = tensor_to_array(obj.get_pos()) - TERRAIN_OFFSET
    terrain_mesh = terrain.geoms[0].mesh
    signed_distance, *_ = igl.signed_distance(obj_pos, terrain_mesh.verts, terrain_mesh.faces)
    assert (signed_distance > 0.0).all()
    assert (signed_distance < 2 * OBJ_SIZE).all()

    # Check if cache is being reloaded as expected
    if is_named:
        scene = gs.Scene()
        terrain_2 = scene.add_entity(
            morph=gs.morphs.Terrain(**{**terrain_kwargs, **dict(randomize=True)}),
        )
        terrain_2_mesh = terrain_2.geoms[0].mesh
        assert_allclose(terrain_mesh.verts, terrain_2_mesh.verts, tol=tol)


@pytest.mark.required
def test_discrete_obstacles():
    scene = gs.Scene()
    terrain = scene.add_entity(
        gs.morphs.Terrain(
            n_subterrains=(1, 1),
            subterrain_size=(6.0, 6.0),
            horizontal_scale=0.5,
            vertical_scale=0.5,
            subterrain_types=[["discrete_obstacles_terrain"]],
            subterrain_parameters={
                "discrete_obstacles_terrain": {
                    "max_height": 1.0,
                    "platform_size": 1.0,
                }
            },
        )
    )
    scene.build()
    height_field = terrain.geoms[0].metadata["height_field"]
    platform = height_field[5:7, 5:7]

    assert height_field.max() == 2.0
    assert height_field.min() == -2.0
    assert (platform < gs.EPS).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_get_terrain_height(n_envs, tol):
    height_field = np.array(
        [
            [0.0, 4.0, 9.0, 15.0],
            [10.0, 20.0, 33.0, 49.0],
            [30.0, 50.0, 73.0, 99.0],
        ],
    )
    quat_yaw = gu.xyz_to_quat(np.array((0.0, 0.0, 90.0)), degrees=True)

    scene = gs.Scene(show_viewer=False)
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=(2.0, -1.0, 0.3),
            quat=quat_yaw,
            batch_fixed_verts=True,
            horizontal_scale=0.5,
            vertical_scale=0.1,
            height_field=height_field,
        ),
    )
    visual_terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=(30.0, 40.0, 3.0),
            quat=gu.xyz_to_quat(np.array((0.0, 0.0, 37.0)), degrees=True),
            collision=False,
            horizontal_scale=0.25,
            vertical_scale=0.01,
            height_field=np.add.outer(np.arange(3), 2 * np.arange(3)),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
    )
    scene.build(n_envs=n_envs)

    initial_height = terrain.get_terrain_height((2.0, -0.5))
    assert_allclose(initial_height, 1.3, tol=tol)

    visual_pose = gu.trans_quat_to_T(np.array(visual_terrain.morph.pos), np.array(visual_terrain.morph.quat))
    visual_position = visual_pose @ np.array((0.5, 0.25, 0.0, 1.0))
    visual_height = visual_terrain.get_terrain_height(visual_position[:2])
    assert_allclose(visual_height, 3.04, tol=tol)

    if n_envs:
        shared_positions = terrain.get_terrain_height((((2.0, -1.0), (2.0, -0.5)),))
        assert_allclose(shared_positions, ((0.3, 1.3), (0.3, 1.3)), tol=tol)

        terrain.set_pos(((10.0, 20.0, 1.0), (-3.0, 4.0, -2.0)), relative=False)
        terrain.set_quat(np.stack((gu.identity_quat(), quat_yaw)), relative=False)
    else:
        terrain.set_pos((10.0, 20.0, 1.0), relative=False)
        terrain.set_quat((1.0, 0.0, 0.0, 0.0), relative=False)

    positions = (
        (10.0, 20.0),
        (10.5, 20.0),
        (10.0, 20.5),
        (10.125, 20.25),
        (10.375, 20.25),
        (11.0, 21.5),
    )
    terrain_mesh = terrain.geoms[0].get_trimesh().copy()
    terrain_mesh.apply_transform(gu.trans_quat_to_T(np.array((10.0, 20.0, 1.0)), gu.identity_quat()))
    ray_origins = np.column_stack((positions, np.full(len(positions), terrain_mesh.bounds[1, 2] + 1.0)))
    ray_directions = np.tile((0.0, 0.0, -1.0), (len(positions), 1))
    locations, ray_ids, _ = terrain_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
    )
    expected = locations[np.argsort(ray_ids), 2]
    if n_envs:
        heights = terrain.get_terrain_height(positions, envs_idx=[0])

        positions_per_env = ((positions[3], positions[4]), ((-3.25, 4.125), (-3.25, 4.375)))
        heights_per_env = terrain.get_terrain_height(positions_per_env)
        assert_allclose(heights_per_env, ((1.45, 2.1), (-1.55, -0.9)), tol=tol)

        one_position_per_env = terrain.get_terrain_height(((positions_per_env[0][0],), (positions_per_env[1][0],)))
        assert_allclose(one_position_per_env, ((1.45,), (-1.55,)), tol=tol)

        heights_env_1 = terrain.get_terrain_height(positions_per_env[1], envs_idx=1)
        assert_allclose(heights_env_1, (-1.55, -0.9), tol=tol)
    else:
        heights = terrain.get_terrain_height(positions)

    assert_allclose(heights, expected, tol=1e-6)

    envs_idx = [0] if n_envs else None
    boundary_heights = terrain.get_terrain_height(((9.5, 20.0), (9.499, 20.0), (np.nan, 20.0)), envs_idx=envs_idx)
    assert_allclose(boundary_heights[..., 0], 1.0, tol=tol)
    assert torch.isnan(boundary_heights[..., 1:]).all()

    with pytest.raises(gs.GenesisException):
        terrain.get_terrain_height((10.0, 20.0, 1.0), envs_idx=envs_idx)

    terrain.set_quat(gu.xyz_to_quat(np.array((5e-4, 0.0, 0.0))), envs_idx=envs_idx, relative=False)
    height = terrain.get_terrain_height((10.0, 20.0), envs_idx=envs_idx)
    assert_allclose(height, 1.0, tol=tol)

    terrain.set_quat(gu.xyz_to_quat(np.array((2e-3, 0.0, 0.0))), envs_idx=envs_idx, relative=False)
    height = terrain.get_terrain_height((10.0, 20.0), envs_idx=envs_idx)
    assert torch.isnan(height).all()


@pytest.mark.required
def test_subterrain_parameters(show_viewer):
    scene_ref = gs.Scene(show_viewer=show_viewer)
    terrain_ref = scene_ref.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
            name="wave_terrain",
        )
    )

    height_ref = terrain_ref.geoms[0].metadata["height_field"]

    scene_test = gs.Scene(show_viewer=show_viewer)
    terrain_test = scene_test.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
            name="wave_terrain",
            subterrain_parameters={"wave_terrain": {"amplitude": 0.2}},
        )
    )

    height_test = terrain_test.geoms[0].metadata["height_field"]

    assert_allclose((height_ref * 2.0), height_test, tol=gs.EPS)


def test_mesh_to_heightfield(tmp_path, show_viewer):
    horizontal_scale = 2.0
    path_terrain = os.path.join(get_assets_dir(), "meshes", "terrain_45.obj")

    hf_terrain, xs, ys = tu.mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=1)

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(5, 0, -5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -5, 7),
            camera_lookat=(10, 15, 4),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    terrain_heightfield = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            height_field=hf_terrain,
            pos=translation,
        ),
        vis_mode="collision",
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(
            pos=(10, 15, 7),
            radius=1,
        ),
        vis_mode="collision",
    )
    scene.build()

    for i in range(70):
        scene.step()

    # The ball is at rest (on the terrain)
    assert_allclose(ball.get_dofs_velocity(), 0, tol=1e-3)


@pytest.mark.required
def test_box_on_terrain_no_spurious_spin(show_viewer):
    BOX_SIZE = (0.12, 0.06, 0.025)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -3.5, 2.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=(-1.5, -1.5, 0.0),
            height_field=np.zeros((4, 4), dtype=np.float32),
            horizontal_scale=1.0,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 0.4),
        ),
    )
    box = scene.add_entity(
        morph=(
            gs.morphs.Box(size=BOX_SIZE),
            gs.morphs.MeshSet(
                files=(trimesh.creation.box(extents=BOX_SIZE),),
                decimate=False,
            ),
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.2, 0.2),
        ),
    )

    # 4x4 grid spread across the 3 m x 3 m terrain.
    grid = np.linspace(-1.2, 1.2, 4)
    xy = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    n_envs = xy.shape[0]
    scene.build(n_envs=n_envs)

    z = BOX_SIZE[2] / 2
    pos = np.concatenate([xy, np.full((n_envs, 1), z)], axis=-1).astype(np.float32)
    box.set_pos(torch.from_numpy(pos))
    box.set_dofs_velocity(torch.zeros((n_envs, 6)))
    quat_initial = tensor_to_array(box.get_quat())

    for _ in range(500):
        scene.step()

    quat_delta = gu.transform_quat_by_quat(tensor_to_array(box.get_quat()), gu.inv_quat(quat_initial))
    assert_allclose(gu.quat_to_rotvec(quat_delta), 0.0, tol=0.02)


@pytest.mark.required
def test_multicontact_sphere_vs_terrain(show_viewer, tol):
    GRID_N = 13
    APEX_IDX = GRID_N // 2
    VERTICAL_SCALE = 0.04
    HORIZONTAL_SCALE = 0.05
    SPHERE_RADIUS = 0.1

    ii, jj = np.meshgrid(np.arange(GRID_N), np.arange(GRID_N), indexing="ij")
    hf = (np.abs(ii - APEX_IDX) + np.abs(jj - APEX_IDX)).astype(np.int16)
    terrain_pos = (-APEX_IDX * HORIZONTAL_SCALE, -APEX_IDX * HORIZONTAL_SCALE, 0.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Terrain(
            pos=terrain_pos,
            height_field=hf,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 0.4),
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.2, 0.2),
        ),
        visualize_contact=True,
    )

    scene.build()

    sphere.set_pos(torch.tensor([0.0, 0.0, 0.25]))

    for _ in range(80):
        scene.step()
        print(tensor_to_array(sphere.get_dofs_velocity()))

    # Sphere is at rest at the apex of the pit. Equilibrium height is set by the four-wall solid angle: contacts on the
    # opposing pyramid walls push the sphere upward, so it sits above the apex vertex but well below the rim.
    pos_final = tensor_to_array(sphere.get_pos())
    assert_allclose(pos_final[:2], 0.0, tol=2.0 * HORIZONTAL_SCALE)
    assert SPHERE_RADIUS < pos_final[2] < APEX_IDX * VERTICAL_SCALE + SPHERE_RADIUS

    vel_final = tensor_to_array(sphere.get_dofs_velocity())
    assert_allclose(vel_final, 0.0, tol=1e-5)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_curriculum_terrain_tracking(n_envs, show_viewer):
    env_cfg, obs_cfg, reward_cfg, command_cfg = go2_locomotion_env.locomotion_cfgs()
    go2_locomotion_env.apply_curriculum(env_cfg, reward_cfg)
    env_cfg["policy"] = go2_locomotion_env.POLICY_DREAMWAQ
    subterrain_size = (4.0, 4.0)
    env_cfg["episode_length_s"] = 1.0
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env_cfg["curriculum_up"] = 0.5
    env_cfg["curriculum_down"] = 0.05
    env_cfg["terrain_subterrain_size"] = subterrain_size
    env_cfg["terrain_horizontal_scale"] = 0.5
    env_cfg["terrain_vertical_scale"] = 0.01
    env_cfg["randomization"] = dict(
        friction=(0.8, 1.2),
        mass_scale=(0.9, 1.1),
        com_offset=0.0,
        inertia_scale=(1.0, 1.0),
        gain_scale=(1.0, 1.0),
        force_scale=(1.0, 1.0),
        gravity_scale=(1.0, 1.0),
        push_xy=0.0,
        push_z=0.0,
    )
    command_cfg["lin_vel_x_range"] = [0.0, 0.0]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]

    env = go2_locomotion_env.Go2Env(
        num_envs=n_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )
    obs = env.get_observations()
    assert obs["proprio"].shape[-1] == 45
    assert obs["est_vel"].shape[-1] == 3
    assert obs["latent"].shape[-1] == LATENT_DIM
    assert obs["disturbance"].shape[-1] == 3
    assert tuple(obs["history"].shape[-2:]) == (HISTORY_STEPS, 45)
    assert torch.allclose(obs["lin_vel"], torch.zeros_like(obs["lin_vel"]))
    assert obs["est_vel"].abs().max() > 1.0e-3
    assert all("height" not in key for key in obs.keys())
    groups = go2_locomotion_env.dreamwaq_obs_groups()
    actor_input = torch.cat([obs[name] for name in groups["actor"]], dim=-1)
    critic_input = torch.cat([obs[name] for name in groups["critic"]], dim=-1)
    assert groups["actor"] == ["proprio", "est_vel", "latent"]
    assert actor_input.shape[-1] == 45 + 3 + LATENT_DIM
    assert "lin_vel" not in groups["actor"]
    assert "disturbance" not in groups["actor"]
    assert groups["critic"] == ["proprio", "lin_vel", "disturbance"]
    assert critic_input.shape[-1] == 45 + 3 + 3

    smooth_level0 = (0.5 * subterrain_size[0], 0.5 * subterrain_size[1])
    smooth_level9 = (9.5 * subterrain_size[0], 0.5 * subterrain_size[1])
    stairs_level0 = (0.5 * subterrain_size[0], 3.5 * subterrain_size[1])
    stairs_level9 = (9.5 * subterrain_size[0], 3.5 * subterrain_size[1])
    terrain_heights = env.terrain.get_terrain_height((smooth_level0, smooth_level9, stairs_level0, stairs_level9))
    if n_envs > 0:
        terrain_heights = terrain_heights[0]
    # The center is the pyramid platform. On this coarse 4 m cell the 22 degree slope only lifts it by about 0.22 m.
    assert terrain_heights[1] > terrain_heights[0] + 0.2
    assert terrain_heights[3] > terrain_heights[2] + 0.15
    assert (terrain_heights[0].abs() < 0.05).all()

    mass_before = env.robot.get_links_mass().clone()
    batch_n = n_envs if n_envs > 0 else 1
    actions = torch.zeros((batch_n, env.num_actions), dtype=gs.tc_float, device=gs.device)
    for _ in range(env.max_episode_length):
        env.step(actions)
    assert torch.isfinite(env.rew_buf).all()

    feet_idx = [env.robot.get_link(name).idx_local for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot")]
    feet_pos = env.robot.get_links_pos(feet_idx)
    base_pos = env.robot.get_pos()
    if n_envs == 0:
        foot_ground = env.terrain.get_terrain_height(feet_pos[:, :2])
        clearance = feet_pos[:, 2] - foot_ground
        base_ground = env.terrain.get_terrain_height(base_pos[:2])
        base_height = base_pos[2] - base_ground
    else:
        foot_ground = env.terrain.get_terrain_height(feet_pos[..., :2])
        clearance = feet_pos[..., 2] - foot_ground
        base_ground = env.terrain.get_terrain_height(base_pos[:, None, :2])[:, 0]
        base_height = base_pos[:, 2] - base_ground
    assert (clearance > -0.03).all()
    nearest = clearance.min() if n_envs == 0 else clearance.min(dim=-1).values
    assert (nearest < 0.08).all()
    assert (base_height > 0.2).all()
    assert (base_height < 0.55).all()

    env.step(actions)
    qpos = env.robot.get_qpos()
    promoted_x = 1.5 * subterrain_size[0]
    if n_envs == 0:
        assert_allclose(qpos[:2], (promoted_x, 0.5 * subterrain_size[1]), atol=1e-4, rtol=0.0)
    else:
        assert_allclose(qpos[:, 0], promoted_x, atol=1e-4, rtol=0.0)
        assert_allclose(
            qpos[:, 1],
            (torch.arange(n_envs, device=qpos.device) + 0.5) * subterrain_size[1],
            atol=1e-4,
            rtol=0.0,
        )
    mass_after = env.robot.get_links_mass()
    assert ((mass_after - mass_before).abs() > 1e-3).any()
    ratio = mass_after / mass_before
    assert (ratio > 0.9 / 1.1).all()
    assert (ratio < 1.1 / 0.9).all()
    if n_envs > 0:
        assert ((mass_after[0] - mass_after[1]).abs() > 1e-3).any()

    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
    y_mid = 0.5 * subterrain_size[1]
    steep_x0 = 9.0 * subterrain_size[0]
    steep_tip = torch.empty((8,), dtype=gs.tc_float, device=gs.device)
    for i in range(8):
        env.base_pos[:, 0] = steep_x0 + (i + 0.5) * 0.5
        env.base_pos[:, 1] = y_mid
        env.base_quat[:] = identity
        steep_tip[i] = env._surface_tip_deg()[0]
    steep_i = int(torch.argmax(steep_tip))
    assert steep_tip[steep_i] > 15.0
    assert steep_tip[steep_i] < 30.0
    env.base_pos[:, 0] = steep_x0 + (steep_i + 0.5) * 0.5
    env.base_pos[:, 1] = y_mid
    normal = env._terrain_normal(env.base_pos[:, :2])
    parallel = gu.normalize(
        torch.stack((1.0 + normal[:, 2], -normal[:, 1], normal[:, 0], torch.zeros_like(normal[:, 0])), dim=-1)
    )
    env.base_quat[:] = parallel
    assert (env._surface_tip_deg() < 1.0).all()

    env.base_pos[:, 0] = smooth_level0[0]
    env.base_pos[:, 1] = smooth_level0[1]
    env.base_quat[:] = identity
    assert (env._surface_tip_deg() < 1.0).all()
    pitched = gu.xyz_to_quat(
        torch.tensor([0.0, 12.0, 0.0], dtype=gs.tc_float, device=gs.device),
        rpy=True,
        degrees=True,
    ).expand(env._batch_n, 4)
    env.base_quat[:] = pitched
    assert (env._surface_tip_deg() > 10.0).all()
    held_terrain = env.terrain
    env.terrain = None
    env.base_quat[:] = identity
    assert (env._surface_tip_deg() < 1.0).all()
    env.base_quat[:] = pitched
    assert (env._surface_tip_deg() > 10.0).all()
    env.terrain = held_terrain

    env.env_cfg["termination_if_roll_greater_than"] = 10
    env.env_cfg["termination_if_pitch_greater_than"] = 10
    # Hold the posed base still. The assertion is the termination flag, and a physics substep would move it.
    env.scene.step = lambda *args, **kwargs: None
    steep_xy = (steep_x0 + (steep_i + 0.5) * 0.5, y_mid)
    for xy, quat, falls in ((steep_xy, parallel, False), (smooth_level0, pitched, True)):
        env.base_pos[:, 0] = xy[0]
        env.base_pos[:, 1] = xy[1]
        height = env._terrain_z(env.base_pos[:, :2]) + env.init_base_pos[2]
        qpos = env.robot.get_qpos().clone()
        if n_envs == 0:
            qpos[0] = xy[0]
            qpos[1] = xy[1]
            qpos[2] = height[0]
            qpos[3:7] = quat[0]
        else:
            qpos[:, 0] = xy[0]
            qpos[:, 1] = xy[1]
            qpos[:, 2] = height
            qpos[:, 3:7] = quat
        env.robot.set_qpos(qpos)
        env.step(actions)
        if falls:
            assert bool(env.reset_buf.all())
        else:
            assert not bool(env.reset_buf.any())
            assert (env.episode_length_buf == 1).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_curriculum_action_reaches_joints(n_envs, show_viewer):
    env_cfg, obs_cfg, reward_cfg, command_cfg = go2_locomotion_env.locomotion_cfgs()
    go2_locomotion_env.apply_curriculum(env_cfg, reward_cfg)
    env_cfg["policy"] = go2_locomotion_env.POLICY_DREAMWAQ
    subterrain_size = (4.0, 4.0)
    env_cfg["episode_length_s"] = 20.0
    env_cfg["termination_if_roll_greater_than"] = 180
    env_cfg["termination_if_pitch_greater_than"] = 180
    env_cfg["terrain_subterrain_size"] = subterrain_size
    env_cfg["terrain_horizontal_scale"] = 0.5
    env_cfg["terrain_vertical_scale"] = 0.01
    env_cfg["max_collision_pairs"] = 2
    env_cfg["randomization"] = dict(
        friction=(1.0, 1.0),
        mass_scale=(1.0, 1.0),
        com_offset=0.0,
        inertia_scale=(1.0, 1.0),
        gain_scale=(1.0, 1.0),
        force_scale=(1.0, 1.0),
        gravity_scale=(1.0, 1.0),
        push_xy=0.0,
        push_z=0.0,
    )
    command_cfg["lin_vel_x_range"] = [0.0, 0.0]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]
    batch_n = n_envs if n_envs > 0 else 1
    held = torch.zeros((batch_n, env_cfg["num_actions"]), dtype=gs.tc_float, device=gs.device)
    overflow = go2_locomotion_env.Go2Env(
        num_envs=n_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )
    with pytest.raises(gs.GenesisException, match="Exceeding max number"):
        overflow.step(held)
    overflow.scene.destroy()

    env_cfg["max_collision_pairs"] = 20
    env = go2_locomotion_env.Go2Env(
        num_envs=n_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )
    assert env.scene.rigid_solver.max_collision_pairs == 256
    # The first step still executes a zero action. The next one must execute this action on the spawn cell.
    env.step(held)
    held_dof = env.robot.get_dofs_position().clone()
    env.step(torch.ones_like(held))
    assert (env.episode_length_buf == 2).all()
    assert not bool(env.reset_buf.any())
    assert (env.robot.get_dofs_position() - held_dof).abs().max() > 1e-3
    qpos = env.robot.get_qpos()
    spawn_x = 0.5 * subterrain_size[0]
    spawn_xs = qpos[0] if n_envs == 0 else qpos[:, 0]
    assert ((spawn_xs - spawn_x).abs() < 0.5).all()


def test_proprio_history_fills_from_integer_reset_mask():
    batch = 2
    proprio_dim = 45
    env = go2_locomotion_env.Go2Env.__new__(go2_locomotion_env.Go2Env)
    env._policy = go2_locomotion_env.POLICY_DREAMWAQ
    env._cenet = object()
    env._proprio_dim = proprio_dim
    env.obs_scales = {"ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
    env.base_ang_vel = torch.zeros(batch, 3)
    env.projected_gravity = torch.zeros(batch, 3)
    env.projected_gravity[:, 2] = -1.0
    env.commands = torch.zeros(batch, 3)
    env.commands_scale = torch.ones(3)
    env.dof_pos = torch.zeros(batch, 12)
    env.default_dof_pos = torch.zeros(12)
    env.dof_vel = torch.zeros(batch, 12)
    env.actions = torch.arange(batch * 12, dtype=gs.tc_float).reshape(batch, 12)
    env._history = torch.full((batch, HISTORY_STEPS, proprio_dim), -1.0)
    env.reset_buf = torch.tensor([1, 0], dtype=gs.tc_int)
    env._update_observation()
    assert_equal(env._history[0], env.obs_buf[0].expand(HISTORY_STEPS, proprio_dim))
    assert_equal(env._history[1, :-1], torch.full((HISTORY_STEPS - 1, proprio_dim), -1.0))
    assert_equal(env._history[1, -1], env.obs_buf[1])


def test_estimator_backward_from_inference_history():
    batch = 2
    steps = 3
    proprio_dim = 45
    env = go2_locomotion_env.Go2Env.__new__(go2_locomotion_env.Go2Env)
    env._policy = go2_locomotion_env.POLICY_DREAMWAQ
    env._batch_n = batch
    env._is_latent_sampled = False
    env._cenet = CENet(proprio_dim)
    env._cenet_optimizer = torch.optim.Adam(env._cenet.parameters(), lr=1.0e-3)
    with torch.inference_mode():
        env._history = torch.randn(batch, HISTORY_STEPS, proprio_dim)
        history = torch.randn(steps, batch, HISTORY_STEPS, proprio_dim)
        proprio = torch.randn(steps, batch, proprio_dim)
        velocity = torch.randn(steps, batch, 3)
        dones = torch.zeros(steps, batch, 1)
    env._published_observations = {}
    env.set_estimator_ground_truth(False)
    assert env._published_observations["est_vel"].shape == (batch, 3)
    assert env._published_observations["latent"].shape == (batch, LATENT_DIM)
    assert not env._published_observations["est_vel"].is_inference()

    storage = SimpleNamespace(
        observations={"history": history, "proprio": proprio, "lin_vel": velocity},
        dones=dones,
    )
    cache_calls = []
    empty_cache = torch.mps.empty_cache
    torch.mps.empty_cache = lambda: cache_calls.append(True)
    try:
        loss = env.update_estimator(storage)
    finally:
        torch.mps.empty_cache = empty_cache
    assert math.isfinite(loss)
    assert env._cenet.encoder[0].weight.grad is not None
    assert env._cenet.encoder[0].weight.grad.abs().sum() > 0
    assert history.grad is None
    assert not history.requires_grad
    assert history.device.type == "cpu"
    assert cache_calls == []


def test_cenet_belief_and_actor_groups():
    proprio_dim = 45
    batch = 4
    net = CENet(proprio_dim)
    history = torch.randn(batch, HISTORY_STEPS, proprio_dim)
    belief = net.encode(history)
    assert belief.velocity.shape == (batch, 3)
    assert belief.mean.shape == (batch, LATENT_DIM)
    assert belief.log_variance.shape == (batch, LATENT_DIM)
    latent = net.reparameterize(belief)
    reconstructed = net.decode(latent)
    assert latent.shape == (batch, LATENT_DIM)
    assert reconstructed.shape == (batch, proprio_dim)
    velocity = torch.randn(batch, 3)
    next_proprio = torch.randn(batch, proprio_dim)
    loss = net.loss(history, velocity, next_proprio)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert net.velocity_head.weight.grad is not None
    assert net.decoder[0].weight.grad is not None

    empty_returns = torch.empty(0)
    assert torch.allclose(adaboot_truth_probability(empty_returns), torch.ones(()))
    assert adaboot_uses_truth(empty_returns, torch.zeros(()))
    same_returns = torch.ones(8)
    assert torch.allclose(adaboot_truth_probability(same_returns), torch.zeros(()))
    assert not adaboot_uses_truth(same_returns, torch.tensor(0.5))
    varied_returns = torch.tensor([0.0, 2.0])
    expected = torch.tanh(varied_returns.std(unbiased=False) / varied_returns.mean().abs())
    assert torch.allclose(adaboot_truth_probability(varied_returns), expected)
    assert adaboot_uses_truth(varied_returns, torch.zeros(()))
    assert not adaboot_uses_truth(varied_returns, torch.ones(()))

    env_cfg, _, reward_cfg, command_cfg = go2_locomotion_env.locomotion_cfgs()
    assert reward_cfg["reward_scales"]["tracking_ang_vel"] == 0.2
    assert reward_cfg["reward_scales"]["similar_to_default"] == -0.1
    assert command_cfg["lin_vel_x_range"] == [0.5, 0.5]
    flat_groups = go2_locomotion_env.policy_obs_groups(env_cfg)
    assert flat_groups["actor"] == ["policy"]
    assert flat_groups["critic"] == ["policy"]
    go2_locomotion_env.apply_curriculum(env_cfg, reward_cfg, command_cfg)
    assert env_cfg.get("policy") is None
    assert go2_locomotion_env.policy_name(env_cfg) == go2_locomotion_env.POLICY_MLP
    env_cfg["policy"] = go2_locomotion_env.POLICY_DREAMWAQ
    scales = reward_cfg["reward_scales"]
    assert scales["tracking_lin_vel"] == 1.0
    assert scales["tracking_ang_vel"] == 0.5
    assert scales["lin_vel_z"] == -2.0
    assert scales["ang_vel_xy"] == -0.05
    assert scales["orientation"] == -0.2
    assert scales["dof_acc"] == -2.5e-7
    assert scales["joint_power"] == -2.0e-5
    assert scales["base_height"] == -1.0
    assert scales["feet_clearance"] == -0.01
    assert scales["action_rate"] == -0.01
    assert scales["action_smoothness"] == -0.01
    assert scales["power_distribution"] == -1.0e-5
    assert "feet_air_time" not in scales
    assert "similar_to_default" not in scales
    assert command_cfg["lin_vel_x_range"] == [-1.0, 1.0]
    assert command_cfg["lin_vel_y_range"] == [-0.5, 0.5]
    assert command_cfg["ang_vel_range"] == [-1.0, 1.0]
    assert env_cfg["speed_curriculum"] is True
    assert go2_locomotion_env.TERRAIN_KIND_FRACTIONS == (0.48, 0.20, 0.12, 0.20)
    held = [0.54] * 100
    assert go2_locomotion_env.raised_speed_cap(0, 100, held) == (0, None)
    assert go2_locomotion_env.raised_speed_cap(0, 99, [0.9] * 99) == (0, None)
    assert go2_locomotion_env.raised_speed_cap(0, 100, [0.0] * 50 + [0.9] * 50)[0] == 1
    assert go2_locomotion_env.raised_speed_cap(0, 100, [0.9] * 50 + [0.0] * 50) == (0, None)
    assert go2_locomotion_env.raised_speed_cap(4, 200, [0.9] * 200) == (4, None)
    randomization = env_cfg["randomization"]
    assert randomization["payload"] == (-1.0, 2.0)
    assert randomization["friction_coefficient"] == (0.2, 1.25)
    assert randomization["gain_scale"] == (0.9, 1.1)
    assert randomization["force_scale"] == (0.9, 1.1)
    assert randomization["com_offset"] == 0.05
    assert randomization["system_delay"] == (0.0, 0.015)
    assert randomization["mass_scale"] == (1.0, 1.0)
    assert randomization["push_xy"] == 0.0
    dream_groups = go2_locomotion_env.policy_obs_groups(env_cfg)
    assert dream_groups["actor"] == ["proprio", "est_vel", "latent"]
    assert dream_groups["critic"] == ["proprio", "lin_vel", "disturbance"]
    assert "lin_vel" not in dream_groups["actor"]
    assert "disturbance" not in dream_groups["actor"]
    assert all("height" not in group for group in dream_groups["actor"])


def test_saved_estimator_key_maps_to_dreamwaq_policy():
    env_cfg = {"estimator": "cenet"}
    assert go2_locomotion_env.is_dreamwaq_policy(env_cfg)
    env_cfg_copy = dict(env_cfg)
    assert go2_locomotion_env.canonicalize_policy_env_cfg(env_cfg_copy) == go2_locomotion_env.POLICY_DREAMWAQ
    assert env_cfg_copy["policy"] == go2_locomotion_env.POLICY_DREAMWAQ
    assert "estimator" not in env_cfg_copy


def test_ult_policy_rollout_and_update():
    pytest.importorskip("rsl_rl")
    go2_train = importlib.import_module("go2_train")
    tensordict = importlib.import_module("tensordict")
    ult = importlib.import_module("ult")

    env_cfg, _, reward_cfg, command_cfg = go2_locomotion_env.locomotion_cfgs()
    go2_locomotion_env.apply_curriculum(env_cfg, reward_cfg, command_cfg)
    env_cfg["policy"] = go2_locomotion_env.POLICY_ULT
    assert not go2_locomotion_env.is_dreamwaq_policy(env_cfg)
    groups = go2_locomotion_env.policy_obs_groups(env_cfg)
    assert groups["critic"] == ["proprio_flat", "privilege"]

    history_steps = go2_locomotion_env.ULT_HISTORY_STEPS
    mask = ult.ult_causal_attention_mask(history_steps, torch.device("cpu"))
    assert mask[:history_steps, history_steps].all()
    assert_equal(mask[:history_steps, :history_steps], torch.ones((history_steps, history_steps)).triu(1).bool())
    assert not mask[history_steps].any()

    torch.manual_seed(0)
    n_envs, n_steps = 4, 3
    train_cfg = go2_train.configure_ult_train_cfg("go2-ult-test", env_cfg)
    train_cfg["num_steps_per_env"] = n_steps
    train_cfg["multi_gpu"] = None

    obs_seq = [
        tensordict.TensorDict(
            {
                "proprio_history": torch.randn((n_envs, history_steps, go2_locomotion_env.ULT_PROPRIO_DIM)),
                "action_history": torch.randn((n_envs, history_steps, 12)),
                "privilege": torch.randn((n_envs, go2_locomotion_env.ULT_PRIVILEGE_DIM)),
                "proprio_flat": torch.randn((n_envs, go2_locomotion_env.ULT_PROPRIO_DIM)),
            },
            batch_size=[n_envs],
        )
        for _ in range(n_steps + 1)
    ]
    env = SimpleNamespace(num_envs=n_envs, num_actions=12)
    alg = ult.ULTPPO.construct_algorithm(obs_seq[0], env, train_cfg, "cpu")
    dones = torch.tensor([False, True, False, False])
    with torch.inference_mode():
        for i_step in range(n_steps):
            alg.act(obs_seq[i_step])
            alg.process_env_step(obs_seq[i_step + 1], torch.zeros(n_envs), dones, {})

    storage = alg.storage
    with torch.no_grad():
        for i_step in range(n_steps):
            step_obs = storage.observations[i_step]
            student_out, teacher_out, _ = alg.actor.forward_heads(step_obs)
            is_teacher = step_obs["ult_is_teacher"]
            alg.actor.distribution.update(torch.where(is_teacher, teacher_out, student_out))
            log_prob = alg.actor.get_output_log_prob(storage.actions[i_step])
            assert_allclose(log_prob, storage.actions_log_prob[i_step, :, 0], tol=1e-5)

            next_obs = obs_seq[i_step + 1]
            nsa_target = torch.cat((next_obs["proprio_history"][:, -1], next_obs["action_history"][:, -1]), dim=-1)
            assert_allclose(step_obs["ult_nsa_target"], nsa_target, tol=0.0)
            assert_equal(step_obs["ult_is_nsa_valid"][:, 0], ~dones)

            student_obs = step_obs.exclude("privilege")
            assert_allclose(alg.actor(student_obs), alg.actor.distribution.deterministic_output(student_out), tol=1e-5)

    losses = alg.update()
    assert all(math.isfinite(value) for value in losses.values())
    assert losses["ult_nsa"] > 0.0
    assert losses["ult_imitation"] > 0.0
