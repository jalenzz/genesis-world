import importlib
from pathlib import Path

import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.fixture
def locomotion(monkeypatch):
    pytest.importorskip("rsl_rl")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "locomotion"))
    return importlib.import_module("go2_env"), importlib.import_module("cenet")


def test_cenet_beta_scales_the_divergence(locomotion):
    _, cenet = locomotion
    proprio_dim = 45
    torch.manual_seed(0)
    history = torch.randn(8, cenet.HISTORY_STEPS, proprio_dim)
    velocity = torch.randn(8, 3)
    next_proprio = history[:, -1] + 0.1 * history[:, 0]
    net = cenet.CENet(proprio_dim)
    assert_allclose(net.beta, 2.0 / proprio_dim, atol=0.0, rtol=0.0)
    free = cenet.CENet(proprio_dim, beta=0.0)
    free.load_state_dict(net.state_dict())
    net.eval()
    free.eval()
    weighted = net.loss(history, velocity, next_proprio)
    unweighted = free.loss(history, velocity, next_proprio)
    assert (weighted > unweighted).all()

    reconstructions = []
    latent_means = []
    for beta in (2.0 / proprio_dim, 100.0):
        generator = torch.Generator().manual_seed(1)
        data_history = torch.randn(64, cenet.HISTORY_STEPS, proprio_dim, generator=generator)
        data_next = data_history[:, -1] + 0.1 * data_history[:, 0]
        data_velocity = data_next[:, :3]
        torch.manual_seed(0)
        model = cenet.CENet(proprio_dim, beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
        model.train()
        for _ in range(60):
            optimizer.zero_grad()
            loss = model.loss(data_history, data_velocity, data_next)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            belief = model.encode(data_history)
            reconstructed = model.decode(belief.mean)
            target = model.normalizer(data_next)
            reconstructions.append(torch.mean(torch.square(reconstructed - target)))
            latent_means.append(belief.mean.detach().abs().mean())
    assert (reconstructions[0] < reconstructions[1]).all()
    assert (latent_means[0] > latent_means[1]).all()


def test_adaboot_probability(locomotion):
    _, cenet = locomotion
    certain = cenet.adaboot_probability(torch.tensor(1.0), torch.tensor(0.0))
    assert_allclose(certain, 1.0, atol=0.0, rtol=0.0)
    mean = torch.tensor(-2.0)
    std = torch.tensor(2.0)
    expected = 1.0 - torch.tanh(std / mean.abs())
    assert_allclose(cenet.adaboot_probability(mean, std), expected, atol=1.0e-6, rtol=0.0)


def test_curriculum_config(locomotion):
    go2_env, _cenet = locomotion
    full_field = go2_env.build_curriculum_height_field(
        n_levels=10,
        subterrain_size=(4.0, 4.0),
        horizontal_scale=0.5,
        vertical_scale=0.01,
        max_slope_deg=22.0,
    )
    kept_field = go2_env.build_curriculum_height_field(
        n_levels=3,
        subterrain_size=(4.0, 4.0),
        horizontal_scale=0.5,
        vertical_scale=0.01,
        max_slope_deg=22.0,
        difficulty_levels=10,
    )
    assert kept_field.shape[0] < full_field.shape[0]
    assert kept_field.shape[1] == full_field.shape[1]
    assert_equal(kept_field[:-1], full_field[: kept_field.shape[0] - 1])

    env_cfg, _obs_cfg, reward_cfg, command_cfg = go2_env.locomotion_cfgs()
    assert_allclose(reward_cfg["reward_scales"]["tracking_ang_vel"], 0.2, atol=0.0, rtol=0.0)
    assert_allclose(command_cfg["lin_vel_x_range"], [0.5, 0.5], atol=0.0, rtol=0.0)
    flat_groups = go2_env.policy_obs_groups(env_cfg)
    assert flat_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert not go2_env.is_dreamwaq_policy(env_cfg)

    go2_env.apply_curriculum(env_cfg, reward_cfg, command_cfg)
    assert env_cfg["terrain"] == "curriculum"
    assert "termination_if_roll_greater_than" not in env_cfg
    assert "termination_if_pitch_greater_than" not in env_cfg
    assert env_cfg["simulate_action_latency"] is False
    assert not go2_env.is_dreamwaq_policy(env_cfg)
    env_cfg["policy"] = go2_env.POLICY_DREAMWAQ
    assert go2_env.is_dreamwaq_policy(env_cfg)
    scales = reward_cfg["reward_scales"]
    assert_allclose(scales["tracking_lin_vel"], 1.0, atol=0.0, rtol=0.0)
    assert_allclose(scales["tracking_ang_vel"], 0.5, atol=0.0, rtol=0.0)
    assert_allclose(scales["lin_vel_z"], -2.0, atol=0.0, rtol=0.0)
    assert_allclose(scales["feet_clearance"], -0.01, atol=0.0, rtol=0.0)
    assert "feet_air_time" not in scales
    assert "similar_to_default" not in scales
    assert_allclose(command_cfg["lin_vel_x_range"], [-1.0, 1.0], atol=0.0, rtol=0.0)
    assert_allclose(command_cfg["lin_vel_y_range"], [-0.5, 0.5], atol=0.0, rtol=0.0)
    assert_allclose(command_cfg["ang_vel_range"], [-1.0, 1.0], atol=0.0, rtol=0.0)
    for fraction in go2_env.TERRAIN_KIND_FRACTIONS:
        assert fraction > 0.0
    assert_allclose(sum(go2_env.TERRAIN_KIND_FRACTIONS), 1.0, atol=1.0e-6, rtol=0.0)
    assert go2_env.raised_speed_cap(0, 100, [0.54] * 100) == (0, None)
    assert go2_env.raised_speed_cap(0, 99, [0.9] * 99) == (0, None)
    assert go2_env.raised_speed_cap(0, 100, [0.0] * 50 + [0.9] * 50)[0] == 1
    assert go2_env.raised_speed_cap(4, 200, [0.9] * 200) == (4, None)
    randomization = env_cfg["randomization"]
    assert_allclose(randomization["payload"], (-1.0, 2.0), atol=0.0, rtol=0.0)
    assert_allclose(randomization["friction_coefficient"], (0.2, 1.25), atol=0.0, rtol=0.0)
    assert_allclose(randomization["gain_scale"], (0.9, 1.1), atol=0.0, rtol=0.0)
    assert_allclose(randomization["force_scale"], (0.9, 1.1), atol=0.0, rtol=0.0)
    assert_allclose(randomization["com_offset"], 0.05, atol=0.0, rtol=0.0)
    assert_allclose(randomization["system_delay"], (0.0, 0.015), atol=0.0, rtol=0.0)
    assert "push_xy" not in randomization
    assert_allclose(go2_env.DEFAULT_RANDOMIZATION.push_xy, 20.0, atol=0.0, rtol=0.0)
    assert_allclose(go2_env.DEFAULT_RANDOMIZATION.push_z, 5.0, atol=0.0, rtol=0.0)
    groups = go2_env.dreamwaq_obs_groups()
    assert groups["actor"] == ["proprio", "est_vel", "latent"]
    assert groups["critic"] == ["proprio", "lin_vel", "disturbance", "height_scan"]
    assert len(go2_env.HEIGHT_SCAN_X) * len(go2_env.HEIGHT_SCAN_Y) == 17 * 11


@pytest.mark.parametrize("n_envs", [0, 2])
def test_dreamwaq_terrain_rows(n_envs, show_viewer, locomotion):
    go2_env, cenet = locomotion
    subterrain_size = (4.0, 4.0)
    env_cfg, obs_cfg, reward_cfg, command_cfg = go2_env.locomotion_cfgs()
    go2_env.apply_curriculum(env_cfg, reward_cfg, command_cfg)
    env_cfg["policy"] = go2_env.POLICY_DREAMWAQ
    env_cfg["episode_length_s"] = 20.0
    env_cfg["domain_randomization"] = False
    env_cfg["terrain_subterrain_size"] = subterrain_size
    env_cfg["terrain_horizontal_scale"] = 0.5
    env_cfg["terrain_vertical_scale"] = 0.01
    env_cfg["curriculum_up"] = 0.5
    command_cfg["lin_vel_x_range"] = [0.0, 0.0]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]
    if n_envs == 0:
        env_cfg["max_collision_pairs"] = 2
        overflow = go2_env.Go2Env(
            num_envs=n_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
        held = torch.zeros((overflow.batch_size, overflow.num_actions), dtype=gs.tc_float, device=gs.device)
        with pytest.raises(gs.GenesisException, match="Exceeding max number"):
            overflow.step(held)
        overflow.scene.destroy()

    env_cfg["max_collision_pairs"] = 256
    env = go2_env.Go2Env(
        num_envs=n_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )
    assert env.scene.rigid_solver.max_collision_pairs == 256

    env.terrain_levels.fill_(1)
    env.terrain_types.zero_()
    obs = env.reset()
    scan_points = len(go2_env.HEIGHT_SCAN_X) * len(go2_env.HEIGHT_SCAN_Y)
    assert obs["proprio"].shape[-1] == 45
    assert obs["est_vel"].shape[-1] == 3
    assert obs["latent"].shape[-1] == cenet.LATENT_DIM
    assert obs["height_scan"].shape[-1] == scan_points
    assert tuple(obs["history"].shape[-2:]) == (cenet.HISTORY_STEPS, 45)
    groups = go2_env.dreamwaq_obs_groups()
    critic = torch.cat([obs[name] for name in groups["critic"]], dim=-1)
    assert critic.shape[-1] == 45 + 3 + 3 + scan_points

    state = env.cenet_state_dict()
    state["model"]["velocity_head.bias"].fill_(1.0)
    state["model"]["mean_head.bias"].fill_(1.0)
    env.load_cenet_state_dict(state)
    estimated = env.get_observations()
    estimated_vel = estimated["est_vel"].clone()
    estimated_latent = estimated["latent"].clone()
    env.set_estimator_ground_truth(True)
    measured = env.get_observations()
    assert_allclose(measured["latent"], estimated_latent, atol=1.0e-5, rtol=0.0)
    assert_allclose(measured["est_vel"], measured["lin_vel"], atol=1.0e-5, rtol=0.0)
    assert ((estimated_vel - measured["lin_vel"]).abs().max() > 1.0e-3).all()
    assert (measured["latent"].abs().max() > 1.0e-3).all()
    env.set_estimator_ground_truth(False)

    actions = torch.zeros((env.batch_size, env.num_actions), dtype=gs.tc_float, device=gs.device)
    env.step(actions)
    held_history = env.get_observations()["history"].clone()
    commanded = env.set_commands(torch.tensor([0.4, -0.2, 0.3], dtype=gs.tc_float, device=gs.device))
    assert_allclose(commanded["history"][:, :-1], held_history[:, :-1], atol=0.0, rtol=0.0)
    assert ((commanded["history"][:, -1] - held_history[:, -1]).abs().max() > 1.0e-3).all()
    env.set_commands(torch.zeros(3, dtype=gs.tc_float, device=gs.device))

    # The base sits on the height-field cell that straddles a stair step, level with the world.
    stair_kind = go2_env.TERRAIN_KINDS.index("stairs")
    row = env.n_levels - 1
    x0 = row * subterrain_size[0]
    y_mid = (stair_kind + 0.5) * subterrain_size[1]
    samples = torch.linspace(x0, x0 + subterrain_size[0], 17, device=gs.device)
    xy = torch.stack((samples, torch.full_like(samples, y_mid)), dim=-1)
    heights = env.terrain_height(xy.unsqueeze(0).expand(env.batch_size, -1, -1))[0]
    jumps = heights[1:] - heights[:-1]
    assert (jumps.abs().max() > 0.05).all()
    edge = int(torch.argmax(jumps.abs()))
    xy_edge = 0.5 * (xy[edge] + xy[edge + 1])
    ground = torch.maximum(heights[edge], heights[edge + 1])
    qpos = env.robot.get_qpos().clone()
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
    base_z = ground + env.init_base_pos[2]
    if qpos.ndim == 1:
        qpos[0] = xy_edge[0]
        qpos[1] = xy_edge[1]
        qpos[2] = base_z
        qpos[3:7] = identity
    else:
        qpos[:, 0] = xy_edge[0]
        qpos[:, 1] = xy_edge[1]
        qpos[:, 2] = base_z
        qpos[:, 3:7] = identity
    env.robot.set_qpos(qpos, zero_velocity=True)
    env.step(actions)
    assert not bool(env.is_fallen.any())
    assert not bool(env.reset_buf.any())

    # Close the stair episode while the horizon is still long, then shorten it. The following reset sees no lived steps.
    env.reset()
    env.max_episode_length = 1
    if n_envs == 0:
        env.terrain_levels.fill_(0)
        env.terrain_types.zero_()
        env.reset()
        env.step(actions)
        env.step(actions)
        assert_equal(env.terrain_levels, 1)
        env.terrain_levels.fill_(1)
        env.reset()
        buried = env.robot.get_qpos().clone()
        buried[2] = env.terrain_height(env.base_pos[:, :2])[0]
        env.robot.set_qpos(buried, zero_velocity=True)
        env.step(actions)
        assert_equal(env.terrain_levels, 0)
    else:
        env.terrain_levels.copy_(torch.tensor([0, 1], dtype=gs.tc_int, device=gs.device))
        env.terrain_types.zero_()
        env.reset()
        buried = env.robot.get_qpos().clone()
        buried[1, 2] = env.terrain_height(env.base_pos[:, :2])[1]
        env.robot.set_qpos(buried, zero_velocity=True)
        env.step(actions)
        assert_equal(env.terrain_levels, torch.tensor([0, 0], dtype=gs.tc_int, device=gs.device))
        env.step(actions)
        assert_equal(env.terrain_levels, torch.tensor([1, 0], dtype=gs.tc_int, device=gs.device))

    env.terrain_levels.fill_(3)
    env.episode_length_buf.fill_(env.max_episode_length + 4)
    env.step(actions)
    assert_equal(env.terrain_levels, 3)

    saved = env.curriculum_state
    restored_levels = saved.terrain_levels.clone()
    env.load_curriculum_state(
        go2_env.CurriculumState(
            speed_level=2,
            speed_iters_at_cap=4,
            speed_tracking_means=[0.8],
            terrain_levels=restored_levels,
            promote_streak=saved.promote_streak.clone(),
        ),
        load_terrain=False,
    )
    assert_allclose(env.speed_vx_cap, go2_env.SPEED_VX_CAPS[2], atol=0.0, rtol=0.0)
    assert_equal(env.terrain_levels, restored_levels)
    env.terrain_levels.zero_()
    env.load_curriculum_state(saved, load_terrain=True)
    assert_equal(env.terrain_levels, restored_levels)
    with pytest.raises(gs.GenesisException, match="terrain rows"):
        env.load_curriculum_state(
            go2_env.CurriculumState(
                speed_level=0,
                speed_iters_at_cap=0,
                speed_tracking_means=[],
                terrain_levels=torch.zeros((3,), dtype=gs.tc_int, device=gs.device),
                promote_streak=torch.zeros((3,), dtype=gs.tc_int, device=gs.device),
            ),
            load_terrain=True,
        )

    env.max_episode_length = 1000
    env.reset()
    env.episode_length_buf.zero_()
    env._episode_steps.zero_()
    motor_velocity = torch.zeros((env.batch_size, env.num_actions), dtype=gs.tc_float, device=gs.device)
    motor_velocity[0] = 1.0e20
    env.robot.set_dofs_velocity(motor_velocity if n_envs > 0 else motor_velocity[0], env.motors_dof_idx)
    env.dof_vel[0] = 101.0
    if n_envs > 0:
        env.dof_vel[1] = 100.0
        env.base_lin_vel[1, 0] = 20.0
        env.base_ang_vel[1, 0] = 100.0
    env.step(actions)
    assert bool(env.reset_buf[0])
    if n_envs > 0:
        assert not bool(env.reset_buf[1])
    assert torch.isfinite(env.robot.get_dofs_velocity()).all()
