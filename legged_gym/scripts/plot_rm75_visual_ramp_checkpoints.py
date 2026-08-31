"""Plot RM75 visual-trot telemetry on flat, uphill, and downhill ramp sections."""

from pathlib import Path

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry


DEFAULT_CHECKPOINTS = (5000, 10000, 15000, 20000, 25000, 30000)
TASKS = {
    25: {
        "task": "RM75_visual_ramp_trot_3ms_25deg",
        "experiment": "RM75_visual_ramp_trot_3ms_25deg",
        "run": "Aug26_15-26-47_warmstart_model25000",
    },
    35: {
        "task": "RM75_visual_ramp_trot_3ms_35deg_hip100_knee160",
        "experiment": "RM75_visual_ramp_trot_3ms_35deg_hip100_knee160",
        "run": "Aug28_10-22-28_warmstart_torque_model30000",
    },
}
POLICY_PROFILES = {
    "torque_limited_trot": {
        "task": "RM75_flat_trot_3ms_hip100_knee160",
        "experiment": "RM75_flat_trot_3ms_hip100_knee160",
        "run": "Aug27_14-56-04_warmstart_earth_model25000",
        "checkpoints": {
            checkpoint: (
                "RM75_flat_trot_3ms_hip100_knee160",
                "Aug27_14-56-04_warmstart_earth_model25000",
                checkpoint,
            )
            for checkpoint in DEFAULT_CHECKPOINTS
        },
    },
    "lunar_1_6g_trot": {
        "task": "RM75_lunar_flat_trot_3ms_recovery",
        "experiment": "RM75_lunar_flat_trot_3ms_recovery",
        "run": "Aug27_17-21-13_warmstart_lunar_model12000",
        # The labels are total-equivalent progress. Recovery checkpoints use a
        # fresh local counter after warm-starting from the original model_12000.
        "checkpoints": {
            5000: (
                "RM75_lunar_flat_trot_3ms",
                "Aug27_14-06-43_warmstart_earth_model25000",
                5000,
            ),
            10000: (
                "RM75_lunar_flat_trot_3ms",
                "Aug27_14-06-43_warmstart_earth_model25000",
                10000,
            ),
            15000: (
                "RM75_lunar_flat_trot_3ms_recovery",
                "Aug27_17-21-13_warmstart_lunar_model12000",
                3000,
            ),
            20000: (
                "RM75_lunar_flat_trot_3ms_recovery",
                "Aug27_17-21-13_warmstart_lunar_model12000",
                8000,
            ),
            25000: (
                "RM75_lunar_flat_trot_3ms_recovery",
                "Aug27_17-21-13_warmstart_lunar_model12000",
                13000,
            ),
            30000: (
                "RM75_lunar_flat_trot_3ms_recovery",
                "Aug27_17-21-13_warmstart_lunar_model12000",
                18000,
            ),
        },
    },
}
LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_GROUPS = {
    "hip": tuple(f"{leg}_hip_joint" for leg in LEG_ORDER),
    "thigh": tuple(f"{leg}_thigh_joint" for leg in LEG_ORDER),
    "calf": tuple(f"{leg}_calf_joint" for leg in LEG_ORDER),
}
PHASE_LOCAL_X = {
    "flat": 0.75,
    "uphill": 11.0,
    "downhill": 31.0,
}
PHASE_LABELS = {
    "flat": "Flat Ground",
    "uphill": "Uphill",
    "downhill": "Downhill",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def parse_args():
    args = get_args(
        additional_parameters=[
            {
                "name": "--slope_angle",
                "type": int,
                "default": 25,
                "help": "Maximum ramp angle to evaluate: 25 or 35 degrees.",
            },
            {
                "name": "--policy_profile",
                "type": str,
                "default": None,
                "help": (
                    "Optional non-visual policy profile: "
                    "torque_limited_trot or lunar_1_6g_trot."
                ),
            },
            {
                "name": "--flat_only",
                "action": "store_true",
                "default": False,
                "help": "Evaluate only the policy's native flat-ground environment.",
            },
            {
                "name": "--checkpoints",
                "type": str,
                "default": ",".join(str(value) for value in DEFAULT_CHECKPOINTS),
                "help": "Comma-separated checkpoint iterations to evaluate.",
            },
            {
                "name": "--output_dir",
                "type": str,
                "default": None,
                "help": "Output directory. Defaults to good_result/<slope_angle>.",
            },
            {
                "name": "--warmup_seconds",
                "type": float,
                "default": 0.4,
                "help": "Warm-up time at each terrain phase before recording.",
            },
            {
                "name": "--sample_seconds",
                "type": float,
                "default": 60.0,
                "help": "Maximum telemetry duration for each terrain phase.",
            },
            {
                "name": "--plot_dpi",
                "type": int,
                "default": 180,
                "help": "PNG resolution in dots per inch.",
            },
        ]
    )
    if args.slope_angle not in TASKS:
        raise ValueError(f"slope_angle must be one of {tuple(TASKS)}, got {args.slope_angle}")
    if args.policy_profile is not None and args.policy_profile not in POLICY_PROFILES:
        raise ValueError(
            f"policy_profile must be one of {tuple(POLICY_PROFILES)}, "
            f"got {args.policy_profile}"
        )
    if args.policy_profile is None:
        args.task = TASKS[args.slope_angle]["task"]
    else:
        args.task = POLICY_PROFILES[args.policy_profile]["task"]
    args.num_envs = 1
    args.headless = True
    return args


def parse_checkpoints(raw_value):
    checkpoints = tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())
    if not checkpoints or any(value <= 0 for value in checkpoints):
        raise ValueError(f"Invalid checkpoints: {checkpoints}")
    return checkpoints


def configure_evaluation(
    env_cfg,
    angle,
    total_seconds,
    measure_heights=True,
    flat_only=False,
):
    env_cfg.env.num_envs = 1
    env_cfg.env.test = False
    env_cfg.env.episode_length_s = max(30.0, total_seconds + 2.0)
    env_cfg.env.enable_camera_sensors = False
    env_cfg.init_state.turn_over = False

    env_cfg.terrain.mesh_type = "plane" if flat_only else "trimesh"
    env_cfg.terrain.terrain_style = "mgdp_parkour"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    # The standard 45-D proprioceptive policies were trained with one scalar
    # critic height value. Enabling the 187-point scan would silently change
    # their privileged-observation shape during evaluation.
    env_cfg.terrain.measure_heights = measure_heights
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_length = 45.0
    env_cfg.terrain.terrain_width = 5.0
    env_cfg.terrain.ramp_segment_length = 10.0
    env_cfg.terrain.start_platform_length = 10.0
    env_cfg.terrain.start_platform_width = 2.4
    env_cfg.terrain.ramp_max_angle_deg = float(angle)
    env_cfg.terrain.mgdp_max_difficulty = 0.9
    env_cfg.terrain.mgdp_add_roughness = False
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.reset_when_outside_block = False
    env_cfg.terrain.random_section_spawn_start_iter = 10**9
    env_cfg.terrain.random_section_spawn_probability = 0.0
    env_cfg.terrain.terrain_proportions = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    env_cfg.commands.curriculum = False
    env_cfg.commands.command_range_curriculum = []
    env_cfg.commands.resampling_time = total_seconds + 20.0
    for name in ("lin_vel_x", "new_lin_vel_x"):
        if hasattr(env_cfg.commands.ranges, name):
            setattr(env_cfg.commands.ranges, name, [3.0, 3.0])
    for name in (
        "lin_vel_y",
        "ang_vel_yaw",
        "heading",
        "new_lin_vel_y",
        "new_ang_vel_yaw",
        "new_heading",
    ):
        if hasattr(env_cfg.commands.ranges, name):
            setattr(env_cfg.commands.ranges, name, [0.0, 0.0])

    env_cfg.noise.add_noise = False
    deterministic_flags = (
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "randomize_link_mass",
        "randomize_base_com",
        "randomize_pd_gains",
        "randomize_motor_zero_offset",
        "randomize_motor_strength",
        "randomize_action_delay",
        "push_robots",
    )
    for flag_name in deterministic_flags:
        if hasattr(env_cfg.domain_rand, flag_name):
            setattr(env_cfg.domain_rand, flag_name, False)
    env_cfg.rewards.no_progress_timeout_s = 0.0

    env_cfg.camera.xyz_error = [0.0, 0.0]
    env_cfg.camera.yaw_error_deg = [0.0, 0.0]
    env_cfg.camera.noise_gaussian = 0.0
    env_cfg.camera.noise_dropout = 0.0


def set_fixed_command(env):
    env.commands[:, :3] = torch.tensor(
        [3.0, 0.0, 0.0], dtype=torch.float, device=env.device
    )
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0
    env.commands_resampling_step[:] = env.cfg.commands.resampling_time / env.dt


def get_mass_properties(env):
    properties = env.gym.get_actor_rigid_body_properties(
        env.envs[0], env.actor_handles[0]
    )
    masses = torch.tensor(
        [prop.mass for prop in properties], dtype=torch.float, device=env.device
    )
    local_com = torch.tensor(
        [[prop.com.x, prop.com.y, prop.com.z] for prop in properties],
        dtype=torch.float,
        device=env.device,
    )
    return masses, local_com


def get_center_of_mass_state(env, masses, local_com):
    body_states = env.rigid_body_states.view(env.num_envs, env.num_bodies, 13)[0]
    body_com_positions = body_states[:, 0:3] + quat_apply(
        body_states[:, 3:7], local_com
    )
    total_mass = masses.sum()
    center_position = (body_com_positions * masses.unsqueeze(1)).sum(dim=0) / total_mass
    center_velocity = (body_states[:, 7:10] * masses.unsqueeze(1)).sum(dim=0) / total_mass
    return center_position, center_velocity


def sample_heightfield(env, xy_world):
    """Sample the collision height field at arbitrary world XY positions."""
    points = xy_world + float(env.terrain.cfg.border_size)
    points = torch.floor(points / float(env.terrain.cfg.horizontal_scale)).long()
    px = torch.clip(points[:, 0], 0, env.height_samples.shape[0] - 2)
    py = torch.clip(points[:, 1], 0, env.height_samples.shape[1] - 2)
    heights = torch.minimum(
        env.height_samples[px, py],
        env.height_samples[px + 1, py],
    )
    heights = torch.minimum(heights, env.height_samples[px, py + 1])
    return heights * float(env.terrain.cfg.vertical_scale)


def sample_height_and_slope(env, xy_world, half_span=0.20):
    if hasattr(env, "_sample_heightfield") and hasattr(env, "_sample_longitudinal_slope"):
        return (
            env._sample_heightfield(xy_world),
            env._sample_longitudinal_slope(xy_world, half_span=half_span),
        )

    center_height = sample_heightfield(env, xy_world)
    backward = xy_world.clone()
    forward = xy_world.clone()
    backward[:, 0] -= half_span
    forward[:, 0] += half_span
    slope = (sample_heightfield(env, forward) - sample_heightfield(env, backward)) / (
        2.0 * half_span
    )
    return center_height, slope


def resolve_joint_groups(dof_names):
    missing = [
        name
        for names in JOINT_GROUPS.values()
        for name in names
        if name not in dof_names
    ]
    if missing:
        raise RuntimeError(f"Missing RM75 joints: {missing}; available={dof_names}")
    return {
        group: np.asarray([dof_names.index(name) for name in names], dtype=int)
        for group, names in JOINT_GROUPS.items()
    }


def reset_runner_inference_state(runner, env):
    dones = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    estimator = getattr(runner.alg, "estimator", None)
    if estimator is not None and hasattr(estimator, "reset"):
        estimator.reset(dones)
    if hasattr(runner, "history"):
        runner.history.zero_()
    if hasattr(runner, "_reset_swav_history"):
        runner._reset_swav_history(dones)
    runner.latest_inference_mcp_code = None
    runner.latest_inference_m_hat = None
    runner.latest_inference_gating_weights = None
    runner.latest_inference_token_dropout_mask = None
    runner.latest_inference_token_grid_shape = None
    runner.latest_inference_active_token_coords = None
    runner.latest_inference_active_token_valid_mass = None


def place_robot_at_phase(env, phase, checkpoint):
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    env.reset_idx(env_ids)
    env.common_step_counter = checkpoint * env.num_steps_per_env

    if env.cfg.terrain.mesh_type == "plane":
        world_x = env.env_origins[0, 0]
        world_y = env.env_origins[0, 1]
        ground_height = torch.zeros(1, dtype=torch.float, device=env.device)
        slope = torch.zeros_like(ground_height)
    else:
        block_start_x = env.env_origins[0, 0] - 0.5 * float(env.cfg.terrain.terrain_length)
        world_x = block_start_x + float(PHASE_LOCAL_X[phase])
        world_y = env.env_origins[0, 1]
        xy_world = torch.stack((world_x, world_y)).reshape(1, 2)
        ground_height, slope = sample_height_and_slope(env, xy_world)
    normal_z = torch.rsqrt(1.0 + torch.square(slope))

    env.root_states[0] = env.base_init_state
    env.root_states[0, 0] = world_x
    env.root_states[0, 1] = world_y
    env.root_states[0, 2] = ground_height[0] + (
        float(env.cfg.rewards.base_height_target) / normal_z[0]
    )
    zero = torch.zeros_like(slope)
    env.root_states[0, 3:7] = quat_from_euler_xyz(
        zero, -torch.atan(slope), zero
    )[0]
    env.root_states[0, 7:13] = 0.0
    env.dof_pos[0] = env.default_dof_pos[0]
    env.dof_vel[0] = 0.0
    env.actions[0] = 0.0
    env.last_actions[0] = 0.0
    env.last_dof_vel[0] = 0.0
    env.episode_length_buf[0] = 0
    env.reset_buf[0] = 0
    env.time_out_buf[0] = False

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        1,
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        1,
    )
    set_fixed_command(env)
    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, dtype=torch.float, device=env.device
    )
    obs, _, _, _, _ = env.step(zero_actions)
    set_fixed_command(env)
    env.reset_buf[0] = 1
    return obs


def collect_phase(
    env,
    runner,
    policy,
    checkpoint,
    phase,
    warmup_steps,
    sample_steps,
    masses,
    local_com,
):
    warmup_failed = False
    rollout_restarts = 0
    attempted_steps = 0
    joint_velocity = []
    torque = []
    power = []
    com_position = []
    com_velocity = []
    foot_contact_force = []

    def phase_is_active():
        if env.cfg.terrain.mesh_type == "plane":
            return phase == "flat"
        xy_world = env.root_states[0, :2].reshape(1, 2)
        _, slope = sample_height_and_slope(env, xy_world)
        threshold = np.tan(np.deg2rad(5.0))
        slope_value = float(slope[0])
        if phase == "uphill":
            return slope_value > threshold
        if phase == "downhill":
            return slope_value < -threshold
        return abs(slope_value) <= threshold

    with torch.no_grad():
        while len(joint_velocity) < sample_steps:
            reset_runner_inference_state(runner, env)
            obs = place_robot_at_phase(env, phase, checkpoint)
            cycle_failed = False
            for _ in range(warmup_steps):
                set_fixed_command(env)
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())
                attempted_steps += 1
                if bool(dones[0]) or not phase_is_active():
                    cycle_failed = True
                    warmup_failed = True
                    break
            if cycle_failed:
                rollout_restarts += 1
                if attempted_steps > sample_steps * 20:
                    break
                continue

            samples_before_cycle = len(joint_velocity)
            while len(joint_velocity) < sample_steps:
                set_fixed_command(env)
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())
                attempted_steps += 1
                if bool(dones[0]) or not phase_is_active():
                    break

                dof_velocity = env.dof_vel[0].detach()
                applied_torque = env.torques[0].detach()
                center_position, center_velocity = get_center_of_mass_state(
                    env, masses, local_com
                )
                contact_force = torch.norm(
                    env.contact_forces[0, env.trot_feet_indices, :], dim=-1
                )
                joint_velocity.append(dof_velocity.cpu().numpy().copy())
                torque.append(applied_torque.cpu().numpy().copy())
                power.append((applied_torque * dof_velocity).cpu().numpy().copy())
                com_position.append(center_position.detach().cpu().numpy().copy())
                com_velocity.append(center_velocity.detach().cpu().numpy().copy())
                foot_contact_force.append(contact_force.detach().cpu().numpy().copy())

            if len(joint_velocity) < sample_steps:
                rollout_restarts += 1
            if len(joint_velocity) == samples_before_cycle and attempted_steps > sample_steps * 20:
                break

    count = len(joint_velocity)
    time = (np.arange(count, dtype=np.float64) + 1.0) * env.dt
    empty_joint = np.empty((0, env.num_actions), dtype=np.float32)
    empty_com = np.empty((0, 3), dtype=np.float32)
    empty_feet = np.empty((0, 4), dtype=np.float32)
    return {
        "checkpoint": checkpoint,
        "phase": phase,
        "time": time,
        "joint_velocity": np.asarray(joint_velocity) if count else empty_joint,
        "torque": np.asarray(torque) if count else empty_joint.copy(),
        "power": np.asarray(power) if count else empty_joint.copy(),
        "com_position": np.asarray(com_position) if count else empty_com,
        "com_velocity": np.asarray(com_velocity) if count else empty_com.copy(),
        "foot_contact_force": (
            np.asarray(foot_contact_force) if count else empty_feet
        ),
        "terminated_early": count < sample_steps,
        "warmup_failed": warmup_failed,
        "rollout_restarts": rollout_restarts,
        "requested_samples": sample_steps,
    }


def padded_limits(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lower = float(finite.min())
    upper = float(finite.max())
    padding = max(0.1, 0.06 * (upper - lower))
    return lower - padding, upper + padding


def build_limits(all_data, joint_groups):
    limits = {}
    for phase in PHASE_LOCAL_X:
        phase_data = [data for data in all_data if data["phase"] == phase]
        for metric in ("joint_velocity", "torque", "power"):
            for group, indices in joint_groups.items():
                arrays = [
                    data[metric][:, indices].reshape(-1)
                    for data in phase_data
                    if data[metric].size
                ]
                limits[(phase, metric, group)] = padded_limits(
                    np.concatenate(arrays) if arrays else np.asarray([])
                )
        for metric in ("com_position", "com_velocity"):
            for component in range(3):
                arrays = [
                    data[metric][:, component]
                    for data in phase_data
                    if data[metric].size
                ]
                limits[(phase, metric, component)] = padded_limits(
                    np.concatenate(arrays) if arrays else np.asarray([])
                )
        contact_arrays = [
            data["foot_contact_force"].reshape(-1)
            for data in phase_data
            if data["foot_contact_force"].size
        ]
        limits[(phase, "foot_contact_force")] = padded_limits(
            np.concatenate(contact_arrays) if contact_arrays else np.asarray([])
        )
    return limits


def style_axis(ax, ylabel, time_limit, y_limit):
    ax.set_xlim(time_limit)
    ax.set_ylim(y_limit)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.axhline(0.0, color="0.45", linewidth=0.7)


def add_no_data_message(ax, data):
    if data["time"].size:
        return False
    ax.text(
        0.5,
        0.5,
        "No valid samples: policy terminated immediately",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    return True


def plot_joint_group(
    data,
    angle,
    metric,
    group,
    indices,
    ylabel,
    title,
    output_path,
    y_limit,
    dpi,
    sample_seconds,
):
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    if not add_no_data_message(ax, data):
        for leg, color, index in zip(LEG_ORDER, COLORS, indices):
            ax.plot(
                data["time"],
                data[metric][:, index],
                label=f"{leg} {group}",
                color=color,
                linewidth=1.25,
            )
        ax.legend(ncol=4, loc="upper center", frameon=False)
    end_time = max(sample_seconds, float(data["time"][-1]) if data["time"].size else 0.0)
    style_axis(ax, ylabel, (0.0, end_time), y_limit)
    status = " | terminated early" if data["terminated_early"] else ""
    ax.set_title(
        f"{angle} deg | Checkpoint {data['checkpoint']} | "
        f"{PHASE_LABELS[data['phase']]} | {title} | {group.capitalize()}{status}"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_com(
    data,
    angle,
    metric,
    labels,
    ylabel,
    title,
    output_path,
    y_limits,
    dpi,
    sample_seconds,
):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True, constrained_layout=True)
    end_time = max(sample_seconds, float(data["time"][-1]) if data["time"].size else 0.0)
    for component, (ax, label, color, y_limit) in enumerate(
        zip(axes, labels, COLORS[:3], y_limits)
    ):
        if not add_no_data_message(ax, data):
            ax.plot(
                data["time"],
                data[metric][:, component],
                label=label,
                color=color,
                linewidth=1.3,
            )
            ax.legend(loc="upper right", frameon=False)
        style_axis(ax, ylabel, (0.0, end_time), y_limit)
    status = " | terminated early" if data["terminated_early"] else ""
    axes[0].set_title(
        f"{angle} deg | Checkpoint {data['checkpoint']} | "
        f"{PHASE_LABELS[data['phase']]} | {title}{status}"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_foot_contact_force(data, angle, output_path, y_limit, dpi, sample_seconds):
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    if not add_no_data_message(ax, data):
        for foot_index, (leg, color) in enumerate(zip(LEG_ORDER, COLORS)):
            ax.plot(
                data["time"],
                data["foot_contact_force"][:, foot_index],
                label=f"{leg} foot",
                color=color,
                linewidth=1.15,
            )
        ax.legend(ncol=4, loc="upper center", frameon=False)
        finite_force = data["foot_contact_force"][
            np.isfinite(data["foot_contact_force"])
        ]
        if finite_force.size:
            peak_force = float(finite_force.max())
            readable_top = max(1.0, 1.08 * float(np.percentile(finite_force, 99.9)))
            y_limit = (-0.03 * readable_top, readable_top)
            if peak_force > readable_top:
                ax.text(
                    0.995,
                    0.93,
                    f"Peak {peak_force:.1f} N (above display range)",
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="0.30",
                )
    end_time = max(sample_seconds, float(data["time"][-1]) if data["time"].size else 0.0)
    style_axis(ax, "Contact force magnitude (N)", (0.0, end_time), y_limit)
    status = " | incomplete" if data["terminated_early"] else ""
    ax.set_title(
        f"{angle} deg | Checkpoint {data['checkpoint']} | "
        f"{PHASE_LABELS[data['phase']]} | Foot-Ground Contact Force{status}"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_all_plots(
    all_data,
    joint_groups,
    output_dir,
    angle,
    dpi,
    sample_seconds,
    flat_only=False,
):
    limits = build_limits(all_data, joint_groups)
    joint_metrics = (
        ("joint_velocity", "Joint velocity (rad/s)", "Joint Velocity"),
        ("torque", "Torque (N m)", "Joint Torque"),
        ("power", "Signed power (W)", "Instantaneous Mechanical Power"),
    )
    for data in all_data:
        phase = data["phase"]
        phase_dir = output_dir / f"checkpoint_{data['checkpoint']}"
        if not flat_only:
            phase_dir = phase_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        for metric, ylabel, title in joint_metrics:
            for group, indices in joint_groups.items():
                plot_joint_group(
                    data,
                    angle,
                    metric,
                    group,
                    indices,
                    ylabel,
                    title,
                    phase_dir / f"{metric}_{group}.png",
                    limits[(phase, metric, group)],
                    dpi,
                    sample_seconds,
                )
        plot_com(
            data,
            angle,
            "com_position",
            ("CoM x", "CoM y", "CoM z"),
            "Position (m)",
            "Whole-body Center-of-Mass Position",
            phase_dir / "com_position.png",
            [limits[(phase, "com_position", component)] for component in range(3)],
            dpi,
            sample_seconds,
        )
        plot_foot_contact_force(
            data,
            angle,
            phase_dir / "foot_contact_force.png",
            limits[(phase, "foot_contact_force")],
            dpi,
            sample_seconds,
        )
        np.savez_compressed(
            phase_dir / "telemetry.npz",
            **{
                key: value
                for key, value in data.items()
                if isinstance(value, np.ndarray)
            },
        )
        plot_com(
            data,
            angle,
            "com_velocity",
            ("CoM vx", "CoM vy", "CoM vz"),
            "Velocity (m/s)",
            "Whole-body Center-of-Mass Velocity",
            phase_dir / "com_velocity.png",
            [limits[(phase, "com_velocity", component)] for component in range(3)],
            dpi,
            sample_seconds,
        )


def main():
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)
    if args.warmup_seconds < 0.0 or args.sample_seconds <= 0.0:
        raise ValueError("warmup_seconds must be non-negative and sample_seconds positive")

    if args.policy_profile is None:
        task_info = TASKS[args.slope_angle]
        run_name = args.load_run if args.load_run not in (None, "-1") else task_info["run"]
        run_dir = (
            Path(LEGGED_GYM_ROOT_DIR)
            / "logs"
            / task_info["experiment"]
            / str(run_name)
        )
        checkpoint_paths = {
            checkpoint: run_dir / f"model_{checkpoint}.pt" for checkpoint in checkpoints
        }
    else:
        profile = POLICY_PROFILES[args.policy_profile]
        unsupported = [
            checkpoint for checkpoint in checkpoints if checkpoint not in profile["checkpoints"]
        ]
        if unsupported:
            raise ValueError(
                f"Profile {args.policy_profile} has no checkpoint mapping for {unsupported}"
            )
        checkpoint_paths = {}
        for checkpoint in checkpoints:
            experiment, run_name, stored_checkpoint = profile["checkpoints"][checkpoint]
            checkpoint_paths[checkpoint] = (
                Path(LEGGED_GYM_ROOT_DIR)
                / "logs"
                / experiment
                / run_name
                / f"model_{stored_checkpoint}.pt"
            )
        run_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / profile["experiment"] / profile["run"]
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.policy_profile is None:
        output_dir = Path("good_result") / TASKS[args.slope_angle]["task"]
    elif args.flat_only:
        output_dir = Path("good_result") / args.policy_profile
    else:
        output_dir = Path("good_result") / args.policy_profile / f"{args.slope_angle}deg"
    if not output_dir.is_absolute():
        output_dir = Path(LEGGED_GYM_ROOT_DIR) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_seconds = args.warmup_seconds + args.sample_seconds
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_evaluation(
        env_cfg,
        args.slope_angle,
        total_seconds,
        measure_heights=args.policy_profile is None,
        flat_only=args.flat_only,
    )
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    train_cfg.runner.warm_start_path = None
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )

    joint_groups = resolve_joint_groups(env.dof_names)
    masses, local_com = get_mass_properties(env)
    warmup_steps = int(round(args.warmup_seconds / env.dt))
    sample_steps = max(2, int(round(args.sample_seconds / env.dt)))
    print(f"Task: {args.task}")
    print(f"Run: {run_dir}")
    print(f"RM75 DOF order: {env.dof_names}")
    print(f"Whole-body mass: {float(masses.sum()):.3f} kg")
    print(
        f"Per phase: {warmup_steps} warm-up steps + {sample_steps} sample steps "
        f"at dt={env.dt:.4f} s"
    )

    all_data = []
    for checkpoint in checkpoints:
        checkpoint_path = checkpoint_paths[checkpoint]
        print(f"Loading checkpoint {checkpoint}: {checkpoint_path}")
        runner.load(str(checkpoint_path), load_optimizer=False)
        try:
            policy = runner.get_inference_policy(
                device=env.device,
                random_token_dropout=False,
            )
        except TypeError:
            policy = runner.get_inference_policy(device=env.device)
        phases = ("flat",) if args.flat_only else tuple(PHASE_LOCAL_X)
        for phase in phases:
            print(f"  Collecting {phase}")
            data = collect_phase(
                env,
                runner,
                policy,
                checkpoint,
                phase,
                warmup_steps,
                sample_steps,
                masses,
                local_com,
            )
            print(
                f"    samples={len(data['time'])}/{sample_steps}, "
                f"warmup_failed={data['warmup_failed']}, "
                f"rollout_restarts={data['rollout_restarts']}, "
                f"terminated_early={data['terminated_early']}"
            )
            all_data.append(data)

    render_all_plots(
        all_data,
        joint_groups,
        output_dir,
        0 if args.flat_only else args.slope_angle,
        args.plot_dpi,
        args.sample_seconds,
        flat_only=args.flat_only,
    )
    if args.flat_only:
        image_count = len(list(output_dir.glob("checkpoint_*/*.png")))
    else:
        image_count = len(list(output_dir.glob("checkpoint_*/*/*.png")))
    print(f"Saved {image_count} PNG files to {output_dir}")
    phase_count = 1 if args.flat_only else len(PHASE_LOCAL_X)
    if image_count != len(checkpoints) * phase_count * 12:
        raise RuntimeError(f"Unexpected image count: {image_count}")
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
