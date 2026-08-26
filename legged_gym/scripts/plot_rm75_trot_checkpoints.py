"""Collect and plot RM75 flat-trot telemetry for multiple PPO checkpoints."""

from pathlib import Path

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry


TASK_NAME = "RM75_flat_trot_3ms"
DEFAULT_RUN = "Aug25_16-23-22_cold_start"
DEFAULT_CHECKPOINTS = (5000, 10000, 15000, 20000, 25000, 30000)
LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_GROUPS = {
    "hip": tuple(f"{leg}_hip_joint" for leg in LEG_ORDER),
    "thigh": tuple(f"{leg}_thigh_joint" for leg in LEG_ORDER),
    "calf": tuple(f"{leg}_calf_joint" for leg in LEG_ORDER),
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def parse_args():
    args = get_args(
        additional_parameters=[
            {
                "name": "--checkpoints",
                "type": str,
                "default": ",".join(str(value) for value in DEFAULT_CHECKPOINTS),
                "help": "Comma-separated checkpoint iterations to evaluate.",
            },
            {
                "name": "--output_dir",
                "type": str,
                "default": "good_reslut/3mstrot",
                "help": "Output directory, relative to the repository unless absolute.",
            },
            {
                "name": "--warmup_seconds",
                "type": float,
                "default": 2.0,
                "help": "Policy warm-up time before telemetry recording.",
            },
            {
                "name": "--sample_seconds",
                "type": float,
                "default": 5.0,
                "help": "Telemetry duration recorded for each checkpoint.",
            },
            {
                "name": "--command_speed",
                "type": float,
                "default": 3.0,
                "help": "Fixed forward command used during evaluation in m/s.",
            },
            {
                "name": "--plot_dpi",
                "type": int,
                "default": 180,
                "help": "PNG resolution in dots per inch.",
            },
        ]
    )
    args.task = TASK_NAME
    args.num_envs = 1
    args.headless = True
    return args


def parse_checkpoints(raw_value):
    checkpoints = tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())
    if not checkpoints:
        raise ValueError("At least one checkpoint must be supplied.")
    if any(value <= 0 for value in checkpoints):
        raise ValueError(f"Checkpoint iterations must be positive: {checkpoints}")
    return checkpoints


def configure_evaluation(env_cfg, total_seconds, command_speed):
    env_cfg.env.num_envs = 1
    env_cfg.env.test = False
    env_cfg.env.episode_length_s = max(float(env_cfg.env.episode_length_s), total_seconds + 2.0)
    env_cfg.env.enable_camera_sensors = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.noise.add_noise = False
    env_cfg.init_state.turn_over = False

    env_cfg.commands.curriculum = False
    env_cfg.commands.command_range_curriculum = []
    env_cfg.commands.resampling_time = total_seconds + 10.0
    for range_name in ("lin_vel_x", "new_lin_vel_x"):
        if hasattr(env_cfg.commands.ranges, range_name):
            setattr(env_cfg.commands.ranges, range_name, [command_speed, command_speed])
    for range_name in (
        "lin_vel_y",
        "ang_vel_yaw",
        "heading",
        "new_lin_vel_y",
        "new_ang_vel_yaw",
        "new_heading",
    ):
        if hasattr(env_cfg.commands.ranges, range_name):
            setattr(env_cfg.commands.ranges, range_name, [0.0, 0.0])

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


def set_fixed_command(env, command_speed):
    env.commands[:, :3] = torch.tensor(
        [command_speed, 0.0, 0.0], dtype=torch.float, device=env.device
    )
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0
    env.commands_resampling_step[:] = env.cfg.commands.resampling_time / env.dt


def reset_deterministically(env, command_speed, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env.reset()
    env_id = torch.tensor([0], dtype=torch.long, device=env.device)
    env_id_int32 = env_id.to(dtype=torch.int32)

    env.root_states[0] = env.base_init_state
    env.root_states[0, :3] += env.env_origins[0]
    env.root_states[0, 7:13] = 0.0
    env.dof_pos[0] = env.default_dof_pos[0]
    env.dof_vel[0] = 0.0
    env.episode_length_buf[0] = 0
    env.reset_buf[0] = 0
    env.time_out_buf[0] = False

    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_id_int32),
        1,
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_id_int32),
        1,
    )

    set_fixed_command(env, command_speed)
    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, dtype=torch.float, device=env.device
    )
    obs, _, _, _, _ = env.step(zero_actions)
    return obs


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
    if len(properties) != env.num_bodies:
        raise RuntimeError(
            f"Rigid-body property count mismatch: {len(properties)} != {env.num_bodies}"
        )
    return masses, local_com


def get_center_of_mass_state(env, masses, local_com):
    body_states = env.rigid_body_states.view(env.num_envs, env.num_bodies, 13)[0]
    body_positions = body_states[:, 0:3]
    body_quaternions = body_states[:, 3:7]
    body_com_positions = body_positions + quat_apply(body_quaternions, local_com)

    total_mass = masses.sum()
    center_position = (body_com_positions * masses.unsqueeze(1)).sum(dim=0) / total_mass
    # PhysX rigid-body linear velocity is the velocity at each body's center of mass.
    center_velocity = (body_states[:, 7:10] * masses.unsqueeze(1)).sum(dim=0) / total_mass
    return center_position, center_velocity


def resolve_joint_groups(dof_names):
    missing = [
        name
        for joint_names in JOINT_GROUPS.values()
        for name in joint_names
        if name not in dof_names
    ]
    if missing:
        raise RuntimeError(f"Missing expected RM75 joints: {missing}; available={dof_names}")
    return {
        group_name: np.asarray([dof_names.index(name) for name in joint_names], dtype=int)
        for group_name, joint_names in JOINT_GROUPS.items()
    }


def collect_checkpoint(
    env,
    runner,
    checkpoint_path,
    checkpoint,
    command_speed,
    warmup_steps,
    sample_steps,
    masses,
    local_com,
    seed,
):
    runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs = reset_deterministically(env, command_speed, seed)
    env.common_step_counter = checkpoint * env.num_steps_per_env

    joint_velocity = []
    torque = []
    power = []
    com_position = []
    com_velocity = []
    reset_steps = []

    with torch.no_grad():
        for step in range(warmup_steps + sample_steps):
            set_fixed_command(env, command_speed)
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

            if step < warmup_steps:
                continue

            sample_index = step - warmup_steps
            dof_velocity = env.dof_vel[0].detach()
            applied_torque = env.torques[0].detach()
            center_position, center_velocity = get_center_of_mass_state(
                env, masses, local_com
            )

            joint_velocity.append(dof_velocity.cpu().numpy().copy())
            torque.append(applied_torque.cpu().numpy().copy())
            power.append((applied_torque * dof_velocity).cpu().numpy().copy())
            com_position.append(center_position.detach().cpu().numpy().copy())
            com_velocity.append(center_velocity.detach().cpu().numpy().copy())
            if bool(dones[0]):
                reset_steps.append(sample_index)

    time = (np.arange(sample_steps, dtype=np.float64) + 1.0) * env.dt
    return {
        "checkpoint": checkpoint,
        "time": time,
        "joint_velocity": np.asarray(joint_velocity),
        "torque": np.asarray(torque),
        "power": np.asarray(power),
        "com_position": np.asarray(com_position),
        "com_velocity": np.asarray(com_velocity),
        "reset_steps": tuple(reset_steps),
    }


def padded_limits(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lower = float(values.min())
    upper = float(values.max())
    if np.isclose(lower, upper):
        padding = max(0.1, abs(lower) * 0.1)
    else:
        padding = 0.06 * (upper - lower)
    return lower - padding, upper + padding


def add_reset_markers(ax, data):
    for reset_step in data["reset_steps"]:
        if reset_step < len(data["time"]):
            ax.axvline(
                data["time"][reset_step],
                color="0.35",
                linewidth=0.8,
                linestyle=":",
                alpha=0.65,
            )


def style_axis(ax, ylabel, time_limit, y_limit=None):
    ax.set_xlim(time_limit)
    if y_limit is not None:
        ax.set_ylim(y_limit)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.axhline(0.0, color="0.45", linewidth=0.7)


def plot_joint_group(
    data,
    metric,
    group_name,
    joint_indices,
    ylabel,
    title_name,
    output_path,
    y_limit,
    dpi,
):
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    for leg, color, joint_index in zip(LEG_ORDER, COLORS, joint_indices):
        ax.plot(
            data["time"],
            data[metric][:, joint_index],
            label=f"{leg} {group_name}",
            color=color,
            linewidth=1.25,
        )
    add_reset_markers(ax, data)
    style_axis(ax, ylabel, (data["time"][0], data["time"][-1]), y_limit)
    ax.set_title(f"Checkpoint {data['checkpoint']} | {title_name} | {group_name.capitalize()}")
    ax.legend(ncol=4, loc="upper center", frameon=False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_com(data, metric, labels, ylabel, title, output_path, y_limits, dpi):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True, constrained_layout=True)
    time_limit = (data["time"][0], data["time"][-1])
    for component, (ax, label, color, y_limit) in enumerate(
        zip(axes, labels, COLORS[:3], y_limits)
    ):
        ax.plot(
            data["time"],
            data[metric][:, component],
            label=label,
            color=color,
            linewidth=1.3,
        )
        add_reset_markers(ax, data)
        style_axis(ax, ylabel, time_limit, y_limit)
        ax.legend(loc="upper right", frameon=False)
    axes[0].set_title(f"Checkpoint {data['checkpoint']} | {title}")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_global_limits(all_data, joint_groups):
    limits = {}
    for metric in ("joint_velocity", "torque", "power"):
        for group_name, joint_indices in joint_groups.items():
            values = np.concatenate(
                [data[metric][:, joint_indices].reshape(-1) for data in all_data]
            )
            limits[(metric, group_name)] = padded_limits(values)

    for metric in ("com_position", "com_velocity"):
        for component in range(3):
            values = np.concatenate([data[metric][:, component] for data in all_data])
            limits[(metric, component)] = padded_limits(values)
    return limits


def render_all_plots(all_data, joint_groups, output_dir, dpi):
    limits = build_global_limits(all_data, joint_groups)
    joint_metrics = (
        ("joint_velocity", "Joint velocity (rad/s)", "Joint Velocity"),
        ("torque", "Torque (N m)", "Joint Torque"),
        ("power", "Signed power (W)", "Instantaneous Mechanical Power"),
    )

    for data in all_data:
        checkpoint_dir = output_dir / f"checkpoint_{data['checkpoint']}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for metric, ylabel, title in joint_metrics:
            for group_name, joint_indices in joint_groups.items():
                plot_joint_group(
                    data=data,
                    metric=metric,
                    group_name=group_name,
                    joint_indices=joint_indices,
                    ylabel=ylabel,
                    title_name=title,
                    output_path=checkpoint_dir / f"{metric}_{group_name}.png",
                    y_limit=limits[(metric, group_name)],
                    dpi=dpi,
                )

        plot_com(
            data=data,
            metric="com_position",
            labels=("CoM x", "CoM y", "CoM z"),
            ylabel="Position (m)",
            title="Whole-body Center-of-Mass Position",
            output_path=checkpoint_dir / "com_position.png",
            y_limits=[limits[("com_position", component)] for component in range(3)],
            dpi=dpi,
        )
        plot_com(
            data=data,
            metric="com_velocity",
            labels=("CoM vx", "CoM vy", "CoM vz"),
            ylabel="Velocity (m/s)",
            title="Whole-body Center-of-Mass Velocity",
            output_path=checkpoint_dir / "com_velocity.png",
            y_limits=[limits[("com_velocity", component)] for component in range(3)],
            dpi=dpi,
        )


def main():
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)
    if args.warmup_seconds < 0.0 or args.sample_seconds <= 0.0:
        raise ValueError("warmup_seconds must be non-negative and sample_seconds positive.")

    run_name = args.load_run if args.load_run not in (None, "-1") else DEFAULT_RUN
    run_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "RM75_flat_trot_3ms" / str(run_name)
    checkpoint_paths = {
        checkpoint: run_dir / f"model_{checkpoint}.pt" for checkpoint in checkpoints
    }
    missing_paths = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing checkpoints: {missing_paths}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(LEGGED_GYM_ROOT_DIR) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_evaluation(
        env_cfg,
        total_seconds=args.warmup_seconds + args.sample_seconds,
        command_speed=args.command_speed,
    )
    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )

    joint_groups = resolve_joint_groups(env.dof_names)
    masses, local_com = get_mass_properties(env)
    warmup_steps = int(round(args.warmup_seconds / env.dt))
    sample_steps = int(round(args.sample_seconds / env.dt))
    if sample_steps < 2:
        raise ValueError("sample_seconds is too short for the environment control period.")

    print(f"RM75 DOF order: {env.dof_names}")
    print(f"Whole-body mass: {float(masses.sum()):.3f} kg")
    print(
        f"Sampling {sample_steps} points over {sample_steps * env.dt:.3f} s "
        f"after {warmup_steps * env.dt:.3f} s warm-up"
    )

    all_data = []
    for checkpoint in checkpoints:
        print(f"Collecting checkpoint {checkpoint}: {checkpoint_paths[checkpoint]}")
        all_data.append(
            collect_checkpoint(
                env=env,
                runner=runner,
                checkpoint_path=checkpoint_paths[checkpoint],
                checkpoint=checkpoint,
                command_speed=args.command_speed,
                warmup_steps=warmup_steps,
                sample_steps=sample_steps,
                masses=masses,
                local_com=local_com,
                seed=int(train_cfg.seed),
            )
        )

    render_all_plots(all_data, joint_groups, output_dir, args.plot_dpi)
    image_count = len(list(output_dir.glob("checkpoint_*/*.png")))
    print(f"Saved {image_count} PNG files to {output_dir}")
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
