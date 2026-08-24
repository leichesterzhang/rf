#!/usr/bin/env python3
"""Batch IsaacGym evaluation for the Go2 parkour MoE policy.

The script evaluates:
  1. Forward pass rate on every MGDP parkour terrain except flat.
  2. Flat-ground fall-recovery success rate for half and full falls.
  3. Omni world-forward success rate on ramp/slope and stairs.
  4. Blind forward pass rate on slope, stairs, and air stones.

Forward, omni, and blind success are judged only when an env refreshes/resets: the
pre-reset progress from the spawn/start point must be at the terrain end.
"""

import copy
import csv
import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import isaacgym  # noqa: F401
from isaacgym import gymutil
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import (
    class_to_dict,
    get_load_path,
    set_seed,
    update_cfg_from_args,
)
import rsl_rl.runners as rsl_runners


MGDP_TERRAIN_NAMES = (
    "single_gap",
    "step_stone",
    "two_row_stones",
    "one_row_stones",
    "single_bridge",
    "air_beams",
    "air_stones",
    "hurdle",
    "ramp",
    "corridor",
    "stairs_up",
    "flat",
    "rough_flat",
)
TERRAIN_NAME_TO_ID = {name: idx for idx, name in enumerate(MGDP_TERRAIN_NAMES)}
TERRAIN_ALIASES = {
    "air_stone": "air_stones",
    "air_stones": "air_stones",
    "airstone": "air_stones",
    "airstones": "air_stones",
    "slope": "ramp",
    "slpoe": "ramp",
    "sloped": "ramp",
    "stair": "stairs_up",
    "stairs": "stairs_up",
    "stairsup": "stairs_up",
    "stairs_up": "stairs_up",
    "rough": "rough_flat",
    "roughflat": "rough_flat",
    "rough_flat": "rough_flat",
}


def parse_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2_parkour_moe"},
        {"name": "--resume", "action": "store_true", "default": False},
        {"name": "--experiment_name", "type": str},
        {"name": "--run_name", "type": str},
        {"name": "--load_run", "type": str},
        {"name": "--checkpoint", "type": int},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--viewer", "action": "store_true", "default": False},
        {"name": "--horovod", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int},
        {"name": "--seed", "type": int},
        {"name": "--max_iterations", "type": int},
        {"name": "--robogauge", "action": "store_true", "default": False},
        {"name": "--robogauge_port", "type": int, "default": 9973},
        {"name": "--levels", "type": str, "default": "4,6,8,10"},
        {"name": "--level_indexing", "type": str, "default": "one_based"},
        {"name": "--trials_per_case", "type": int, "default": 256},
        {"name": "--forward_terrains", "type": str, "default": "all"},
        {"name": "--omni_terrains", "type": str, "default": "slope,stairs"},
        {"name": "--blind_terrains", "type": str, "default": "stairs,slope,air_stones"},
        {"name": "--skip_forward", "action": "store_true", "default": False},
        {"name": "--skip_recovery", "action": "store_true", "default": False},
        {"name": "--skip_omni", "action": "store_true", "default": False},
        {"name": "--skip_blind", "action": "store_true", "default": False},
        {"name": "--command_speed", "type": float, "default": 1.0},
        {"name": "--eval_time", "type": float, "default": 0.0},
        {"name": "--end_tolerance", "type": float, "default": 1.0},
        {"name": "--recovery_time", "type": float, "default": 5.0},
        {"name": "--recovery_stable_time", "type": float, "default": 0.2},
        {"name": "--keep_camera_noise", "action": "store_true", "default": False},
        {"name": "--out_dir", "type": str, "default": ""},
    ]
    args = gymutil.parse_arguments(
        description="Evaluate Go2 parkour MoE policy metrics.",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    if args.viewer:
        args.headless = False
    else:
        args.headless = True
    return args


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_levels(value):
    levels = [int(item) for item in split_csv(value)]
    if not levels:
        raise ValueError("At least one terrain level is required.")
    return levels


def normalize_terrain_name(name):
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = TERRAIN_ALIASES.get(key, key)
    if key not in TERRAIN_NAME_TO_ID:
        allowed = ", ".join(MGDP_TERRAIN_NAMES)
        raise ValueError(f"Unknown terrain '{name}'. Allowed: {allowed}")
    return key


def parse_terrain_list(value, *, include_flat):
    if value.strip().lower() == "all":
        names = list(MGDP_TERRAIN_NAMES)
    else:
        names = [normalize_terrain_name(item) for item in split_csv(value)]
    if not include_flat:
        names = [name for name in names if name != "flat"]
    seen = set()
    unique_names = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)
    return unique_names


def level_to_row(level, indexing):
    indexing = indexing.lower()
    if indexing in ("one_based", "1", "one-based"):
        row = level - 1
    elif indexing in ("zero_based", "0", "zero-based"):
        row = level
    else:
        raise ValueError("--level_indexing must be one_based or zero_based.")
    if row < 0:
        raise ValueError(f"Level {level} maps to invalid row {row}.")
    return row


def terrain_one_hot(name):
    proportions = [0.0] * len(MGDP_TERRAIN_NAMES)
    proportions[TERRAIN_NAME_TO_ID[name]] = 1.0
    return proportions


def clone_cfgs(task_name):
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    return copy.deepcopy(env_cfg), copy.deepcopy(train_cfg)


def make_case_args(args):
    case_args = copy.copy(args)
    case_args.num_envs = None
    return case_args


def disable_randomization(env_cfg, keep_camera_noise=False):
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_action_delay = False
    if hasattr(env_cfg, "camera") and not keep_camera_noise:
        env_cfg.camera.noise_gaussian = 0.0
        env_cfg.camera.noise_dropout = 0.0


def configure_env_cfg(
    env_cfg,
    args,
    terrain_name,
    num_envs,
    num_rows,
    mode,
    recovery_kind=None,
):
    env_cfg.env.num_envs = num_envs
    env_cfg.env.test = False
    env_cfg.env.enable_camera_sensors = False
    if hasattr(env_cfg, "camera"):
        env_cfg.camera.source = "proxy"

    env_cfg.terrain.terrain_style = "mgdp_parkour"
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_rows = num_rows
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = num_rows - 1
    env_cfg.terrain.terrain_proportions = terrain_one_hot(terrain_name)

    disable_randomization(env_cfg, keep_camera_noise=args.keep_camera_noise)
    env_cfg.commands.resampling_time = max(
        float(getattr(env_cfg.commands, "resampling_time", 0.0)),
        float(args.eval_time) + 2.0,
        float(args.recovery_time) + 2.0,
    )
    env_cfg.commands.zero_command = False
    env_cfg.commands.zero_command_curriculum = None

    if mode == "recovery":
        env_cfg.init_state.turn_over = True
        env_cfg.init_state.turn_over_terrain_ids = [TERRAIN_NAME_TO_ID["flat"]]
        if recovery_kind == "half":
            env_cfg.init_state.turn_over_proportions = [0.0, 1.0, 0.0]
            env_cfg.init_state.turn_over_roll_ranges["sideflip"] = [math.pi / 2, math.pi / 2]
        elif recovery_kind == "full":
            env_cfg.init_state.turn_over_proportions = [1.0, 0.0, 0.0]
            env_cfg.init_state.turn_over_roll_ranges["backflip"] = [math.pi, math.pi]
        else:
            raise ValueError(f"Unknown recovery kind: {recovery_kind}")
        env_cfg.commands.turn_over_zero_time["backflip"] = float(args.recovery_time)
        env_cfg.commands.turn_over_zero_time["sideflip"] = float(args.recovery_time)
        env_cfg.commands.omni_enabled = False
        env_cfg.commands.omni_ratio = 0.0
    else:
        env_cfg.init_state.turn_over = False
        env_cfg.commands.omni_enabled = mode == "omni"
        env_cfg.commands.omni_ratio = 1.0 if mode == "omni" else 0.0
        env_cfg.commands.omni_terrain_ids = [TERRAIN_NAME_TO_ID[terrain_name]] if mode == "omni" else []
        if mode == "omni":
            env_cfg.commands.omni_random_yaw = True


def build_runner(env, train_cfg, args):
    train_cfg.runner.resume = True
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    runner_cls = getattr(rsl_runners, train_cfg.runner_class_name)
    runner = runner_cls(env, class_to_dict(train_cfg), log_dir=None, device=args.rl_device)

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    load_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    print(f"Loading model from: {load_path}")
    runner.load(load_path)
    return runner, train_cfg, load_path


def annotate_rows(rows, load_path):
    for row in rows:
        row["load_path"] = load_path
    return rows


def reset_runner_state(runner, env):
    runner.history.zero_()
    dones = torch.ones(env.num_envs, dtype=torch.bool, device=runner.device)
    if hasattr(runner.alg, "estimator"):
        runner.alg.estimator.reset(dones)
    if hasattr(runner, "_reset_swav_history"):
        runner._reset_swav_history(dones)
    runner.latest_inference_mcp_code = None
    runner.latest_inference_m_hat = None
    runner.latest_inference_gating_weights = None


def configure_vision_flag_for_mode(runner, env, mode):
    if mode == "blind":
        runner.omni_blind = True
        runner.omni_idx_tensor = torch.arange(
            env.num_envs,
            device=runner.device,
            dtype=torch.long,
        )
    elif mode == "forward":
        runner.omni_blind = False
        runner.omni_idx_tensor = torch.zeros(0, device=runner.device, dtype=torch.long)


def force_levels_and_reset(env, row_indices, trials_per_level):
    device = env.device
    level_values = []
    for row in row_indices:
        level_values.extend([row] * trials_per_level)
    if len(level_values) != env.num_envs:
        raise ValueError(
            f"Level assignment has {len(level_values)} entries for {env.num_envs} envs."
        )
    levels = torch.tensor(level_values, dtype=torch.long, device=device)
    terrain_types = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    env.cfg.terrain.curriculum = False
    env.terrain_levels[:] = levels
    env.terrain_types[:] = terrain_types
    env.env_origins[:] = env.terrain_origins[levels, terrain_types]
    if hasattr(env, "terrain_spawn_origins"):
        env.reset_origins[:] = env.terrain_spawn_origins[levels, terrain_types]
        env.progress_origins[:] = env.reset_origins
    else:
        env.reset_origins[:] = env.env_origins
        env.progress_origins[:] = env.env_origins
    if hasattr(env, "terrain_id_map"):
        env.terrain_ids[:] = env.terrain_id_map[levels, terrain_types]
        env._update_env_command_ranges()
    if hasattr(env, "_init_omni_buffers"):
        env._init_omni_buffers()

    obs, _ = env.reset()
    return obs


def apply_forward_command(env, command_speed):
    env.commands_resampling_step[:] = env.cfg.commands.resampling_time / env.dt
    env.commands[:, 0] = command_speed
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0
    if hasattr(env, "stop_heading"):
        env.stop_heading[:] = False


def apply_zero_command(env):
    env.commands_resampling_step[:] = env.cfg.commands.resampling_time / env.dt
    env.commands[:, :3] = 0.0
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0
    if hasattr(env, "stop_heading"):
        env.stop_heading[:] = True


def apply_omni_command(env, command_speed):
    env.commands_resampling_step[:] = env.cfg.commands.resampling_time / env.dt
    omni_mask = env._omni_env_mask()
    speed = torch.full((env.num_envs,), float(command_speed), device=env.device)
    stairs_max_speed = getattr(env.cfg.commands, "omni_stairs_max_speed", None)
    if stairs_max_speed is not None and hasattr(env, "terrain_ids"):
        stairs_mask = env.terrain_ids == TERRAIN_NAME_TO_ID["stairs_up"]
        speed = torch.where(stairs_mask, speed.clamp(max=float(stairs_max_speed)), speed)
    env.omni_command_speed[omni_mask] = speed[omni_mask]
    env._apply_omni_body_commands()


def primary_reasons(env, env_ids):
    reasons = []
    masks = getattr(env, "termination_reason_masks", {})
    for env_id in env_ids.detach().cpu().tolist():
        matched = [name for name, mask in masks.items() if bool(mask[env_id].detach().cpu().item())]
        reasons.append("+".join(matched) if matched else "reset")
    return reasons


def progress_from_start(env):
    progress_origins = getattr(env, "progress_origins", env.env_origins)
    return env.root_states[:, 0] - progress_origins[:, 0]


def terrain_end_relative_x(env, args):
    return 0.5 * float(env.cfg.terrain.terrain_length) - float(args.end_tolerance)


def terrain_end_progress_thresholds(env, args):
    progress_origins = getattr(env, "progress_origins", env.env_origins)
    end_x = env.env_origins[:, 0] + terrain_end_relative_x(env, args)
    return end_x - progress_origins[:, 0]


def progress_eval_steps(env, args):
    if float(args.eval_time) > 0.0:
        return int(math.ceil(float(args.eval_time) / env.dt))
    return int(env.max_episode_length) + 2


def progress_eval_time_s(env, args):
    if float(args.eval_time) > 0.0:
        return float(args.eval_time)
    return float(progress_eval_steps(env, args) * env.dt)


def effective_command_speed(env, args, terrain_name, mode):
    speed = float(args.command_speed)
    if mode != "omni" or terrain_name != "stairs_up":
        return speed
    stairs_max_speed = getattr(env.cfg.commands, "omni_stairs_max_speed", None)
    if stairs_max_speed is None:
        return speed
    return min(speed, float(stairs_max_speed))


def progress_category(mode):
    if mode in ("forward", "omni", "blind"):
        return mode
    return "forward"


def progress_terrain_alias(terrain_name, mode):
    if terrain_name == "ramp" and mode in ("omni", "blind"):
        return "slope"
    return terrain_name


def vision_flag_for_mode(mode):
    if mode == "blind":
        return False
    if mode == "forward":
        return True
    return ""


def install_reset_snapshot_hook(env):
    if hasattr(env, "_eval_original_reset_idx"):
        return
    env._eval_original_reset_idx = env.reset_idx
    env._eval_capture_resets = False
    env._eval_reset_snapshots = []

    def wrapped_reset_idx(env_ids):
        if getattr(env, "_eval_capture_resets", False) and len(env_ids) > 0:
            progress_origins = getattr(env, "progress_origins", env.env_origins)
            ids = env_ids.detach().clone()
            reasons = primary_reasons(env, ids)
            env._eval_reset_snapshots.append(
                {
                    "ids": ids,
                    "progress_x": (env.root_states[ids, 0] - progress_origins[ids, 0]).detach().clone(),
                    "relative_x": (env.root_states[ids, 0] - env.env_origins[ids, 0]).detach().clone(),
                    "reasons": reasons,
                }
            )
        return env._eval_original_reset_idx(env_ids)

    env.reset_idx = wrapped_reset_idx


def evaluate_progress(
    env,
    runner,
    policy,
    args,
    terrain_name,
    level_labels,
    row_indices,
    mode,
):
    obs = env.get_observations()
    device = env.device
    num_envs = env.num_envs
    trials_per_level = args.trials_per_case
    max_steps = progress_eval_steps(env, args)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    time_to_success = torch.full((num_envs,), float("nan"), dtype=torch.float, device=device)
    time_to_reset = torch.full((num_envs,), float("nan"), dtype=torch.float, device=device)
    max_progress = progress_from_start(env).clone()
    reset_progress = torch.full((num_envs,), float("nan"), dtype=torch.float, device=device)
    reset_relative_x = torch.full((num_envs,), float("nan"), dtype=torch.float, device=device)
    reset_reason = [""] * num_envs
    failure_reason = [""] * num_envs
    end_progress_thresholds = terrain_end_progress_thresholds(env, args)
    install_reset_snapshot_hook(env)

    started_at = time.time()
    env._eval_capture_resets = True
    try:
        with torch.inference_mode():
            for step in range(max_steps):
                if mode == "omni":
                    apply_omni_command(env, args.command_speed)
                else:
                    apply_forward_command(env, args.command_speed)

                env._eval_reset_snapshots = []
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())

                progress = progress_from_start(env)
                if torch.any(active):
                    max_progress[active] = torch.maximum(max_progress[active], progress[active])

                for snapshot in env._eval_reset_snapshots:
                    ids = snapshot["ids"]
                    active_mask = active[ids]
                    if not torch.any(active_mask):
                        continue
                    active_ids = ids[active_mask]
                    snapshot_progress = snapshot["progress_x"][active_mask]
                    snapshot_relative_x = snapshot["relative_x"][active_mask]
                    at_terrain_end = snapshot_progress >= end_progress_thresholds[active_ids]

                    reset_progress[active_ids] = snapshot_progress
                    reset_relative_x[active_ids] = snapshot_relative_x
                    max_progress[active_ids] = torch.maximum(max_progress[active_ids], snapshot_progress)
                    time_to_reset[active_ids] = (step + 1) * env.dt

                    success_ids = active_ids[at_terrain_end]
                    if success_ids.numel() > 0:
                        succeeded[success_ids] = True
                        time_to_success[success_ids] = (step + 1) * env.dt

                    active_ids_cpu = active_ids.detach().cpu().tolist()
                    at_end_cpu = at_terrain_end.detach().cpu().tolist()
                    reason_indices = active_mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
                    for env_id, is_success, reason_idx in zip(active_ids_cpu, at_end_cpu, reason_indices):
                        reset_reason[env_id] = snapshot["reasons"][reason_idx]
                        if not is_success:
                            failure_reason[env_id] = snapshot["reasons"][reason_idx]

                    active[active_ids] = False

                if not torch.any(active):
                    break
    finally:
        env._eval_capture_resets = False

    unresolved_ids = active.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    for env_id in unresolved_ids:
        failure_reason[env_id] = "no_refresh_timeout"
    active[:] = False

    elapsed = time.time() - started_at
    case_rows, trial_rows = aggregate_progress_rows(
        terrain_name,
        level_labels,
        row_indices,
        trials_per_level,
        succeeded,
        time_to_success,
        time_to_reset,
        max_progress,
        reset_progress,
        reset_relative_x,
        reset_reason,
        failure_reason,
        mode,
        elapsed,
        args,
        env,
    )
    return case_rows, trial_rows


def aggregate_progress_rows(
    terrain_name,
    level_labels,
    row_indices,
    trials_per_level,
    succeeded,
    time_to_success,
    time_to_reset,
    max_progress,
    reset_progress,
    reset_relative_x,
    reset_reason,
    failure_reason,
    mode,
    elapsed,
    args,
    env,
):
    rows = []
    trial_rows = []
    succeeded_cpu = succeeded.detach().cpu()
    time_cpu = time_to_success.detach().cpu()
    reset_time_cpu = time_to_reset.detach().cpu()
    progress_cpu = max_progress.detach().cpu()
    reset_progress_cpu = reset_progress.detach().cpu()
    reset_relative_x_cpu = reset_relative_x.detach().cpu()
    end_relative_x = terrain_end_relative_x(env, args)
    end_progress_threshold_cpu = terrain_end_progress_thresholds(env, args).detach().cpu()
    for level_idx, (level_label, row_index) in enumerate(zip(level_labels, row_indices)):
        start = level_idx * trials_per_level
        stop = start + trials_per_level
        ids = list(range(start, stop))
        end_progress_threshold = float(end_progress_threshold_cpu[start].item())
        successes = int(succeeded_cpu[start:stop].sum().item())
        reasons = Counter(
            failure_reason[env_id] or "none"
            for env_id in ids
            if not bool(succeeded_cpu[env_id].item())
        )
        success_reasons = Counter(
            reset_reason[env_id] or "none"
            for env_id in ids
            if bool(succeeded_cpu[env_id].item())
        )
        rows.append(
            {
                "category": progress_category(mode),
                "terrain": terrain_name,
                "terrain_alias": progress_terrain_alias(terrain_name, mode),
                "terrain_id": TERRAIN_NAME_TO_ID[terrain_name],
                "level": level_label,
                "row_index": row_index,
                "difficulty": row_index / float(env.cfg.terrain.num_rows),
                "trials": trials_per_level,
                "successes": successes,
                "failures": trials_per_level - successes,
                "success_rate": successes / float(trials_per_level),
                "command_speed": args.command_speed,
                "effective_command_speed": effective_command_speed(env, args, terrain_name, mode),
                "vision_flag": vision_flag_for_mode(mode),
                "end_relative_x_threshold": end_relative_x,
                "end_progress_threshold_m": end_progress_threshold,
                "end_tolerance": args.end_tolerance,
                "eval_time_s": progress_eval_time_s(env, args),
                "wall_time_s": elapsed,
                "failure_reasons": json.dumps(dict(sorted(reasons.items())), sort_keys=True),
                "success_reset_reasons": json.dumps(dict(sorted(success_reasons.items())), sort_keys=True),
            }
        )
        for env_id in ids:
            trial_rows.append(
                {
                    "category": progress_category(mode),
                    "terrain": terrain_name,
                    "terrain_id": TERRAIN_NAME_TO_ID[terrain_name],
                    "level": level_label,
                    "row_index": row_index,
                    "env_id": env_id,
                    "success": bool(succeeded_cpu[env_id].item()),
                    "vision_flag": vision_flag_for_mode(mode),
                    "time_to_success_s": (
                        float(time_cpu[env_id].item())
                        if bool(succeeded_cpu[env_id].item())
                        else ""
                    ),
                    "time_to_reset_s": (
                        float(reset_time_cpu[env_id].item())
                        if not torch.isnan(reset_time_cpu[env_id])
                        else ""
                    ),
                    "max_progress_m": float(progress_cpu[env_id].item()),
                    "reset_progress_m": (
                        float(reset_progress_cpu[env_id].item())
                        if not torch.isnan(reset_progress_cpu[env_id])
                        else ""
                    ),
                    "reset_relative_x_m": (
                        float(reset_relative_x_cpu[env_id].item())
                        if not torch.isnan(reset_relative_x_cpu[env_id])
                        else ""
                    ),
                    "end_progress_threshold_m": float(end_progress_threshold_cpu[env_id].item()),
                    "reset_reason": reset_reason[env_id],
                    "failure_reason": "" if bool(succeeded_cpu[env_id].item()) else failure_reason[env_id],
                }
            )
    return rows, trial_rows


def evaluate_recovery(env, runner, policy, args, recovery_kind):
    obs = env.get_observations()
    device = env.device
    num_envs = env.num_envs
    max_steps = int(math.ceil(float(args.recovery_time) / env.dt))
    stable_steps_required = max(1, int(math.ceil(float(args.recovery_stable_time) / env.dt)))

    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    time_to_success = torch.full((num_envs,), float("nan"), dtype=torch.float, device=device)
    failure_reason = [""] * num_envs

    started_at = time.time()
    with torch.inference_mode():
        for step in range(max_steps):
            apply_zero_command(env)
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

            upright = env._turn_over_success_mask()
            stable_steps = torch.where(active & upright, stable_steps + 1, torch.zeros_like(stable_steps))
            newly_succeeded = active & (stable_steps >= stable_steps_required)
            if torch.any(newly_succeeded):
                succeeded[newly_succeeded] = True
                time_to_success[newly_succeeded] = (step + 1) * env.dt
                active[newly_succeeded] = False

            failed = active & dones.bool()
            if torch.any(failed):
                failed_ids = failed.nonzero(as_tuple=False).flatten()
                for env_id, reason in zip(failed_ids.detach().cpu().tolist(), primary_reasons(env, failed_ids)):
                    failure_reason[env_id] = reason
                active[failed] = False

            if not torch.any(active):
                break

    unresolved_ids = active.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    for env_id in unresolved_ids:
        failure_reason[env_id] = "recovery_timeout"

    elapsed = time.time() - started_at
    succeeded_cpu = succeeded.detach().cpu()
    time_cpu = time_to_success.detach().cpu()
    successes = int(succeeded_cpu.sum().item())
    reasons = Counter(
        failure_reason[env_id] or "none"
        for env_id in range(num_envs)
        if not bool(succeeded_cpu[env_id].item())
    )
    case_rows = [
        {
            "category": "recovery",
            "terrain": "flat",
            "terrain_alias": "flat",
            "terrain_id": TERRAIN_NAME_TO_ID["flat"],
            "level": "",
            "row_index": "",
            "difficulty": recovery_kind,
            "trials": num_envs,
            "successes": successes,
            "failures": num_envs - successes,
            "success_rate": successes / float(num_envs),
            "command_speed": 0.0,
            "effective_command_speed": 0.0,
            "vision_flag": "",
            "end_relative_x_threshold": "",
            "end_progress_threshold_m": "",
            "end_tolerance": "",
            "eval_time_s": args.recovery_time,
            "wall_time_s": elapsed,
            "failure_reasons": json.dumps(dict(sorted(reasons.items())), sort_keys=True),
            "success_reset_reasons": "",
        }
    ]
    trial_rows = []
    for env_id in range(num_envs):
        trial_rows.append(
            {
                "category": "recovery",
                "terrain": "flat",
                "terrain_id": TERRAIN_NAME_TO_ID["flat"],
                "level": "",
                "row_index": "",
                "env_id": env_id,
                "success": bool(succeeded_cpu[env_id].item()),
                "vision_flag": "",
                "time_to_success_s": (
                    float(time_cpu[env_id].item())
                    if bool(succeeded_cpu[env_id].item())
                    else ""
                ),
                "max_progress_m": "",
                "failure_reason": "" if bool(succeeded_cpu[env_id].item()) else failure_reason[env_id],
                "recovery_kind": recovery_kind,
            }
        )
    return case_rows, trial_rows


def destroy_env(env):
    if env is None:
        return
    if getattr(env, "viewer", None) is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


def run_progress_suite(args, level_labels, row_indices, terrain_names, mode):
    all_case_rows = []
    all_trial_rows = []
    for terrain_name in terrain_names:
        print(f"\n[{mode}] terrain={terrain_name}")
        env = None
        try:
            env_cfg, train_cfg = clone_cfgs(args.task)
            num_envs = args.trials_per_case * len(row_indices)
            num_rows = max(max(row_indices) + 1, 1)
            configure_env_cfg(env_cfg, args, terrain_name, num_envs, num_rows, mode)
            env, _ = task_registry.make_env(
                name=args.task,
                args=make_case_args(args),
                env_cfg=env_cfg,
            )
            env.cfg.terrain.curriculum = False
            runner, _, load_path = build_runner(env, train_cfg, args)
            policy = runner.get_inference_policy(device=env.device)
            force_levels_and_reset(env, row_indices, args.trials_per_case)
            reset_runner_state(runner, env)
            configure_vision_flag_for_mode(runner, env, mode)
            case_rows, trial_rows = evaluate_progress(
                env,
                runner,
                policy,
                args,
                terrain_name,
                level_labels,
                row_indices,
                mode,
            )
            annotate_rows(case_rows, load_path)
            print_case_rows(case_rows)
            all_case_rows.extend(case_rows)
            all_trial_rows.extend(trial_rows)
        finally:
            destroy_env(env)
    return all_case_rows, all_trial_rows


def run_recovery_suite(args):
    all_case_rows = []
    all_trial_rows = []
    for recovery_kind in ("half", "full"):
        print(f"\n[recovery] kind={recovery_kind}")
        env = None
        try:
            env_cfg, train_cfg = clone_cfgs(args.task)
            configure_env_cfg(
                env_cfg,
                args,
                "flat",
                args.trials_per_case,
                1,
                "recovery",
                recovery_kind=recovery_kind,
            )
            env, _ = task_registry.make_env(
                name=args.task,
                args=make_case_args(args),
                env_cfg=env_cfg,
            )
            env.cfg.terrain.curriculum = False
            runner, _, load_path = build_runner(env, train_cfg, args)
            policy = runner.get_inference_policy(device=env.device)
            force_levels_and_reset(env, [0], args.trials_per_case)
            reset_runner_state(runner, env)
            case_rows, trial_rows = evaluate_recovery(env, runner, policy, args, recovery_kind)
            annotate_rows(case_rows, load_path)
            print_case_rows(case_rows)
            all_case_rows.extend(case_rows)
            all_trial_rows.extend(trial_rows)
        finally:
            destroy_env(env)
    return all_case_rows, all_trial_rows


def print_case_rows(rows):
    for row in rows:
        level = row["level"] if row["level"] != "" else row["difficulty"]
        print(
            f"  {row['category']:>8s} {row['terrain_alias']:<14s} "
            f"level={level!s:<5s} success={row['successes']:>4d}/"
            f"{row['trials']:<4d} rate={100.0 * row['success_rate']:6.2f}%"
        )


def write_outputs(out_dir, case_rows, trial_rows, args, load_info):
    out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = out_dir / "results.csv"
    trial_csv = out_dir / "trials.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    if case_rows:
        case_fields = sorted({key for row in case_rows for key in row.keys()})
        with case_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=case_fields)
            writer.writeheader()
            writer.writerows(case_rows)
    if trial_rows:
        trial_fields = sorted({key for row in trial_rows for key in row.keys()})
        with trial_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=trial_fields)
            writer.writeheader()
            writer.writerows(trial_rows)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": json_safe(vars(args)),
        "load_info": load_info,
        "results": json_safe(case_rows),
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Parkour MoE Evaluation",
        "",
        f"- task: `{args.task}`",
        f"- trials per case: `{args.trials_per_case}`",
        f"- levels: `{args.levels}` ({args.level_indexing})",
        f"- command speed: `{args.command_speed}`",
        f"- progress eval time: `env episode` when `--eval_time <= 0`, otherwise `--eval_time`",
        f"- forward/omni/blind success: first pre-reset progress from spawn/start point at terrain end",
        f"- forward vision flag: `true`; blind vision flag: `false`",
        f"- end tolerance: `{args.end_tolerance}`",
        "",
        "| category | terrain | level/difficulty | success | rate |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in case_rows:
        level = row["level"] if row["level"] != "" else row["difficulty"]
        lines.append(
            f"| {row['category']} | {row['terrain_alias']} | {level} | "
            f"{row['successes']}/{row['trials']} | {100.0 * row['success_rate']:.2f}% |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return case_csv, trial_csv, summary_json, summary_md


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    args = parse_args()
    set_seed(args.seed if args.seed is not None else 1)

    level_labels = parse_levels(args.levels)
    row_indices = [level_to_row(level, args.level_indexing) for level in level_labels]
    if args.trials_per_case <= 0:
        raise ValueError("--trials_per_case must be positive.")

    forward_terrains = parse_terrain_list(args.forward_terrains, include_flat=False)
    omni_terrains = parse_terrain_list(args.omni_terrains, include_flat=False)
    blind_terrains = parse_terrain_list(args.blind_terrains, include_flat=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "go2_parkour_moe" / "eval" / timestamp

    all_case_rows = []
    all_trial_rows = []
    load_info = {
        "experiment_name": args.experiment_name or "config_default",
        "load_run": args.load_run if args.load_run is not None else "config_default",
        "checkpoint": args.checkpoint if args.checkpoint is not None else "config_default",
    }

    if not args.skip_forward:
        rows, trials = run_progress_suite(args, level_labels, row_indices, forward_terrains, "forward")
        all_case_rows.extend(rows)
        all_trial_rows.extend(trials)
    if not args.skip_recovery:
        rows, trials = run_recovery_suite(args)
        all_case_rows.extend(rows)
        all_trial_rows.extend(trials)
    if not args.skip_omni:
        rows, trials = run_progress_suite(args, level_labels, row_indices, omni_terrains, "omni")
        all_case_rows.extend(rows)
        all_trial_rows.extend(trials)
    if not args.skip_blind:
        rows, trials = run_progress_suite(args, level_labels, row_indices, blind_terrains, "blind")
        all_case_rows.extend(rows)
        all_trial_rows.extend(trials)

    outputs = write_outputs(out_dir, all_case_rows, all_trial_rows, args, load_info)
    print("\nSaved evaluation outputs:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
