import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math
import sys
from collections import deque
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_apply, quat_mul, quat_rotate_inverse
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.math import quat_apply_yaw
from legged_gym.utils.exporter import export_policy_as_jit, export_policy_as_onnx, export_policy_as_pkl

import numpy as np
import torch
import pygame


def init_depth_viewer(depth_tensor, tile_scale=2, max_cols=10, extra_panel_width=0):
    num_envs, _, height, width = depth_tensor.shape
    cols = min(max_cols, max(1, math.ceil(math.sqrt(num_envs))))
    rows = math.ceil(num_envs / cols)
    content_width = cols * width * tile_scale
    content_height = rows * height * tile_scale
    window_size = (content_width + extra_panel_width, content_height)
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Proxy Depth Cameras")
    font = pygame.font.SysFont("Consolas", 16)
    return screen, cols, rows, height, width, tile_scale, font, content_width, content_height, extra_panel_width


def draw_gating_history_panel(screen, panel_rect, font, gating_history, env_id):
    pygame.draw.rect(screen, (18, 18, 24), panel_rect)
    pygame.draw.rect(screen, (70, 70, 85), panel_rect, width=1)

    title = font.render(f"Env {env_id} Gating Weights", True, (235, 235, 235))
    screen.blit(title, (panel_rect.x + 10, panel_rect.y + 8))

    if not gating_history:
        empty_text = font.render("Waiting for estimator output...", True, (170, 170, 170))
        screen.blit(empty_text, (panel_rect.x + 10, panel_rect.y + 34))
        return

    history = np.asarray(gating_history, dtype=np.float32)
    num_experts = history.shape[1]
    colors = [
        (239, 83, 80),
        (66, 165, 245),
        (102, 187, 106),
        (255, 202, 40),
        (171, 71, 188),
        (38, 198, 218),
        (255, 112, 67),
        (156, 204, 101),
    ]

    plot_margin_left = 44
    plot_margin_right = 12
    plot_margin_top = 34
    plot_margin_bottom = 48
    plot_rect = pygame.Rect(
        panel_rect.x + plot_margin_left,
        panel_rect.y + plot_margin_top,
        max(1, panel_rect.width - plot_margin_left - plot_margin_right),
        max(1, panel_rect.height - plot_margin_top - plot_margin_bottom),
    )

    pygame.draw.rect(screen, (28, 28, 36), plot_rect)
    pygame.draw.rect(screen, (95, 95, 110), plot_rect, width=1)

    for value in (0.0, 0.5, 1.0):
        y = int(plot_rect.bottom - value * plot_rect.height)
        pygame.draw.line(screen, (55, 55, 65), (plot_rect.left, y), (plot_rect.right, y), width=1)
        label = font.render(f"{value:.1f}", True, (170, 170, 170))
        screen.blit(label, (panel_rect.x + 8, y - label.get_height() // 2))

    if history.shape[0] >= 2:
        x_coords = np.linspace(plot_rect.left, plot_rect.right, num=history.shape[0], dtype=np.float32)
        for expert_idx in range(num_experts):
            y_coords = plot_rect.bottom - np.clip(history[:, expert_idx], 0.0, 1.0) * plot_rect.height
            points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
            if len(points) >= 2:
                pygame.draw.lines(screen, colors[expert_idx % len(colors)], False, points, width=2)

    latest = history[-1]
    legend_y = plot_rect.bottom + 10
    for expert_idx, weight in enumerate(latest):
        legend = font.render(f"E{expert_idx}: {weight:.3f}", True, colors[expert_idx % len(colors)])
        screen.blit(legend, (panel_rect.x + 10, legend_y))
        legend_y += legend.get_height() + 2


def update_depth_viewer(
    screen,
    depth_tensor,
    cols,
    height,
    width,
    tile_scale,
    font,
    content_width,
    content_height,
    panel_width,
    overlay_lines=None,
    gating_history=None,
    gating_env_id=0,
    token_dropout_mask=None,
    token_grid_shape=None,
    show_token_dropout_overlay=False,
    active_token_coords=None,
    active_token_valid_mass=None,
    show_active_token_overlay=False,
):
    latest_depth = depth_tensor[:, -1].detach().cpu().float()
    latest_depth = (latest_depth + 0.5).clamp(0.0, 1.0)
    latest_depth = (latest_depth * 255.0).to(torch.uint8).numpy()

    token_dropout_grids = None
    if show_token_dropout_overlay and token_dropout_mask is not None:
        token_dropout_grids = token_dropout_mask.detach().cpu().bool()
        if token_dropout_grids.ndim == 3:
            token_dropout_grids = token_dropout_grids[:, -1]
        elif token_dropout_grids.ndim == 1:
            token_dropout_grids = token_dropout_grids.unsqueeze(0)
        if token_dropout_grids.ndim != 2:
            token_dropout_grids = None

    token_height = token_width = 0
    if token_dropout_grids is not None:
        if token_grid_shape is not None:
            token_height, token_width = int(token_grid_shape[0]), int(token_grid_shape[1])
        token_count = int(token_dropout_grids.shape[-1])
        if token_height <= 0 or token_width <= 0 or token_height * token_width != token_count:
            if token_count == 88:
                token_height, token_width = 8, 11
            else:
                token_height, token_width = 1, token_count
        if token_height * token_width != token_count:
            token_dropout_grids = None

    active_token_points = None
    active_token_masses = None
    if show_active_token_overlay and active_token_coords is not None:
        active_token_points = active_token_coords.detach().cpu().float()
        if active_token_points.ndim == 4:
            active_token_points = active_token_points[:, -1]
        elif active_token_points.ndim == 2:
            active_token_points = active_token_points.unsqueeze(0)
        if active_token_points.ndim != 3 or active_token_points.shape[-1] != 2:
            active_token_points = None
        if active_token_points is not None and active_token_valid_mass is not None:
            active_token_masses = active_token_valid_mass.detach().cpu().float()
            if active_token_masses.ndim == 3:
                active_token_masses = active_token_masses[:, -1]
            elif active_token_masses.ndim == 1:
                active_token_masses = active_token_masses.unsqueeze(0)
            if active_token_masses.ndim != 2:
                active_token_masses = None

    screen.fill((0, 0, 0))
    for env_id, frame in enumerate(latest_depth):
        rgb_frame = np.repeat(frame[:, :, None], 3, axis=2)
        surface = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
        if tile_scale != 1:
            surface = pygame.transform.scale(surface, (width * tile_scale, height * tile_scale))
        col = env_id % cols
        row = env_id // cols
        tile_x = col * width * tile_scale
        tile_y = row * height * tile_scale
        screen.blit(surface, (tile_x, tile_y))
        if token_dropout_grids is not None and env_id < token_dropout_grids.shape[0]:
            token_grid = token_dropout_grids[env_id].reshape(token_height, token_width)
            if torch.any(token_grid):
                overlay = pygame.Surface((width * tile_scale, height * tile_scale), pygame.SRCALPHA)
                for token_row in range(token_height):
                    y0 = round(token_row * height * tile_scale / token_height)
                    y1 = round((token_row + 1) * height * tile_scale / token_height)
                    for token_col in range(token_width):
                        if not bool(token_grid[token_row, token_col]):
                            continue
                        x0 = round(token_col * width * tile_scale / token_width)
                        x1 = round((token_col + 1) * width * tile_scale / token_width)
                        rect = pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
                        pygame.draw.rect(overlay, (255, 42, 42, 112), rect)
                        pygame.draw.rect(overlay, (255, 30, 30, 210), rect, width=1)
                screen.blit(overlay, (tile_x, tile_y))
                count_text = font.render(
                    f"drop {int(token_grid.sum())}/{token_grid.numel()}",
                    True,
                    (255, 235, 235),
                )
                screen.blit(count_text, (tile_x + 4, tile_y + 4))
        if active_token_points is not None and env_id < active_token_points.shape[0]:
            points = active_token_points[env_id]
            masses = active_token_masses[env_id] if active_token_masses is not None and env_id < active_token_masses.shape[0] else None
            valid_count = 0
            for token_idx, coord in enumerate(points):
                x_norm = float(coord[0].clamp(-1.0, 1.0))
                y_norm = float(coord[1].clamp(-1.0, 1.0))
                x = tile_x + int(round((x_norm + 1.0) * 0.5 * (width - 1) * tile_scale))
                y = tile_y + int(round((y_norm + 1.0) * 0.5 * (height - 1) * tile_scale))
                valid = True
                if masses is not None and token_idx < masses.shape[0]:
                    valid = float(masses[token_idx]) > 0.05
                color = (40, 240, 255) if valid else (130, 130, 130)
                outline = (5, 20, 24) if valid else (35, 35, 35)
                radius = max(3, int(round(3 * tile_scale)))
                pygame.draw.circle(screen, outline, (x, y), radius + 1)
                pygame.draw.circle(screen, color, (x, y), radius)
                valid_count += int(valid)
            count_text = font.render(
                f"probe {valid_count}/{points.shape[0]}",
                True,
                (210, 255, 255),
            )
            screen.blit(count_text, (tile_x + 4, tile_y + 22))
    if overlay_lines:
        y = 8
        for line in overlay_lines:
            text = font.render(line, True, (255, 220, 120))
            screen.blit(text, (8, y))
            y += text.get_height() + 2
    if panel_width > 0:
        panel_rect = pygame.Rect(content_width, 0, panel_width, content_height)
        draw_gating_history_panel(screen, panel_rect, font, gating_history, gating_env_id)
    pygame.display.flip()


def get_velocity_overlay_lines(env, env_id):
    env_id = min(env_id, env.num_envs - 1)
    cmd = env.commands[env_id, :3].detach().cpu()
    cur_lin = env.base_lin_vel[env_id, :3].detach().cpu()
    cur_ang = env.base_ang_vel[env_id, 2].detach().cpu()
    return [
        f"env {env_id}",
        f"cmd    vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}",
        f"curr   vx={cur_lin[0]:+.2f} vy={cur_lin[1]:+.2f} vz={cur_lin[2]:+.2f}",
        f"curr_w wz={cur_ang:+.2f}",
    ]


def get_height_scan_world_points(env, env_id):
    if not hasattr(env, "height_points") or not torch.is_tensor(env.measured_heights):
        return None
    env_id = min(env_id, env.num_envs - 1)
    if env.measured_heights.ndim != 2:
        return None

    local_points = env.height_points[env_id:env_id + 1]
    base_quat = env.base_quat[env_id:env_id + 1]
    base_pos = env.root_states[env_id:env_id + 1, :3]
    world_points = quat_apply_yaw(
        base_quat.repeat(1, env.num_height_points),
        local_points,
    ) + base_pos.unsqueeze(1)
    return world_points.squeeze(0)


def draw_scan_points(env, env_id, world_z_values, sphere_geom, z_offset=0.02):
    if env.viewer is None or world_z_values is None:
        return
    world_points = get_height_scan_world_points(env, env_id)
    if world_points is None:
        return
    world_points = world_points.clone()
    world_points[:, 2] = world_z_values + z_offset

    for point in world_points.detach().cpu().numpy():
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(point[0]), float(point[1]), float(point[2]))
        gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[env_id], pose)


def draw_world_points(env, env_id, world_points, sphere_geom):
    if env.viewer is None or world_points is None:
        return
    env_id = min(env_id, env.num_envs - 1)
    for point in world_points.detach().cpu().numpy():
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(point[0]), float(point[1]), float(point[2]))
        gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[env_id], pose)


def draw_targeted_footholds(env, env_id, sphere_geoms, z_offset=0.025):
    if (
        env.viewer is None
        or not hasattr(env, "targeted_foothold_targets_world")
        or not hasattr(env, "targeted_foothold_target_valid")
        or sphere_geoms is None
    ):
        return
    env_id = min(env_id, env.num_envs - 1)
    target_points = env.targeted_foothold_targets_world[env_id]
    target_valid = env.targeted_foothold_target_valid[env_id]
    for leg_idx, geom in enumerate(sphere_geoms):
        if leg_idx >= target_points.shape[0] or not bool(target_valid[leg_idx]):
            continue
        point = target_points[leg_idx].detach().cpu().numpy().copy()
        point[2] += z_offset
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(point[0]), float(point[1]), float(point[2]))
        gymutil.draw_lines(geom, env.gym, env.viewer, env.envs[env_id], pose)


def get_camera_world_pose(env, env_id):
    if (
        env.viewer is None
        or not hasattr(env, "camera_local_pos")
        or not hasattr(env, "camera_ray_dirs_base")
    ):
        return None

    env_id = min(env_id, env.num_envs - 1)
    base_quat = env.base_quat[env_id : env_id + 1]
    base_pos = env.root_states[env_id : env_id + 1, :3]
    if hasattr(env, "camera_local_pos_env"):
        camera_local_pos = env.camera_local_pos_env[env_id : env_id + 1]
    else:
        camera_local_pos = env.camera_local_pos.unsqueeze(0)
    camera_pos = base_pos + quat_apply(base_quat, camera_local_pos)
    camera_ray_quat = base_quat
    if hasattr(env, "camera_yaw_quat"):
        camera_ray_quat = quat_mul(base_quat, env.camera_yaw_quat[env_id : env_id + 1])

    raw_height, raw_width = getattr(
        env,
        "raw_depth_shape",
        (env.cfg.camera.raw_height, env.cfg.camera.raw_width),
    )
    center_idx = (raw_height // 2) * raw_width + (raw_width // 2)
    corner_indices = [
        0,
        raw_width - 1,
        (raw_height - 1) * raw_width,
        raw_height * raw_width - 1,
    ]
    ray_indices = [center_idx] + corner_indices
    ray_dirs_base = env.camera_ray_dirs_base[ray_indices]
    ray_dirs_world = quat_apply(camera_ray_quat.repeat(len(ray_indices), 1), ray_dirs_base)

    return {
        "position": camera_pos.squeeze(0),
        "forward": ray_dirs_world[0],
        "corners": ray_dirs_world[1:],
    }


def get_camera_hit_points_world(env, env_id, row_stride=4, col_stride=4):
    if (
        env.viewer is None
        or not hasattr(env, "camera_local_pos")
        or not hasattr(env, "camera_ray_dirs_base")
        or not hasattr(env, "_render_proxy_distance_to_camera")
    ):
        return None

    env_id = min(env_id, env.num_envs - 1)
    row_stride = max(1, int(row_stride))
    col_stride = max(1, int(col_stride))

    raw_height, raw_width = getattr(
        env,
        "raw_depth_shape",
        (env.cfg.camera.raw_height, env.cfg.camera.raw_width),
    )
    row_ids = torch.arange(0, raw_height, row_stride, device=env.device, dtype=torch.long)
    col_ids = torch.arange(0, raw_width, col_stride, device=env.device, dtype=torch.long)
    grid_rows, grid_cols = torch.meshgrid(row_ids, col_ids, indexing="ij")
    ray_indices = (grid_rows * raw_width + grid_cols).reshape(-1)

    env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    ray_distances = env._render_proxy_distance_to_camera(env_ids=env_ids)[0]
    sampled_distances = ray_distances[grid_rows, grid_cols].reshape(-1)
    hit_mask = sampled_distances < (env.cfg.camera.clipping_range - 1.0e-3)
    if not torch.any(hit_mask):
        return None

    base_quat = env.base_quat[env_id : env_id + 1]
    base_pos = env.root_states[env_id : env_id + 1, :3]
    if hasattr(env, "camera_local_pos_env"):
        camera_local_pos = env.camera_local_pos_env[env_id : env_id + 1]
    else:
        camera_local_pos = env.camera_local_pos.unsqueeze(0)
    camera_pos = base_pos + quat_apply(base_quat, camera_local_pos)
    camera_ray_quat = base_quat
    if hasattr(env, "camera_yaw_quat"):
        camera_ray_quat = quat_mul(base_quat, env.camera_yaw_quat[env_id : env_id + 1])
    ray_dirs_world = quat_apply(
        camera_ray_quat.repeat(ray_indices.numel(), 1),
        env.camera_ray_dirs_base[ray_indices],
    )
    hit_points = camera_pos.repeat(ray_indices.numel(), 1) + ray_dirs_world * sampled_distances.unsqueeze(-1)
    return hit_points[hit_mask]


def draw_camera_pose_marker(
    env,
    env_id,
    origin_geom,
    forward_geom,
    corner_geom,
    forward_distance=0.35,
    corner_distance=0.55,
):
    camera_pose = get_camera_world_pose(env, env_id)
    if camera_pose is None:
        return

    marker_points = [camera_pose["position"]]
    marker_geoms = [origin_geom]

    marker_points.append(camera_pose["position"] + camera_pose["forward"] * forward_distance)
    marker_geoms.append(forward_geom)

    for corner_dir in camera_pose["corners"]:
        marker_points.append(camera_pose["position"] + corner_dir * corner_distance)
        marker_geoms.append(corner_geom)

    env_id = min(env_id, env.num_envs - 1)
    for point, geom in zip(marker_points, marker_geoms):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(point[0]), float(point[1]), float(point[2]))
        gymutil.draw_lines(geom, env.gym, env.viewer, env.envs[env_id], pose)


def update_viewer_to_camera_pose(env, env_id, look_ahead=1.2):
    camera_pose = get_camera_world_pose(env, env_id)
    if camera_pose is None:
        return

    camera_pos = camera_pose["position"]
    look_at = camera_pos + camera_pose["forward"] * look_ahead
    env.gym.viewer_camera_look_at(
        env.viewer,
        None,
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Vec3(float(look_at[0]), float(look_at[1]), float(look_at[2])),
    )


def update_viewer_follow_robot(env, env_id=0):
    env_id = min(env_id, env.num_envs - 1)
    robot_pos = env.root_states[env_id, :3]
    camera_pos = robot_pos + torch.tensor([-4.0, -3.0, 2.0], device=env.device)
    look_at = robot_pos + torch.tensor([0.8, 0.0, 0.3], device=env.device)

    env.gym.viewer_camera_look_at(
        env.viewer,
        None,
        gymapi.Vec3(*camera_pos.detach().cpu().tolist()),
        gymapi.Vec3(*look_at.detach().cpu().tolist()),
    )


def get_reconstructed_height_world(env, runner, env_id):
    if not hasattr(runner, "latest_inference_m_hat") or runner.latest_inference_m_hat is None:
        return None
    env_id = min(env_id, env.num_envs - 1)
    if runner.latest_inference_m_hat.shape[-1] != env.num_height_points:
        return None
    obs_scale = env.obs_scales.height_measurements
    root_z = env.root_states[env_id, 2]
    pred_rel_heights = runner.latest_inference_m_hat[env_id]
    return root_z - 0.5 - pred_rel_heights / obs_scale


def append_gating_history(history_buffer, runner, env_id):
    if not hasattr(runner, "latest_inference_gating_weights"):
        return
    if runner.latest_inference_gating_weights is None:
        return
    env_id = min(env_id, runner.latest_inference_gating_weights.shape[0] - 1)
    history_buffer.append(
        runner.latest_inference_gating_weights[env_id].detach().cpu().float().numpy().copy()
    )


def resolve_play_command(env, args):
    """Resolve an optional fixed replay command without assuming a task speed."""
    overrides = (args.command_vx, args.command_vy, args.command_yaw)
    if args.use_sampled_commands:
        if any(value is not None for value in overrides):
            raise ValueError(
                "--use_sampled_commands cannot be combined with command overrides."
            )
        return None

    configured_command = getattr(env.cfg.commands, "play_command", None)
    if configured_command is None and all(value is None for value in overrides):
        return None

    command = list(configured_command) if configured_command is not None else [0.0, 0.0, 0.0]
    if len(command) < 3:
        raise ValueError(
            "env.cfg.commands.play_command must contain [vx, vy, yaw_rate]."
        )
    command = command[:3]
    for index, value in enumerate(overrides):
        if value is not None:
            command[index] = value
    return torch.tensor(command, dtype=torch.float, device=env.device)


def apply_play_command(env, fixed_command):
    """Apply a task-configured or CLI-overridden fixed command, if requested."""
    if fixed_command is None:
        return
    if hasattr(env, "omni_world_lin_vel_x") and hasattr(env, "_omni_env_mask"):
        omni_mask = env._omni_env_mask()
        if torch.any(omni_mask):
            env.omni_world_lin_vel_x[omni_mask] = fixed_command[0]
            env._apply_omni_world_commands()
        non_omni_mask = ~omni_mask
        env.commands[non_omni_mask, :3] = fixed_command
    else:
        env.commands[:, :3] = fixed_command


def prepare_playback_start(env, fixed_command):
    """Create a clean, observation-consistent state after runner construction.

    Runner construction calls ``env.reset()``, which advances the physics once with
    zero actions. Reset once more without stepping so the first policy action sees
    exactly the state that is displayed by the simulator.
    """
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)

    # Indexed state setters used by reset_idx update the simulator. Refresh their
    # tensor views before deriving the quantities consumed by observations.
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    if hasattr(env, "base_quat"):
        env.base_quat[:] = env.root_states[:, 3:7]
    if hasattr(env, "base_pos"):
        env.base_pos[:] = env.root_states[:, :3]
    if hasattr(env, "rpy"):
        env.rpy[:] = get_euler_xyz_in_tensor(env.root_states[:, 3:7])
    if hasattr(env, "base_lin_vel"):
        env.base_lin_vel[:] = quat_rotate_inverse(
            env.root_states[:, 3:7], env.root_states[:, 7:10]
        )
    if hasattr(env, "base_ang_vel"):
        env.base_ang_vel[:] = quat_rotate_inverse(
            env.root_states[:, 3:7], env.root_states[:, 10:13]
        )
    if hasattr(env, "projected_gravity"):
        env.projected_gravity[:] = quat_rotate_inverse(
            env.root_states[:, 3:7], env.gravity_vec
        )
    if hasattr(env, "last_dof_vel"):
        env.last_dof_vel[env_ids] = env.dof_vel[env_ids]
    if hasattr(env, "last_root_vel"):
        env.last_root_vel[env_ids] = env.root_states[env_ids, 7:13]

    # Refresh terrain/depth observations that reset_idx deliberately invalidates.
    if hasattr(env, "_post_physics_step_callback"):
        env._post_physics_step_callback()
    apply_play_command(env, fixed_command)
    env.compute_observations()
    return env.get_observations()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 20 if args.continuous_replay else 7
    env_cfg.terrain.num_cols = 7
    env_cfg.terrain.curriculum = False
    if args.continuous_replay:
        env_cfg.terrain.reset_when_outside_block = False
        if args.play_duration is None or args.play_duration < 0.0:
            env_cfg.env.episode_length_s = 1.0e9
        else:
            env_cfg.env.episode_length_s = max(
                float(env_cfg.env.episode_length_s),
                float(args.play_duration) + 1.0,
            )
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False

    if env_cfg.init_state.turn_over:
        env_cfg.init_state.turn_over_proportions = [0.0, 1.0, 0.0]


    env_cfg.env.test = True
    if hasattr(env_cfg, "camera"):
        env_cfg.camera.source = "proxy"
    env_cfg.env.enable_camera_sensors = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    if train_cfg.runner_class_name == "OnPolicyRunnerParkourMoE":
        policy = runner.get_inference_policy(
            device=env.device,
            random_token_dropout=PLAY_RANDOM_TOKEN_DROPOUT,
            token_dropout_min=PLAY_TOKEN_DROPOUT_MIN,
            token_dropout_max=PLAY_TOKEN_DROPOUT_MAX,
        )
    else:
        policy = runner.get_inference_policy(device=env.device)

    fixed_command = resolve_play_command(env, args)
    obs = prepare_playback_start(env, fixed_command)
    if fixed_command is None:
        print("Replay command source: environment sampling")
    else:
        command_values = fixed_command.detach().cpu().tolist()
        print(
            "Replay fixed command: "
            f"vx={command_values[0]:.3f} m/s, "
            f"vy={command_values[1]:.3f} m/s, "
            f"yaw_rate={command_values[2]:.3f} rad/s"
        )

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        if hasattr(runner, 'export_policy'):
            runner.export_policy(path)
            print('Exported policy to: ', path)
        else:
            if hasattr(runner.alg, 'actor_critic'):
                model = runner.alg.actor_critic
            else:
                model = runner.alg.model
            export_policy_as_jit(model, path)
            export_policy_as_onnx(model, path)
            export_policy_as_pkl(model, path)
            print('Exported policy as jit script / onnx to: ', path)

    depth_viewer = None
    depth_layout = None
    height_scan_geom = None
    recon_scan_geom = None
    camera_origin_geom = None
    camera_forward_geom = None
    camera_corner_geom = None
    camera_hit_geom = None
    targeted_foothold_geoms = None
    cached_camera_hit_points = None
    last_camera_hit_update_step = -1
    gating_history = deque(maxlen=GATING_HISTORY_LENGTH)
    viewer_panel_width = GATING_PANEL_WIDTH if SHOW_GATING_WEIGHTS else 0
    additional_obs = env.get_additional_observations()
    if "depth_camera" in additional_obs:
        depth_viewer = init_depth_viewer(
            additional_obs["depth_camera"],
            extra_panel_width=viewer_panel_width,
        )
        depth_layout = (
            depth_viewer[1],
            depth_viewer[3],
            depth_viewer[4],
            depth_viewer[5],
            depth_viewer[6],
            depth_viewer[7],
            depth_viewer[8],
            depth_viewer[9],
        )
    if (
        SHOW_HEIGHT_SCAN
        and env.viewer is not None
        and hasattr(env, "height_points")
        and torch.is_tensor(env.measured_heights)
        and env.measured_heights.ndim == 2
    ):
        height_scan_geom = gymutil.WireframeSphereGeometry(0.015, 6, 6, None, color=(0.1, 0.9, 0.1))
    if SHOW_RECON_SCAN and env.viewer is not None:
        recon_scan_geom = gymutil.WireframeSphereGeometry(0.012, 6, 6, None, color=(0.95, 0.45, 0.1))
    if SHOW_CAMERA_POSE and env.viewer is not None:
        camera_origin_geom = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(0.95, 0.2, 0.15))
        camera_forward_geom = gymutil.WireframeSphereGeometry(0.02, 6, 6, None, color=(0.95, 0.82, 0.2))
        camera_corner_geom = gymutil.WireframeSphereGeometry(0.012, 5, 5, None, color=(0.25, 0.85, 0.95))
    if SHOW_CAMERA_HIT_POINTS and env.viewer is not None:
        camera_hit_geom = gymutil.WireframeSphereGeometry(0.008, 4, 4, None, color=(0.95, 0.2, 0.2))
    if SHOW_TARGETED_FOOTHOLDS and env.viewer is not None:
        targeted_foothold_geoms = [
            gymutil.WireframeSphereGeometry(0.022, 6, 6, None, color=(0.95, 0.25, 0.2)),
            gymutil.WireframeSphereGeometry(0.022, 6, 6, None, color=(0.2, 0.55, 0.95)),
            gymutil.WireframeSphereGeometry(0.022, 6, 6, None, color=(0.15, 0.85, 0.35)),
            gymutil.WireframeSphereGeometry(0.022, 6, 6, None, color=(0.95, 0.82, 0.2)),
        ]

    if args.play_duration is None:
        play_steps = 10 * int(env.max_episode_length)
    elif args.play_duration < 0.0:
        play_steps = None
    else:
        play_steps = max(0, int(math.ceil(args.play_duration / env.dt)))

    i = 0
    while play_steps is None or i < play_steps:
        if depth_viewer is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        if fixed_command is not None:
            # env.step() may have reset an environment and built its returned
            # observation with a newly sampled command. Keep the command stored
            # in the environment and the command seen by the policy identical.
            commands_before = env.commands[:, :3].clone()
            apply_play_command(env, fixed_command)
            if not torch.equal(commands_before, env.commands[:, :3]):
                env.compute_observations()
                obs = env.get_observations()

        actions = policy(obs.detach())
        if SHOW_GATING_WEIGHTS:
            append_gating_history(gating_history, runner, DEBUG_ENV_ID)

        if args.follow_camera and env.viewer is not None:
            update_viewer_follow_robot(env, DEBUG_ENV_ID)

        if (
            height_scan_geom is not None
            or recon_scan_geom is not None
            or camera_origin_geom is not None
            or camera_hit_geom is not None
        ):
            env.gym.clear_lines(env.viewer)
            if height_scan_geom is not None:
                draw_scan_points(
                    env,
                    DEBUG_ENV_ID,
                    env.measured_heights[min(DEBUG_ENV_ID, env.num_envs - 1)],
                    height_scan_geom,
                    z_offset=0.02,
                )
            if recon_scan_geom is not None:
                draw_scan_points(
                    env,
                    DEBUG_ENV_ID,
                    get_reconstructed_height_world(env, runner, DEBUG_ENV_ID),
                    recon_scan_geom,
                    z_offset=0.035,
                )
            if camera_hit_geom is not None:
                update_interval = max(1, CAMERA_HIT_POINT_UPDATE_INTERVAL)
                current_step = int(env.common_step_counter)
                if (
                    cached_camera_hit_points is None
                    or last_camera_hit_update_step < 0
                    or current_step - last_camera_hit_update_step >= update_interval
                ):
                    cached_camera_hit_points = get_camera_hit_points_world(
                        env,
                        DEBUG_ENV_ID,
                        row_stride=CAMERA_HIT_POINT_ROW_STRIDE,
                        col_stride=CAMERA_HIT_POINT_COL_STRIDE,
                    )
                    last_camera_hit_update_step = current_step
                draw_world_points(
                    env,
                    DEBUG_ENV_ID,
                    cached_camera_hit_points,
                    camera_hit_geom,
                )
            if camera_origin_geom is not None:
                draw_camera_pose_marker(
                    env,
                    DEBUG_ENV_ID,
                    camera_origin_geom,
                    camera_forward_geom,
                    camera_corner_geom,
                )
            if targeted_foothold_geoms is not None:
                draw_targeted_footholds(
                    env,
                    DEBUG_ENV_ID,
                    targeted_foothold_geoms,
                )
        if depth_viewer is not None:
            update_depth_viewer(
                depth_viewer[0],
                env.get_additional_observations()["depth_camera"],
                *depth_layout,
                overlay_lines=get_velocity_overlay_lines(env, DEBUG_ENV_ID) if SHOW_VELOCITY_TEXT else None,
                gating_history=gating_history if SHOW_GATING_WEIGHTS else None,
                gating_env_id=DEBUG_ENV_ID,
                token_dropout_mask=getattr(runner, "latest_inference_token_dropout_mask", None),
                token_grid_shape=getattr(runner, "latest_inference_token_grid_shape", None),
                show_token_dropout_overlay=SHOW_TOKEN_DROPOUT_OVERLAY,
                active_token_coords=getattr(runner, "latest_inference_active_token_coords", None),
                active_token_valid_mass=getattr(runner, "latest_inference_active_token_valid_mass", None),
                show_active_token_overlay=SHOW_ACTIVE_TOKEN_OVERLAY,
            )
        elif SHOW_VELOCITY_TEXT and i % 20 == 0:
            print(" | ".join(get_velocity_overlay_lines(env, DEBUG_ENV_ID)))
        obs, _, rews, dones, infos = env.step(actions.detach())
        i += 1

    if depth_viewer is not None:
        pygame.quit()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    SHOW_HEIGHT_SCAN = True
    SHOW_RECON_SCAN = True
    SHOW_CAMERA_POSE = False
    SHOW_CAMERA_HIT_POINTS = True
    SHOW_TARGETED_FOOTHOLDS = True
    SHOW_GATING_WEIGHTS = True
    SHOW_VELOCITY_TEXT = False
    PLAY_RANDOM_TOKEN_DROPOUT = True
    SHOW_TOKEN_DROPOUT_OVERLAY = True
    SHOW_ACTIVE_TOKEN_OVERLAY = True
    PLAY_TOKEN_DROPOUT_MIN = None
    PLAY_TOKEN_DROPOUT_MAX = None
    CAMERA_HIT_POINT_ROW_STRIDE = 4
    CAMERA_HIT_POINT_COL_STRIDE = 4
    CAMERA_HIT_POINT_UPDATE_INTERVAL = 5
    GATING_HISTORY_LENGTH = 240
    GATING_PANEL_WIDTH = 360
    DEBUG_ENV_ID = 0
    args = get_args(
        additional_parameters=[
            {
                "name": "--follow_camera",
                "action": "store_true",
                "default": False,
                "help": "Keep the replay viewer camera following the debug robot.",
            },
            {
                "name": "--play_duration",
                "type": float,
                "default": None,
                "help": (
                    "Replay duration in simulated seconds. Omit to keep the existing "
                    "10-episode default; use -1 to replay until the window is closed."
                ),
            },
            {
                "name": "--continuous_replay",
                "action": "store_true",
                "default": False,
                "help": (
                    "Disable terrain-boundary and episode-timeout resets during replay. "
                    "True falls and other failure terminations still reset the robot."
                ),
            },
            {
                "name": "--use_sampled_commands",
                "action": "store_true",
                "default": False,
                "help": (
                    "Use commands sampled by the environment instead of a fixed "
                    "replay command."
                ),
            },
            {
                "name": "--command_vx",
                "type": float,
                "default": None,
                "help": "Override the fixed replay forward velocity in m/s.",
            },
            {
                "name": "--command_vy",
                "type": float,
                "default": None,
                "help": "Override the fixed replay lateral velocity in m/s.",
            },
            {
                "name": "--command_yaw",
                "type": float,
                "default": None,
                "help": "Override the fixed replay yaw rate in rad/s.",
            },
        ]
    )
    play(args)
