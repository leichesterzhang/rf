import math
import numpy as np
import torch
import torch.nn.functional as F

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul, quat_rotate_inverse, torch_rand_float

from legged_gym.envs.go2.go2_env import Go2Robot
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
try:
    import warp as wp
except Exception:  # pragma: no cover - optional dependency
    wp = None


_WARP_INITIALIZED = False


if wp is not None:
    @wp.kernel
    def _raycast_mesh_distance_kernel(
        mesh: wp.uint64,
        ray_origins: wp.array(dtype=wp.vec3),
        ray_directions: wp.array(dtype=wp.vec3),
        hit_distances: wp.array(dtype=float),
        hits_found: wp.array(dtype=wp.int32),
        max_dist: float,
    ):
        tid = wp.tid()

        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        face_index = int(0)

        hit = wp.mesh_query_ray(
            mesh,
            ray_origins[tid],
            ray_directions[tid],
            max_dist,
            t,
            u,
            v,
            sign,
            n,
            face_index,
        )

        if hit:
            hit_distances[tid] = t
            hits_found[tid] = 1
        else:
            hit_distances[tid] = max_dist
            hits_found[tid] = 0


def _ensure_warp_initialized():
    global _WARP_INITIALIZED
    if wp is None or _WARP_INITIALIZED:
        return
    wp.init()
    wp.config.quiet = True
    _WARP_INITIALIZED = True


def _torch_device_to_warp(device):
    device = torch.device(device)
    if device.type == "cpu":
        return "cpu"
    if device.index is None:
        return "cuda:0"
    return f"cuda:{device.index}"


class Go2ParkourRobot(Go2Robot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        cfg.camera.source = "proxy"
        cfg.env.enable_camera_sensors = False
        self._camera_source = "proxy"
        self._default_total_mass = None
        self._default_base_com = None
        self._depth_initialized = False
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.feet_height_dim = self.cfg.parkour.feet_height_dim
        self.height_map_dim = self.cfg.parkour.height_map_dim
        self.terrain_id_dim = 1
        self.fall_recovery_dim = 1
        self.base_privileged_obs_dim = (
            self.cfg.env.num_privileged_obs
            - self.feet_height_dim
            - self.terrain_id_dim
            - self.fall_recovery_dim
        )
        self.depth_buffer_len = self.cfg.camera.buffer_len
        self.raw_depth_shape = (self.cfg.camera.raw_height, self.cfg.camera.raw_width)
        self.depth_shape = (self.cfg.camera.output_height, self.cfg.camera.output_width)
        self.raycast_chunk_size = self.cfg.camera.raycast_chunk_size
        self.raycast_num_coarse_steps = self.cfg.camera.raycast_num_coarse_steps
        self.raycast_num_refine_steps = self.cfg.camera.raycast_num_refine_steps
        self.vt_privileged_slice = slice(0, 3)
        self.mt_privileged_slice = slice(
            self.base_privileged_obs_dim - self.height_map_dim,
            self.base_privileged_obs_dim,
        )
        self.ht_privileged_slice = slice(
            self.base_privileged_obs_dim,
            self.base_privileged_obs_dim + self.feet_height_dim,
        )
        self.terrain_id_privileged_slice = slice(
            self.base_privileged_obs_dim + self.feet_height_dim,
            self.base_privileged_obs_dim + self.feet_height_dim + self.terrain_id_dim,
        )
        self.fall_recovery_privileged_slice = slice(
            self.base_privileged_obs_dim + self.feet_height_dim + self.terrain_id_dim,
            self.cfg.env.num_privileged_obs,
        )

        self._force_depth_refresh = torch.ones(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        self.depth_camera_buf = torch.zeros(
            self.num_envs,
            self.depth_buffer_len,
            self.depth_shape[0],
            self.depth_shape[1],
            device=self.device,
            dtype=torch.float,
        )
        self.additional_obs_buf["depth_camera"] = self.depth_camera_buf
        self.clearance_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.float,
        )
        self.clearance_last_contacts = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.heading_conditioned_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "heading_conditioned_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.tracking_lin_vel_full_reward_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "tracking_lin_vel_full_reward_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.clearance_excluded_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "clearance_excluded_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.base_height_excluded_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "base_height_excluded_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.hip_to_default_excluded_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "hip_to_default_excluded_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.target_posture_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "target_posture_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.stand_still_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "stand_still_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.flat_gait_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "flat_gait_terrain_ids", [11]),
            device=self.device,
            dtype=torch.long,
        )
        self.targeted_foothold_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "targeted_foothold_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.no_progress_excluded_terrain_ids = torch.tensor(
            getattr(self.cfg.rewards, "no_progress_excluded_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self.flat_speed_search_enabled = bool(
            getattr(self.cfg.commands, "flat_speed_search_enabled", False)
        )
        self.flat_speed_search_terrain_ids = torch.tensor(
            getattr(self.cfg.commands, "flat_speed_search_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        self._init_omni_buffers()
        self.no_progress_steps = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        termination_cfg = getattr(self.cfg, "termination", None)
        default_fall_terrain_ids = list(range(len(getattr(self.cfg.terrain, "terrain_proportions", []))))
        self.fall_terrain_ids = torch.tensor(
            getattr(termination_cfg, "fall_terrain_ids", default_fall_terrain_ids),
            device=self.device,
            dtype=torch.long,
        )
        self.edge_last_contacts = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.flat_gait_last_contacts = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.flat_gait_last_contact_filt = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.flat_gait_stance_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.float,
        )
        self.flat_gait_contact_ema = torch.full(
            (self.num_envs, self.feet_indices.shape[0]),
            float(getattr(self.cfg.rewards, "flat_gait_min_contact_duty", 0.2)),
            device=self.device,
            dtype=torch.float,
        )
        self.targeted_foothold_last_contacts = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.targeted_foothold_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.float,
        )
        self.targeted_foothold_targets_world = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            3,
            device=self.device,
            dtype=torch.float,
        )
        self.targeted_foothold_target_valid = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.stairs_omni_last_contacts = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.stairs_omni_phase_active = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        self.stairs_omni_phase_timer = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self.stairs_omni_phase_anchor_progress = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self.stairs_omni_phase_prev_progress = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self.stairs_omni_phase_lead_mask = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.stairs_omni_phase_trail_mask = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self.stairs_omni_phase_trail_touched = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        self._stairs_omni_reward_cache_step = -1
        self._stairs_omni_reward_cache = {}
        self._build_camera_rays()
        self._init_camera_randomization_buffers()
        self._resample_camera_randomization(
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        )
        self._init_warp_proxy_backend()
        self.max_forward_progress = self._forward_progress().clone()

    def _init_omni_buffers(self):
        commands_cfg = getattr(self.cfg, "commands", None)
        enabled = bool(getattr(commands_cfg, "omni_enabled", False))
        terrain_ids = torch.tensor(
            getattr(commands_cfg, "omni_terrain_ids", []),
            device=self.device,
            dtype=torch.long,
        )
        if (
            self.flat_speed_search_enabled
            and bool(getattr(commands_cfg, "flat_speed_search_disable_omni", True))
            and terrain_ids.numel() > 0
            and self.flat_speed_search_terrain_ids.numel() > 0
        ):
            terrain_ids = terrain_ids[~torch.isin(terrain_ids, self.flat_speed_search_terrain_ids)]
        self.omni_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.omni_command_speed = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.omni_body_command_dir = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.omni_body_command_dir[:, 0] = 1.0
        self.omni_target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        if not enabled or terrain_ids.numel() == 0 or not hasattr(self, "terrain_ids"):
            return

        ratio = float(getattr(commands_cfg, "omni_ratio", 0.0))
        ratio = max(0.0, min(1.0, ratio))
        if ratio <= 0.0:
            return

        for terrain_id in terrain_ids.tolist():
            terrain_env_ids = (self.terrain_ids == int(terrain_id)).nonzero(as_tuple=False).flatten()
            if terrain_env_ids.numel() == 0:
                continue
            omni_count = int(round(float(terrain_env_ids.numel()) * ratio))
            omni_count = max(0, min(int(terrain_env_ids.numel()), omni_count))
            if omni_count == 0:
                continue
            perm = torch.randperm(terrain_env_ids.numel(), device=self.device)
            self.omni_mask[terrain_env_ids[perm[:omni_count]]] = True

    def _omni_env_mask(self, env_ids=None):
        omni_mask = getattr(self, "omni_mask", None)
        if omni_mask is None:
            if env_ids is None:
                return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        if env_ids is None:
            return omni_mask
        return omni_mask[env_ids]

    def _omni_disables_forward_rewards(self):
        return bool(getattr(self.cfg.commands, "omni_disable_forward_rewards", True))

    def _root_yaw(self, env_ids=None):
        quat = self.root_states[:, 3:7] if env_ids is None else self.root_states[env_ids, 3:7]
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _apply_omni_body_commands(self, env_ids=None):
        omni_mask = self._omni_env_mask(env_ids)
        if not torch.any(omni_mask):
            return
        if env_ids is None:
            target_env_ids = omni_mask.nonzero(as_tuple=False).flatten()
        else:
            target_env_ids = env_ids[omni_mask]
        yaw = self._root_yaw(target_env_ids)
        speed = self.omni_command_speed[target_env_ids]
        self.commands[target_env_ids, :2] = speed.unsqueeze(-1) * self.omni_body_command_dir[target_env_ids]
        if self.cfg.commands.heading_command:
            target_yaw = self.omni_target_yaw[target_env_ids]
            self.commands[target_env_ids, 3] = target_yaw
            self.commands[target_env_ids, 2] = torch.clip(
                0.5 * wrap_to_pi(target_yaw - yaw),
                self.env_command_ranges["ang_vel_yaw"][target_env_ids, 0],
                self.env_command_ranges["ang_vel_yaw"][target_env_ids, 1],
            )
            self.stop_heading[target_env_ids] = False
        else:
            self.commands[target_env_ids, 2] = 0.0

    def _resample_omni_speed_commands(self, env_ids):
        omni_mask = self._omni_env_mask(env_ids)
        if not torch.any(omni_mask):
            return
        omni_env_ids = env_ids[omni_mask]
        speed = torch.clamp(self.commands[omni_env_ids, 0], min=0.0)
        omni_stairs_max_speed = getattr(self.cfg.commands, "omni_stairs_max_speed", None)
        if omni_stairs_max_speed is not None and hasattr(self, "terrain_ids"):
            stairs_mask = self.terrain_ids[omni_env_ids] == 10
            speed = torch.where(
                stairs_mask,
                torch.clamp(speed, max=float(omni_stairs_max_speed)),
                speed,
            )
        self.omni_command_speed[omni_env_ids] = speed
        self._apply_omni_body_commands(omni_env_ids)

    def _randomize_omni_yaw(self, env_ids):
        if len(env_ids) == 0 or not bool(getattr(self.cfg.commands, "omni_random_yaw", True)):
            return
        omni_mask = self._omni_env_mask(env_ids)
        if not torch.any(omni_mask):
            return
        omni_env_ids = env_ids[omni_mask]
        yaw_min, yaw_max = getattr(self.cfg.commands, "omni_yaw_range", [-math.pi, math.pi])
        yaw = torch_rand_float(
            float(yaw_min),
            float(yaw_max),
            (len(omni_env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if (
            bool(getattr(self.cfg.commands, "omni_stairs_discrete_yaw", False))
            and hasattr(self, "terrain_ids")
        ):
            stairs_terrain_id = int(getattr(self.cfg.commands, "omni_stairs_terrain_id", 10))
            stairs_mask = self.terrain_ids[omni_env_ids] == stairs_terrain_id
            if torch.any(stairs_mask):
                yaw_values = torch.tensor(
                    getattr(
                        self.cfg.commands,
                        "omni_stairs_yaw_values",
                        [0.0, math.pi, math.pi / 2, -math.pi / 2],
                    ),
                    device=self.device,
                    dtype=yaw.dtype,
                )
                if yaw_values.numel() > 0:
                    sampled_idx = torch.randint(
                        0,
                        yaw_values.numel(),
                        (int(stairs_mask.sum().item()),),
                        device=self.device,
                    )
                    yaw[stairs_mask] = yaw_values[sampled_idx]
        zeros = torch.zeros_like(yaw)
        self.root_states[omni_env_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        self.omni_target_yaw[omni_env_ids] = yaw
        self.omni_body_command_dir[omni_env_ids, 0] = torch.cos(yaw)
        self.omni_body_command_dir[omni_env_ids, 1] = -torch.sin(yaw)

    def _turn_over_env_mask(self, env_ids=None):
        mask = super()._turn_over_env_mask(env_ids)
        omni_mask = getattr(self, "omni_mask", None)
        if omni_mask is None:
            return mask
        if env_ids is None:
            return mask & ~omni_mask
        return mask & ~omni_mask[env_ids]

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        if not hasattr(self, "total_body_mass"):
            self.total_body_mass = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
            self.base_com_offset = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        if env_id == 0 and self._default_total_mass is None:
            self._default_total_mass = float(sum(p.mass for p in self.default_body_props))
            self._default_base_com = torch.tensor(
                [
                    self.default_body_props[0].com.x,
                    self.default_body_props[0].com.y,
                    self.default_body_props[0].com.z,
                ],
                device=self.device,
                dtype=torch.float,
            )
        self.total_body_mass[env_id, 0] = float(sum(p.mass for p in props))
        current_base_com = torch.tensor(
            [props[0].com.x, props[0].com.y, props[0].com.z],
            device=self.device,
            dtype=torch.float,
        )
        self.base_com_offset[env_id] = current_base_com - self._default_base_com
        return props

    def _get_noise_scale_vec(self, cfg):
        return super()._get_noise_scale_vec(cfg)

    def _post_physics_step_callback(self):
        timing_store = self._active_step_timing
        timed_start = self._timing_start()
        super()._post_physics_step_callback()
        self._apply_omni_body_commands()
        self._timing_stop(timed_start, "parkour_callback_base", timing_store)
        if not self.cfg.camera.enabled and self._camera_source != "proxy":
            return
        force_height_refresh = torch.any(self._force_depth_refresh) or not self._depth_initialized
        if self.cfg.terrain.measure_heights and force_height_refresh:
            periodic_height_update = self.common_step_counter % self.cfg.terrain.measure_heights_interval == 0
            if not periodic_height_update:
                timed_start = self._timing_start()
                if not torch.is_tensor(self.measured_heights):
                    self.measured_heights = self._get_heights()
                else:
                    refresh_ids = self._force_depth_refresh.nonzero(as_tuple=False).flatten()
                    if refresh_ids.numel() > 0:
                        self.measured_heights[refresh_ids] = self._get_heights(refresh_ids)
                self._timing_stop(timed_start, "parkour_forced_height_refresh", timing_store)
        need_update = (
            self.common_step_counter % self.cfg.camera.update_interval == 0
            or torch.any(self._force_depth_refresh)
            or not self._depth_initialized
        )
        if need_update:
            timed_start = self._timing_start()
            self._update_depth_camera_buffer()
            self._timing_stop(timed_start, "parkour_depth_update", timing_store)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self._resample_camera_randomization(env_ids)
        self.max_forward_progress[env_ids] = self._forward_progress()[env_ids]
        self.no_progress_steps[env_ids] = 0
        self.depth_camera_buf[env_ids] = 0.0
        self._force_depth_refresh[env_ids] = True
        self.clearance_air_time[env_ids] = 0.0
        self.clearance_last_contacts[env_ids] = False
        self.edge_last_contacts[env_ids] = False
        self.flat_gait_last_contacts[env_ids] = False
        self.flat_gait_last_contact_filt[env_ids] = False
        self.flat_gait_stance_time[env_ids] = 0.0
        self.flat_gait_contact_ema[env_ids] = float(
            getattr(self.cfg.rewards, "flat_gait_min_contact_duty", 0.2)
        )
        self.targeted_foothold_last_contacts[env_ids] = False
        self.targeted_foothold_air_time[env_ids] = 0.0
        self.targeted_foothold_targets_world[env_ids] = 0.0
        self.targeted_foothold_target_valid[env_ids] = False
        self.stairs_omni_last_contacts[env_ids] = False
        self.stairs_omni_phase_active[env_ids] = False
        self.stairs_omni_phase_timer[env_ids] = 0.0
        self.stairs_omni_phase_anchor_progress[env_ids] = 0.0
        self.stairs_omni_phase_prev_progress[env_ids] = 0.0
        self.stairs_omni_phase_lead_mask[env_ids] = False
        self.stairs_omni_phase_trail_mask[env_ids] = False
        self.stairs_omni_phase_trail_touched[env_ids] = False
        self._stairs_omni_reward_cache_step = -1

    def _terrain_id_mask(self, terrain_ids):
        if terrain_ids.numel() == 0 or not hasattr(self, "terrain_ids"):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return (self.terrain_ids.unsqueeze(1) == terrain_ids.unsqueeze(0)).any(dim=1)

    def _flat_speed_search_mask(self, env_ids=None):
        if (
            not self.flat_speed_search_enabled
            or self.flat_speed_search_terrain_ids.numel() == 0
            or not hasattr(self, "terrain_ids")
        ):
            if env_ids is None:
                return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        terrain_ids = self.terrain_ids if env_ids is None else self.terrain_ids[env_ids]
        return (terrain_ids.unsqueeze(1) == self.flat_speed_search_terrain_ids.unsqueeze(0)).any(dim=1)

    def _heading_conditioned_mask(self):
        active_cmd = self.commands[:, 0] > self.cfg.rewards.heading_conditioned_cmd_threshold
        return self._terrain_id_mask(self.heading_conditioned_terrain_ids) & active_cmd & ~self._omni_env_mask()

    def _fall_terrain_mask(self):
        if self.fall_terrain_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if not hasattr(self, "terrain_ids"):
            return torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        return self._terrain_id_mask(self.fall_terrain_ids)

    def _fall_recovery_mask(self):
        if not getattr(self.cfg.init_state, "turn_over", False):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return self._turn_over_env_mask() & ~self._turn_over_success_mask()

    def _targeted_foothold_mask(self):
        active_cmd = self._forward_reward_command_x() > self.cfg.rewards.targeted_foothold_cmd_threshold
        return self._terrain_id_mask(self.targeted_foothold_terrain_ids) & active_cmd & ~self._omni_env_mask()

    def _flat_gait_mask(self):
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active_cmd = cmd_speed > float(getattr(self.cfg.rewards, "flat_gait_cmd_threshold", 0.2))
        return self._terrain_id_mask(self.flat_gait_terrain_ids) & active_cmd

    def _stairs_omni_mask(self):
        stairs_terrain_id = int(getattr(self.cfg.rewards, "stairs_omni_terrain_id", 10))
        active_cmd = self._forward_reward_command_x() > float(
            getattr(self.cfg.rewards, "stairs_omni_cmd_threshold", 0.2)
        )
        if not hasattr(self, "terrain_ids"):
            return self._omni_env_mask() & active_cmd
        return (self.terrain_ids == stairs_terrain_id) & self._omni_env_mask() & active_cmd

    def _clear_stairs_omni_phase(self, env_mask):
        if env_mask.dtype != torch.bool:
            raise TypeError("_clear_stairs_omni_phase expects a boolean mask")
        if not torch.any(env_mask):
            return
        self.stairs_omni_phase_active[env_mask] = False
        self.stairs_omni_phase_timer[env_mask] = 0.0
        self.stairs_omni_phase_anchor_progress[env_mask] = 0.0
        self.stairs_omni_phase_prev_progress[env_mask] = 0.0
        self.stairs_omni_phase_lead_mask[env_mask] = False
        self.stairs_omni_phase_trail_mask[env_mask] = False
        self.stairs_omni_phase_trail_touched[env_mask] = False

    def _stairs_omni_progress_direction(self):
        direction = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        direction[:, 0] = 1.0
        return direction

    def _compute_stairs_omni_dynamic_pairs(self, feet_pos):
        num_feet = feet_pos.shape[1]
        progress_dir = self._stairs_omni_progress_direction()
        progress_coord = torch.sum(feet_pos[:, :, :2] * progress_dir.unsqueeze(1), dim=-1)
        pair_count = min(2, num_feet)
        lead_idx = torch.topk(progress_coord, k=pair_count, dim=1).indices
        trail_idx = torch.topk(-progress_coord, k=pair_count, dim=1).indices
        lead_mask = torch.zeros(
            self.num_envs,
            num_feet,
            device=self.device,
            dtype=torch.bool,
        )
        trail_mask = torch.zeros_like(lead_mask)
        lead_mask.scatter_(1, lead_idx, True)
        trail_mask.scatter_(1, trail_idx, True)
        return lead_mask, trail_mask

    def _stairs_omni_surface_gain(self, feet_pos):
        probe_distance = float(getattr(self.cfg.rewards, "stairs_omni_height_probe_distance", 0.08))
        progress_dir = self._stairs_omni_progress_direction()
        feet_xy = feet_pos[:, :, :2]
        current_ground = self._sample_heightfield(feet_xy)
        behind_ground = self._sample_heightfield(
            feet_xy - progress_dir.unsqueeze(1) * probe_distance
        )
        return current_ground - behind_ground

    def _update_stairs_omni_reward_cache(self):
        if getattr(self, "_stairs_omni_reward_cache_step", -1) == self.common_step_counter:
            return

        zero = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        cache = {
            "stairs_omni_lead_touch_above": zero.clone(),
            "stairs_omni_trail_followup": zero.clone(),
            "stairs_omni_support_transfer_progress": zero.clone(),
        }

        active_envs = self._stairs_omni_mask()
        inactive_phase_envs = self.stairs_omni_phase_active & ~active_envs
        self._clear_stairs_omni_phase(inactive_phase_envs)

        num_feet = self.feet_indices.shape[0]
        if num_feet < 2 or not torch.any(active_envs):
            self.stairs_omni_last_contacts[:] = self.contact_forces[:, self.feet_indices, 2] > 1.0
            self._stairs_omni_reward_cache = cache
            self._stairs_omni_reward_cache_step = self.common_step_counter
            return

        feet_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        feet_pos = feet_states[:, :, 0:3]
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        prev_contact = self.stairs_omni_last_contacts
        first_contact = contact & ~prev_contact

        lead_mask_dyn, trail_mask_dyn = self._compute_stairs_omni_dynamic_pairs(feet_pos)
        upper_gain = self._stairs_omni_surface_gain(feet_pos)
        touch_height = float(getattr(self.cfg.rewards, "stairs_omni_touch_height", 0.03))
        contact_above = contact & (upper_gain > touch_height)

        lead_new_touches = lead_mask_dyn & first_contact & contact_above
        start_envs = active_envs & ~self.stairs_omni_phase_active & lead_new_touches.any(dim=1)
        if torch.any(start_envs):
            self.stairs_omni_phase_active[start_envs] = True
            self.stairs_omni_phase_timer[start_envs] = 0.0
            self.stairs_omni_phase_anchor_progress[start_envs] = self.root_states[start_envs, 0]
            self.stairs_omni_phase_prev_progress[start_envs] = 0.0
            self.stairs_omni_phase_lead_mask[start_envs] = lead_mask_dyn[start_envs]
            self.stairs_omni_phase_trail_mask[start_envs] = trail_mask_dyn[start_envs]
            self.stairs_omni_phase_trail_touched[start_envs] = False

        lead_touch_reward = (
            lead_new_touches.float().sum(dim=1)
            / lead_mask_dyn.float().sum(dim=1).clamp_min(1.0)
        )
        cache["stairs_omni_lead_touch_above"] = lead_touch_reward * start_envs.float()

        followup_window = float(getattr(self.cfg.rewards, "stairs_omni_followup_window", 0.35))
        phase_active = self.stairs_omni_phase_active & active_envs
        within_window = self.stairs_omni_phase_timer <= followup_window
        trail_contact_above = self.stairs_omni_phase_trail_mask & contact_above
        new_trail_touches = trail_contact_above & ~self.stairs_omni_phase_trail_touched
        new_trail_touches &= phase_active.unsqueeze(-1) & within_window.unsqueeze(-1)
        trail_followup_reward = (
            new_trail_touches.float().sum(dim=1)
            / self.stairs_omni_phase_trail_mask.float().sum(dim=1).clamp_min(1.0)
        )
        cache["stairs_omni_trail_followup"] = trail_followup_reward
        self.stairs_omni_phase_trail_touched |= new_trail_touches

        phase_progress = (self.root_states[:, 0] - self.stairs_omni_phase_anchor_progress).clamp(min=0.0)
        progress_delta = (phase_progress - self.stairs_omni_phase_prev_progress).clamp(min=0.0)
        progress_step = float(getattr(self.cfg.rewards, "stairs_omni_support_progress_step", 0.04))
        progress_delta = torch.clamp(progress_delta, max=progress_step)
        support_transfer_reward = torch.where(
            phase_active,
            progress_delta / max(progress_step, 1.0e-6),
            torch.zeros_like(progress_delta),
        )
        cache["stairs_omni_support_transfer_progress"] = support_transfer_reward
        self.stairs_omni_phase_prev_progress = torch.where(
            phase_active,
            phase_progress,
            self.stairs_omni_phase_prev_progress,
        )

        self.stairs_omni_phase_timer = torch.where(
            phase_active,
            self.stairs_omni_phase_timer + self.dt,
            self.stairs_omni_phase_timer,
        )
        trail_touched_count = (
            self.stairs_omni_phase_trail_touched & self.stairs_omni_phase_trail_mask
        ).sum(dim=1)
        trail_target_count = self.stairs_omni_phase_trail_mask.sum(dim=1)
        done_envs = phase_active & (
            (trail_touched_count >= trail_target_count) | (self.stairs_omni_phase_timer > followup_window)
        )
        self._clear_stairs_omni_phase(done_envs)

        self.stairs_omni_last_contacts = contact
        self._stairs_omni_reward_cache = cache
        self._stairs_omni_reward_cache_step = self.common_step_counter

    def _forward_heading_alignment(self):
        return torch.clamp(torch.cos(wrap_to_pi(self.rpy[:, 2])), min=0.0)

    def _forward_progress(self):
        progress_origins = getattr(self, "progress_origins", self.env_origins)
        return self.root_states[:, 0] - progress_origins[:, 0]

    def _forward_reward_command_x(self):
        if not hasattr(self, "omni_command_speed"):
            return self.commands[:, 0]
        return torch.where(
            self._omni_env_mask(),
            self.omni_command_speed,
            self.commands[:, 0],
        )

    def _forward_reward_velocity_x(self):
        if not hasattr(self, "omni_command_speed"):
            return self.base_lin_vel[:, 0]
        return torch.where(
            self._omni_env_mask(),
            self.root_states[:, 7],
            self.base_lin_vel[:, 0],
        )

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.init_state.turn_over:
            self.turn_over_timer[env_ids] = 0.0
            if hasattr(self, "turn_over_pending"):
                self.turn_over_pending[env_ids] = False

        def apply_parkour_reset_position(target_env_ids, keep_current_height=False):
            reset_origins = getattr(self, "reset_origins", self.env_origins)
            self.root_states[target_env_ids, 0] = self.base_init_state[0] + reset_origins[target_env_ids, 0]
            self.root_states[target_env_ids, 1] = self.base_init_state[1] + reset_origins[target_env_ids, 1]
            if keep_current_height:
                z_offset = self.root_states[target_env_ids, 2] - self.env_origins[target_env_ids, 2]
                self.root_states[target_env_ids, 2] = reset_origins[target_env_ids, 2] + z_offset
            else:
                self.root_states[target_env_ids, 2] = self.base_init_state[2] + reset_origins[target_env_ids, 2]

            x_jitter = float(getattr(self.cfg.terrain, "reset_platform_x_jitter", 0.15))
            y_jitter = float(getattr(self.cfg.terrain, "reset_platform_y_jitter", 0.30))
            if x_jitter > 0.0:
                x_offset = torch_rand_float(-x_jitter, x_jitter, (len(target_env_ids), 1), device=self.device).squeeze(1)
                self.root_states[target_env_ids, 0] += x_offset
            if y_jitter > 0.0:
                y_offset = torch_rand_float(-y_jitter, y_jitter, (len(target_env_ids), 1), device=self.device).squeeze(1)
                self.root_states[target_env_ids, 1] += y_offset

        turn_over_mask = self._turn_over_env_mask(env_ids)
        turn_over_env_ids = env_ids[turn_over_mask]
        regular_env_ids = env_ids[~turn_over_mask]

        if len(turn_over_env_ids) > 0:
            super()._reset_root_states(turn_over_env_ids)
            apply_parkour_reset_position(turn_over_env_ids, keep_current_height=True)

        if len(regular_env_ids) > 0:
            base_init_state = self.base_init_state.reshape(1, -1).repeat(len(regular_env_ids), 1)
            zero_angles = torch.zeros(len(regular_env_ids), device=self.device)
            base_init_state[:, 3:7] = quat_from_euler_xyz(zero_angles, zero_angles, zero_angles)

            self.root_states[regular_env_ids] = base_init_state
            apply_parkour_reset_position(regular_env_ids)
            self._randomize_omni_yaw(regular_env_ids)

            if getattr(self.cfg.init_state, "zero_reset_root_velocities", False):
                self.root_states[regular_env_ids, 7:13] = 0.0
            else:
                # Keep the velocity randomization used by the base task.
                self.root_states[regular_env_ids, 7:13] = torch_rand_float(
                    -0.5, 0.5, (len(regular_env_ids), 6), device=self.device
                )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
    
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done or self.cfg.terrain.mesh_type == 'plane':
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.reset_origins[env_ids, :2], dim=1)
        #print(torch.norm(self.root_states[env_ids, :2] - self.reset_origins[env_ids, :2], dim=1))
        #print(torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1))
        # reset_origins是刷新点，env_origins是中心点
        # distance = self.max_move_distance[env_ids]
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        if self.cfg.terrain.move_down_by_accumulated_xy_command:
            move_down = (distance < torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * (self.cfg.commands.resampling_time * (1 - self.zero_command_proba)) * 0.5) * ~move_up
        else:
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        if hasattr(self, "terrain_spawn_origins"):
            self.reset_origins[env_ids] = self.terrain_spawn_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            self.progress_origins[env_ids] = self.reset_origins[env_ids]
        else:
            self.reset_origins[env_ids] = self.env_origins[env_ids]
            self.progress_origins[env_ids] = self.env_origins[env_ids]
        if hasattr(self, "terrain_id_map"):
            self.terrain_ids[env_ids] = self.terrain_id_map[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            self._update_env_command_ranges()
        self.max_move_distance[env_ids] = 0.0

    def check_termination(self):
        super().check_termination()
        termination_cfg = getattr(self.cfg, "termination", None)
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        fall_base_height_cutoff = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        fall_foot_height_cutoff = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if getattr(termination_cfg, "use_fall_height", True):
            fall_mask = self._fall_terrain_mask()
            fall_base_height = float(getattr(termination_cfg, "fall_base_height", -0.1))
            default_fall_foot_height = float(getattr(termination_cfg, "fall_foot_height", -0.1))
            fall_foot_height = torch.full(
                (self.num_envs,),
                default_fall_foot_height,
                device=self.device,
                dtype=feet_pos.dtype,
            )
            fall_foot_height_overrides = getattr(termination_cfg, "fall_foot_height_overrides", {})
            if fall_foot_height_overrides and hasattr(self, "terrain_ids"):
                for terrain_id, terrain_fall_foot_height in fall_foot_height_overrides.items():
                    terrain_mask = self.terrain_ids == int(terrain_id)
                    if torch.any(terrain_mask):
                        fall_foot_height[terrain_mask] = float(terrain_fall_foot_height)
            fall_base_height_cutoff = fall_mask & (self.root_states[:, 2] < fall_base_height)
            fall_foot_height_cutoff = fall_mask & (
                torch.min(feet_pos[:, :, 2], dim=1).values < fall_foot_height
            )
        self._record_termination_reason("fall_base_height_cutoff", fall_base_height_cutoff)
        self._record_termination_reason("fall_foot_height_cutoff", fall_foot_height_cutoff)
        self.reset_buf |= fall_base_height_cutoff | fall_foot_height_cutoff

        no_progress = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        no_progress_timeout_s = float(getattr(self.cfg.rewards, "no_progress_timeout_s", 2.0))
        if no_progress_timeout_s > 0.0:
            progress_command = self.commands[:, 0]
            if hasattr(self, "omni_command_speed"):
                progress_command = torch.where(
                    self._omni_env_mask(),
                    self.omni_command_speed,
                    progress_command,
                )
            monitor_mask = progress_command > float(getattr(self.cfg.rewards, "no_progress_cmd_threshold", 0.2))
            if hasattr(self, "terrain_levels"):
                monitor_mask &= self.terrain_levels >= int(
                    getattr(self.cfg.rewards, "no_progress_min_terrain_level", 3)
                )
            if self.no_progress_excluded_terrain_ids.numel() > 0:
                monitor_mask &= ~self._terrain_id_mask(self.no_progress_excluded_terrain_ids)
            forward_progress = self._forward_progress()
            improved = forward_progress > (
                self.max_forward_progress + float(getattr(self.cfg.rewards, "no_progress_min_delta", 0.1))
            )
            self.max_forward_progress = torch.where(improved, forward_progress, self.max_forward_progress)
            self.no_progress_steps = torch.where(
                monitor_mask,
                torch.where(improved, torch.zeros_like(self.no_progress_steps), self.no_progress_steps + 1),
                torch.zeros_like(self.no_progress_steps),
            )
            timeout_steps = max(1, int(np.ceil(no_progress_timeout_s / self.dt)))
            timeout_steps_per_env = torch.full_like(self.no_progress_steps, timeout_steps)
            if hasattr(self, "omni_mask"):
                omni_timeout_s = float(
                    getattr(self.cfg.rewards, "omni_no_progress_timeout_s", no_progress_timeout_s)
                )
                omni_timeout_steps = max(1, int(np.ceil(omni_timeout_s / self.dt)))
                timeout_steps_per_env = torch.where(
                    self._omni_env_mask(),
                    torch.full_like(timeout_steps_per_env, omni_timeout_steps),
                    timeout_steps_per_env,
                )
            no_progress = monitor_mask & (self.no_progress_steps >= timeout_steps_per_env)
        self._record_termination_reason("no_progress", no_progress)
        self.reset_buf |= no_progress

        out_of_block = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not getattr(self.cfg.terrain, "reset_when_outside_block", False):
            self._record_termination_reason("out_of_block", out_of_block)
            return
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"] or not getattr(self, "custom_origins", False):
            self._record_termination_reason("out_of_block", out_of_block)
            return

        margin = getattr(self.cfg.terrain, "reset_outside_block_margin", 0.0)
        half_length = 0.5 * self.cfg.terrain.terrain_length + margin
        half_width = 0.5 * self.cfg.terrain.terrain_width + margin
        relative_pos = self.root_states[:, :2] - self.env_origins[:, :2]
        out_of_block = (torch.abs(relative_pos[:, 0]) > half_length) | (torch.abs(relative_pos[:, 1]) > half_width)
        self._record_termination_reason("out_of_block", out_of_block)
        self.reset_buf |= out_of_block

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.stop_heading[env_ids] = False
        if len(self.cfg.commands.command_range_curriculum):
            current_iter = self.common_step_counter // self.num_steps_per_env
            for i in range(len(self.cfg.commands.command_range_curriculum) - 1, -1, -1):
                cfg = self.cfg.commands.command_range_curriculum[i]
                if current_iter >= cfg["iter"]:
                    self.command_ranges["lin_vel_x"] = cfg["lin_vel_x"]
                    self.command_ranges["lin_vel_y"] = cfg["lin_vel_y"]
                    self.command_ranges["ang_vel_yaw"] = cfg["ang_vel_yaw"]
                    self.command_ranges["heading"] = cfg["heading"]
                    self.max_lin_vel = max(
                        abs(self.command_ranges["lin_vel_x"][0]),
                        abs(self.command_ranges["lin_vel_x"][1]),
                        abs(self.command_ranges["lin_vel_y"][0]),
                        abs(self.command_ranges["lin_vel_y"][1]),
                    )
                    self.cfg.commands.command_range_curriculum.pop(i)
                    self._update_env_command_ranges()
                    print(f"Command range updated at iter {current_iter}: {self.command_ranges}")

        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        self.commands[env_ids, :4] = 0.0

        easy_terrain_ids = torch.tensor(
            getattr(self.cfg.commands, "easy_command_terrain_ids", [0, 1, 2]),
            device=self.device,
            dtype=torch.long,
        )
        if hasattr(self, "terrain_ids") and easy_terrain_ids.numel() > 0:
            easy_mask = (self.terrain_ids[env_ids].unsqueeze(1) == easy_terrain_ids.unsqueeze(0)).any(dim=1)
        else:
            easy_mask = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        easy_ids = env_ids[easy_mask]
        hard_ids = env_ids[~easy_mask]

        self._resample_command_group(
            easy_ids,
            lin_x_key="lin_vel_x",
            lin_y_key="lin_vel_y",
            yaw_key="ang_vel_yaw",
            heading_key="heading",
        )
        self._resample_command_group(
            hard_ids,
            lin_x_key="new_lin_vel_x",
            lin_y_key="new_lin_vel_y",
            yaw_key="new_ang_vel_yaw",
            heading_key="new_heading",
        )

        if getattr(self.cfg.commands, "zero_command", False) and env_ids.numel() > 0:
            zero_threshold = float(getattr(self.cfg.commands, "zero_command_threshold", 0.2))
            self.commands[env_ids, :2] *= (
                torch.norm(self.commands[env_ids, :2], dim=1) > zero_threshold
            ).unsqueeze(1)

        self._resample_omni_speed_commands(env_ids)

        speed_search_mask = self._flat_speed_search_mask(env_ids)
        if torch.any(speed_search_mask):
            speed_search_env_ids = env_ids[speed_search_mask]
            self.commands[speed_search_env_ids, :4] = 0.0
            self.stop_heading[speed_search_env_ids] = True
            if hasattr(self, "omni_command_speed"):
                self.omni_command_speed[speed_search_env_ids] = 0.0

        if self.cfg.init_state.turn_over:
            zero_mask = self._turn_over_env_mask(env_ids) & (self.turn_over_timer[env_ids] > 0)
        else:
            zero_mask = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        if zero_mask.any():
            zero_env_ids = env_ids[zero_mask]
            self.commands[zero_env_ids, :3] = 0.0
            self.stop_heading[zero_env_ids] = True

        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

    def _resample_command_group(self, env_ids, lin_x_key, lin_y_key, yaw_key, heading_key):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges[lin_x_key][0],
            self.command_ranges[lin_x_key][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges[lin_y_key][0],
            self.command_ranges[lin_y_key][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges[heading_key][0],
                self.command_ranges[heading_key][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[env_ids, 2] = 0.0
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges[yaw_key][0],
                self.command_ranges[yaw_key][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _get_feet_heights(self):
        base_height = self._get_base_height()
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        base_pos = self.root_states[:, 0:3].unsqueeze(1)
        delta_feet = feet_pos - base_pos
        feet_to_base_height = (delta_feet * self.projected_gravity.unsqueeze(1)).sum(-1)
        return torch.clamp(base_height.unsqueeze(1) - feet_to_base_height, min=0.0)

    def _reward_low_speed_when_commanded(self):
        cmd_xy = self.commands[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        active_cmd = cmd_speed > self.cfg.rewards.low_speed_cmd_threshold

        actual_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        min_speed = torch.minimum(
            cmd_speed,
            torch.full_like(cmd_speed, self.cfg.rewards.low_speed_min_xy_speed),
        )
        return (min_speed - actual_speed).clamp(min=0.0) * active_cmd

    def _base_height_penalty_mask(self):
        return ~self._terrain_id_mask(self.base_height_excluded_terrain_ids)

    def _reward_base_height(self):
        rew = super()._reward_base_height()
        return rew * self._base_height_penalty_mask().float()

    def _reward_correct_base_height(self):
        rew = super()._reward_correct_base_height()
        return rew * self._base_height_penalty_mask().float()

    def _reward_hip_to_default(self):
        rew = super()._reward_hip_to_default()
        valid_mask = ~self._terrain_id_mask(self.hip_to_default_excluded_terrain_ids)
        return rew * valid_mask.float()

    def _reward_target_posture(self):
        rew = super()._reward_target_posture()
        valid_mask = self._terrain_id_mask(self.target_posture_terrain_ids)
        return rew * valid_mask.float()

    def _reward_tracking_lin_vel(self):
        reward = super()._reward_tracking_lin_vel()
        speed_search_mask = self._flat_speed_search_mask()
        if torch.any(speed_search_mask):
            reward[speed_search_mask] = 0.0
        conditioned_mask = self._heading_conditioned_mask()
        if torch.any(conditioned_mask):
            heading_align = self._forward_heading_alignment()
            reward[conditioned_mask] = reward[conditioned_mask] * heading_align[conditioned_mask]

        full_reward_mask = self._terrain_id_mask(self.tracking_lin_vel_full_reward_terrain_ids)
        active_cmd = self.commands[:, 0] > 1.0e-6
        full_reward_mask &= active_cmd & (self.base_lin_vel[:, 0] >= self.commands[:, 0])
        reward[full_reward_mask] = 1.0
        return reward

    def _reward_stand_still(self):
        reward = super()._reward_stand_still()
        if self.stand_still_terrain_ids.numel() > 0:
            reward = reward * self._terrain_id_mask(self.stand_still_terrain_ids).float()
        speed_search_mask = self._flat_speed_search_mask()
        if torch.any(speed_search_mask):
            reward[speed_search_mask] = 0.0
        return reward

    def _reward_flat_forward_speed(self):
        speed_search_mask = self._flat_speed_search_mask()
        if not torch.any(speed_search_mask):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        speed = self._forward_reward_velocity_x().clamp(min=0.0)
        max_reward_speed = float(getattr(self.cfg.rewards, "flat_speed_search_max_speed_reward", 2.5))
        if max_reward_speed > 0.0:
            speed = torch.clamp(speed, max=max_reward_speed)
        if self.cfg.init_state.turn_over:
            speed_search_mask &= ~self._fall_recovery_mask()
        return speed * speed_search_mask.float()

    def _reward_hard_terrain_yaw_penalty(self):
        conditioned_mask = self._heading_conditioned_mask()
        return torch.square(self.base_ang_vel[:, 2]) * conditioned_mask.float()

    def _reward_omni_target_yaw_error(self):
        omni_mask = self._omni_env_mask()
        if not torch.any(omni_mask):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        yaw_error = torch.abs(wrap_to_pi(self.omni_target_yaw - self._root_yaw()))
        deadzone = float(getattr(self.cfg.rewards, "omni_target_yaw_deadzone", 0.0))
        return torch.square((yaw_error - deadzone).clamp(min=0.0)) * omni_mask.float()

    def _reward_stairs_omni_lead_touch_above(self):
        self._update_stairs_omni_reward_cache()
        return self._stairs_omni_reward_cache["stairs_omni_lead_touch_above"]

    def _reward_stairs_omni_trail_followup(self):
        self._update_stairs_omni_reward_cache()
        return self._stairs_omni_reward_cache["stairs_omni_trail_followup"]

    def _reward_stairs_omni_support_transfer_progress(self):
        self._update_stairs_omni_reward_cache()
        return self._stairs_omni_reward_cache["stairs_omni_support_transfer_progress"]

    def _reward_single_bridge_narrow_stance(self):
        if not hasattr(self, "terrain_ids"):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        bridge_id = int(getattr(self.cfg.rewards, "single_bridge_terrain_id", 4))
        active_cmd = self._forward_reward_command_x() > float(
            getattr(self.cfg.rewards, "single_bridge_narrow_stance_cmd_threshold", 0.2)
        )
        active_envs = (self.terrain_ids == bridge_id) & active_cmd
        if not torch.any(active_envs):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        num_feet = self.feet_indices.shape[0]
        feet_pos_world = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[
            :, self.feet_indices, 0:3
        ]
        feet_pos_relative = feet_pos_world - self.root_states[:, 0:3].unsqueeze(1)
        local_feet_pos = quat_rotate_inverse(
            self.base_quat.repeat_interleave(num_feet, dim=0),
            feet_pos_relative.reshape(-1, 3),
        ).reshape(self.num_envs, num_feet, 3)

        lateral_target = float(getattr(self.cfg.rewards, "single_bridge_narrow_stance_lateral_target", 0.06))
        lateral_tolerance = float(
            getattr(self.cfg.rewards, "single_bridge_narrow_stance_lateral_tolerance", 0.01)
        )
        sigma = float(getattr(self.cfg.rewards, "single_bridge_narrow_stance_sigma", 0.0025))
        foot_signs = torch.tensor(
            [1.0, -1.0, 1.0, -1.0],
            device=self.device,
            dtype=local_feet_pos.dtype,
        ).unsqueeze(0)[:, :num_feet]
        target_y = foot_signs * lateral_target
        y_error = (torch.abs(local_feet_pos[:, :, 1] - target_y) - lateral_tolerance).clamp(min=0.0)
        reward = torch.exp(-torch.square(y_error) / max(sigma, 1.0e-6)).mean(dim=1)
        return reward * active_envs.float()

    def _reward_flat_gait(self):
        active_envs = self._flat_gait_mask()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        prev_contact = self.flat_gait_last_contacts
        prev_contact_filt = self.flat_gait_last_contact_filt
        contact_filt = torch.logical_or(contact, prev_contact)
        active_feet = active_envs.unsqueeze(-1)

        min_stance_time = float(getattr(self.cfg.rewards, "flat_gait_min_stance_time", 0.10))
        liftoff = prev_contact_filt & ~contact_filt
        early_liftoff = liftoff & (self.flat_gait_stance_time < min_stance_time) & active_feet

        next_stance_time = torch.where(
            contact_filt,
            self.flat_gait_stance_time + self.dt,
            torch.zeros_like(self.flat_gait_stance_time),
        )
        self.flat_gait_stance_time = torch.where(
            active_feet,
            next_stance_time,
            torch.zeros_like(self.flat_gait_stance_time),
        )
        self.flat_gait_last_contacts = torch.where(
            active_feet,
            contact,
            torch.zeros_like(contact),
        )
        self.flat_gait_last_contact_filt = torch.where(
            active_feet,
            contact_filt,
            torch.zeros_like(contact_filt),
        )

        ema_alpha = float(getattr(self.cfg.rewards, "flat_gait_contact_ema_alpha", 0.98))
        ema_alpha = min(max(ema_alpha, 0.0), 1.0)
        neutral_duty = float(getattr(self.cfg.rewards, "flat_gait_min_contact_duty", 0.2))
        neutral_ema = torch.full_like(self.flat_gait_contact_ema, neutral_duty)
        contact_ema = ema_alpha * self.flat_gait_contact_ema + (1.0 - ema_alpha) * contact_filt.float()
        self.flat_gait_contact_ema = torch.where(active_feet, contact_ema, neutral_ema)

        min_contact_duty = float(getattr(self.cfg.rewards, "flat_gait_min_contact_duty", 0.20))
        low_contact_duty = (min_contact_duty - self.flat_gait_contact_ema).clamp(min=0.0).sum(dim=1)

        num_feet = self.feet_indices.shape[0]
        if num_feet >= 4:
            rear_contact_duty = self.flat_gait_contact_ema[:, 2:4]
        else:
            rear_contact_duty = self.flat_gait_contact_ema[:, max(0, num_feet // 2):]

        min_rear_contact_duty = float(getattr(self.cfg.rewards, "flat_gait_min_rear_contact_duty", 0.30))
        low_rear_duty = (min_rear_contact_duty - rear_contact_duty).clamp(min=0.0).sum(dim=1)
        if rear_contact_duty.shape[1] >= 2:
            rear_balance_tolerance = float(getattr(self.cfg.rewards, "flat_gait_rear_balance_tolerance", 0.20))
            rear_balance = (
                torch.abs(rear_contact_duty[:, 0] - rear_contact_duty[:, 1]) - rear_balance_tolerance
            ).clamp(min=0.0)
        else:
            rear_balance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        reward = (
            float(getattr(self.cfg.rewards, "flat_gait_early_liftoff_weight", 1.0))
            * early_liftoff.float().sum(dim=1)
            + float(getattr(self.cfg.rewards, "flat_gait_contact_duty_weight", 0.5))
            * low_contact_duty
            + float(getattr(self.cfg.rewards, "flat_gait_rear_duty_weight", 1.0))
            * low_rear_duty
            + float(getattr(self.cfg.rewards, "flat_gait_rear_balance_weight", 0.5))
            * rear_balance
        )
        return reward * active_envs.float()

    def _reward_feet_stumble(self):
        return self._reward_stumble()

    def _reward_feet_edge(self):
        if getattr(self, "x_edge_mask", None) is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        #gap_mask = self._gap_terrain_mask()
        # if not torch.any(gap_mask):
        #     return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.edge_last_contacts)
        self.edge_last_contacts = contact

        feet_pos_xy = (
            (
                self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, :2]
                + self.terrain.cfg.border_size
            )
            / self.cfg.terrain.horizontal_scale
        ).round().long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        feet_at_edge = contact_filt & feet_at_edge

        rew = torch.sum(feet_at_edge, dim=-1).float()
        #rew[~gap_mask] = 0.0
        return rew

    def _reward_upward_foothold_clearance(self):
        feet_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        feet_pos = feet_states[:, :, 0:3]
        feet_xy_speed = torch.norm(feet_states[:, :, 7:9], dim=-1)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.clearance_last_contacts)
        self.clearance_last_contacts = contact
        in_air = ~contact_filt
        self.clearance_air_time = torch.where(
            in_air,
            self.clearance_air_time + self.dt,
            torch.zeros_like(self.clearance_air_time),
        )

        current_ground = self._sample_heightfield(feet_pos[:, :, :2])
        foot_clearance = (feet_pos[:, :, 2] - current_ground).clamp(min=0.0)

        base_forward = quat_apply(
            self.base_quat,
            torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1),
        )
        forward_xy = base_forward[:, :2]
        forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        omni_mask = self._omni_env_mask()
        if torch.any(omni_mask):
            world_x_forward = torch.zeros_like(forward_xy)
            world_x_forward[:, 0] = 1.0
            forward_xy = torch.where(omni_mask.unsqueeze(-1), world_x_forward, forward_xy)

        lookahead_distance = torch.full(
            (self.num_envs,),
            float(self.cfg.rewards.clearance_lookahead_distance),
            device=self.device,
            dtype=feet_pos.dtype,
        )
        omni_stairs_lookahead = float(
            getattr(
                self.cfg.rewards,
                "omni_stairs_clearance_lookahead_distance",
                self.cfg.rewards.clearance_lookahead_distance,
            )
        )
        if hasattr(self, "terrain_ids") and omni_stairs_lookahead != float(self.cfg.rewards.clearance_lookahead_distance):
            omni_stairs_mask = self._omni_env_mask() & (self.terrain_ids == 10)
            lookahead_distance = torch.where(
                omni_stairs_mask,
                torch.full_like(lookahead_distance, omni_stairs_lookahead),
                lookahead_distance,
            )

        lookahead_xy = feet_pos[:, :, :2] + forward_xy.unsqueeze(1) * lookahead_distance.view(-1, 1, 1)
        ahead_ground = self._sample_heightfield(lookahead_xy)
        terrain_rise = (ahead_ground - current_ground).clamp(min=0.0)
        #print(terrain_rise)
        cmd_x = self._forward_reward_command_x()
        active_cmd = (cmd_x > self.cfg.rewards.clearance_min_command_x).unsqueeze(-1)
        min_base_vel = torch.clamp(
            cmd_x * self.cfg.rewards.clearance_min_base_vel_ratio,
            min=self.cfg.rewards.clearance_min_base_vel,
        ).unsqueeze(-1)
        progressing = self._forward_reward_velocity_x().unsqueeze(-1) > min_base_vel
        min_foot_speed = torch.full_like(
            feet_xy_speed,
            float(self.cfg.rewards.clearance_min_foot_speed),
        )
        omni_stairs_mask = self._omni_env_mask() & (self.terrain_ids == 10) if hasattr(self, "terrain_ids") else self._omni_env_mask()
        if torch.any(omni_stairs_mask):
            min_foot_speed = torch.where(
                omni_stairs_mask.unsqueeze(-1),
                torch.full_like(
                    min_foot_speed,
                    float(getattr(self.cfg.rewards, "omni_stairs_clearance_min_foot_speed", self.cfg.rewards.clearance_min_foot_speed)),
                ),
                min_foot_speed,
            )
            if not bool(getattr(self.cfg.rewards, "omni_stairs_clearance_requires_progress", True)):
                progressing = torch.where(
                    omni_stairs_mask.unsqueeze(-1),
                    torch.ones_like(progressing),
                    progressing,
                )
        moving_feet = feet_xy_speed > min_foot_speed
        clearance_enabled = ~self._terrain_id_mask(self.clearance_excluded_terrain_ids).unsqueeze(-1)
        active_feet = active_cmd & progressing & moving_feet & in_air & clearance_enabled & (
            terrain_rise > self.cfg.rewards.clearance_active_step_height
        )
        desired_clearance = (
            self.cfg.rewards.clearance_base_height
            + self.cfg.rewards.clearance_rise_gain * terrain_rise
        ).clamp(max=self.cfg.rewards.clearance_max_height)
        clearance_error = foot_clearance - desired_clearance
        foot_reward = torch.exp(
            -torch.square(clearance_error) / self.cfg.rewards.clearance_tracking_sigma
        )
        overtime_ratio = torch.clamp(
            (self.clearance_air_time - self.cfg.rewards.clearance_max_air_time)
            / self.cfg.rewards.clearance_max_air_time,
            min=0.0,
            max=1.0,
        )
        foot_reward = foot_reward - self.cfg.rewards.clearance_overtime_penalty * overtime_ratio
        foot_reward = foot_reward * active_feet.float()
        #print(foot_reward[0])
        return foot_reward.sum(dim=1)

    def _compute_targeted_foothold_candidates(self):
        candidate_targets_world = torch.zeros_like(self.targeted_foothold_targets_world)
        candidate_valid = torch.zeros_like(self.targeted_foothold_target_valid)
        if not torch.is_tensor(self.measured_heights):
            return candidate_targets_world, candidate_valid

        active_envs = self._targeted_foothold_mask()
        if not torch.any(active_envs):
            return candidate_targets_world, candidate_valid

        num_feet = self.feet_indices.shape[0]
        local_scan_points = self.height_points[:, :, :2]
        world_scan_points = quat_apply_yaw(
            self.base_quat.repeat(1, self.num_height_points),
            self.height_points,
        ) + self.root_states[:, :3].unsqueeze(1)
        world_scan_points[:, :, 2] = self.measured_heights
        omni_mask = self._omni_env_mask()
        if torch.any(omni_mask):
            world_x_scan_points = world_scan_points[:, :, :2] - self.root_states[:, :2].unsqueeze(1)
            local_scan_points = torch.where(
                omni_mask.view(self.num_envs, 1, 1),
                world_x_scan_points,
                local_scan_points,
            )

        scan_edge = torch.zeros(self.num_envs, self.num_height_points, device=self.device, dtype=torch.bool)
        if getattr(self, "x_edge_mask", None) is not None:
            scan_xy = (
                (world_scan_points[:, :, :2] + self.terrain.cfg.border_size)
                / self.cfg.terrain.horizontal_scale
            ).round().long()
            scan_xy[..., 0] = torch.clip(scan_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
            scan_xy[..., 1] = torch.clip(scan_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
            scan_edge = self.x_edge_mask[scan_xy[..., 0], scan_xy[..., 1]]

        cmd_forward = torch.clamp(self._forward_reward_command_x(), min=0.0, max=1.0)
        front_center_x = (
            self.cfg.rewards.targeted_foothold_front_center_x
            + cmd_forward * self.cfg.rewards.targeted_foothold_front_lookahead
        )
        rear_center_x = (
            self.cfg.rewards.targeted_foothold_rear_center_x
            + cmd_forward * self.cfg.rewards.targeted_foothold_rear_lookahead
        )
        lateral_center_y = float(self.cfg.rewards.targeted_foothold_lateral_center_y)
        center_x = torch.stack(
            (front_center_x, front_center_x, rear_center_x, rear_center_x),
            dim=1,
        )[:, :num_feet]
        center_y = torch.tensor(
            [lateral_center_y, -lateral_center_y, lateral_center_y, -lateral_center_y],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0).repeat(self.num_envs, 1)[:, :num_feet]
        target_centers = torch.stack((center_x, center_y), dim=-1)

        search_x_half = float(self.cfg.rewards.targeted_foothold_search_x_half)
        search_y_half = float(self.cfg.rewards.targeted_foothold_search_y_half)
        retry_step_x = float(getattr(self.cfg.rewards, "targeted_foothold_retry_step_x", 0.25))
        min_target_z = float(getattr(self.cfg.rewards, "targeted_foothold_min_target_z", -0.1))
        max_scan_x = max(self.cfg.terrain.measured_points_x)
        min_center_x = min(
            float(self.cfg.rewards.targeted_foothold_front_center_x),
            float(self.cfg.rewards.targeted_foothold_rear_center_x),
        )
        if retry_step_x > 0.0:
            max_search_iters = max(
                1,
                int(np.ceil((max_scan_x + search_x_half - min_center_x) / retry_step_x)) + 1,
            )
        else:
            max_search_iters = 1

        local_scan_points = local_scan_points.unsqueeze(1).expand(-1, num_feet, -1, -1)
        world_scan_points = world_scan_points.unsqueeze(1).expand(-1, num_feet, -1, -1)

        scan_heights = self.measured_heights
        scan_min = scan_heights.min(dim=1, keepdim=True).values
        scan_max = scan_heights.max(dim=1, keepdim=True).values
        normalized_height = (scan_heights - scan_min) / (scan_max - scan_min).clamp_min(1.0e-4)
        score_base = (
            self.cfg.rewards.targeted_foothold_height_gain * normalized_height.unsqueeze(1)
            - self.cfg.rewards.targeted_foothold_edge_penalty * scan_edge.unsqueeze(1).float()
        )
        search_centers = target_centers.clone()
        remaining = active_envs.unsqueeze(-1).expand(-1, num_feet).clone()

        for _ in range(max_search_iters):
            searchable = remaining & (search_centers[..., 0] - search_x_half <= max_scan_x)
            if not torch.any(searchable):
                break

            delta = local_scan_points - search_centers.unsqueeze(2)
            dist_sq_anchor = torch.sum(torch.square(delta), dim=-1)
            candidate_mask = searchable.unsqueeze(-1) & (
                torch.abs(delta[..., 0]) <= search_x_half
            ) & (
                torch.abs(delta[..., 1]) <= search_y_half
            )

            score = score_base - self.cfg.rewards.targeted_foothold_anchor_penalty * dist_sq_anchor
            score = torch.where(candidate_mask, score, torch.full_like(score, -1.0e6))

            best_idx = torch.argmax(score, dim=-1)
            candidate_exists = candidate_mask.any(dim=-1)
            best_score = torch.gather(score, 2, best_idx.unsqueeze(-1)).squeeze(-1)
            valid = searchable & candidate_exists & torch.isfinite(best_score) & (best_score > -1.0e5)

            gather_idx = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
            target_world = torch.gather(world_scan_points, 2, gather_idx).squeeze(2)
            accepted = valid & (target_world[:, :, 2] >= min_target_z)

            candidate_targets_world = torch.where(
                accepted.unsqueeze(-1),
                target_world,
                candidate_targets_world,
            )
            candidate_valid |= accepted
            remaining &= ~accepted
            if not torch.any(remaining):
                break
            search_centers[..., 0] = torch.where(
                remaining,
                search_centers[..., 0] + retry_step_x,
                search_centers[..., 0],
            )

        return candidate_targets_world, candidate_valid

    def _reward_targeted_foothold_touchdown(self):
        feet_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        feet_pos = feet_states[:, :, 0:3]
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        prev_contact = self.targeted_foothold_last_contacts
        contact_filt = torch.logical_or(contact, prev_contact)
        swing_start = prev_contact & ~contact

        candidate_targets_world, candidate_valid = self._compute_targeted_foothold_candidates()
        need_new_target = swing_start | ((~contact_filt) & (~self.targeted_foothold_target_valid))
        update_mask = need_new_target & candidate_valid
        self.targeted_foothold_targets_world = torch.where(
            update_mask.unsqueeze(-1),
            candidate_targets_world,
            self.targeted_foothold_targets_world,
        )
        self.targeted_foothold_target_valid = torch.where(
            need_new_target,
            candidate_valid,
            self.targeted_foothold_target_valid,
        )

        first_contact = (
            self.targeted_foothold_air_time > self.cfg.rewards.targeted_foothold_min_air_time
        ) & contact_filt
        self.targeted_foothold_air_time += self.dt
        self.targeted_foothold_air_time *= ~contact_filt
        self.targeted_foothold_last_contacts = contact

        target_delta_xy = feet_pos[:, :, :2] - self.targeted_foothold_targets_world[:, :, :2]
        target_dist_sq = torch.sum(torch.square(target_delta_xy), dim=-1)
        foot_reward = torch.exp(
            -target_dist_sq / self.cfg.rewards.targeted_foothold_distance_sigma
        )
        foot_reward = foot_reward * first_contact.float() * self.targeted_foothold_target_valid.float()
        self.targeted_foothold_target_valid &= ~contact_filt
        self.targeted_foothold_targets_world = torch.where(
            self.targeted_foothold_target_valid.unsqueeze(-1),
            self.targeted_foothold_targets_world,
            torch.zeros_like(self.targeted_foothold_targets_world),
        )
        return foot_reward.sum(dim=1)

    def _build_camera_rays(self):
        height, width = self.raw_depth_shape
        fx = self.cfg.camera.focal_length * width / self.cfg.camera.horizontal_aperture
        fy = self.cfg.camera.focal_length * height / self.cfg.camera.vertical_aperture
        cx = width * 0.5
        cy = height * 0.5

        if self.cfg.camera.convention.lower() != "ros":
            raise ValueError(f"Unsupported camera convention: {self.cfg.camera.convention}")

        grid_x, grid_y = torch.meshgrid(
            torch.arange(start=0, end=width, dtype=torch.float, device=self.device),
            torch.arange(start=0, end=height, dtype=torch.float, device=self.device),
            indexing="xy",
        )
        pixels = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.ones(width * height, device=self.device)), dim=-1)
        pixels[:, :2] += 0.5

        pix_in_optical = torch.stack(
            (
                (pixels[:, 0] - cx) / fx,
                (pixels[:, 1] - cy) / fy,
                torch.ones(width * height, device=self.device),
            ),
            dim=-1,
        )
        # Isaac Lab converts optical frame (x right, y down, z forward) into robotics/world camera frame
        # (x forward, y left, z up) before applying the camera pose.
        dirs_camera_world = torch.stack(
            (
                pix_in_optical[:, 2],
                -pix_in_optical[:, 0],
                -pix_in_optical[:, 1],
            ),
            dim=-1,
        )
        dirs_camera_world = dirs_camera_world / torch.norm(
            dirs_camera_world, dim=-1, keepdim=True
        ).clamp_min(1e-6)

        camera_quat_ros = quat_from_euler_xyz(
            torch.tensor(self.cfg.camera.local_euler_xyz[0], device=self.device),
            torch.tensor(self.cfg.camera.local_euler_xyz[1], device=self.device),
            torch.tensor(self.cfg.camera.local_euler_xyz[2], device=self.device),
        ).view(1, 4)
        # Match Isaac Lab's quat_from_euler_xyz_tuple helper:
        # it stores quaternions in wxyz and flips the z component before passing
        # them into the camera offset config. In Isaac Gym xyzw layout, this means
        # flipping index 2 after the standard Euler->quat conversion.
        camera_quat_ros[:, 2] = -camera_quat_ros[:, 2]
        self.camera_local_pos = torch.tensor(
            self.cfg.camera.local_pos,
            device=self.device,
            dtype=torch.float,
        )

        # The configured quaternion is in Isaac Lab ROS camera convention:
        # forward +Z, up -Y, right +X.
        # Convert its local axes into the robot/base convention:
        # forward -> +X, left -> +Y, up -> +Z.
        right_base = quat_apply(camera_quat_ros, torch.tensor([[1.0, 0.0, 0.0]], device=self.device))
        down_base = quat_apply(camera_quat_ros, torch.tensor([[0.0, 1.0, 0.0]], device=self.device))
        forward_base = quat_apply(camera_quat_ros, torch.tensor([[0.0, 0.0, 1.0]], device=self.device))
        left_base = -right_base
        up_base = -down_base

        dirs_base = (
            dirs_camera_world[:, 0:1] * forward_base
            + dirs_camera_world[:, 1:2] * left_base
            + dirs_camera_world[:, 2:3] * up_base
        )
        dirs_base = dirs_base / torch.norm(dirs_base, dim=-1, keepdim=True).clamp_min(1e-6)
        self.camera_ray_dirs_base = dirs_base.contiguous()

    def _init_camera_randomization_buffers(self):
        self.camera_local_pos_error = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
            dtype=torch.float,
        )
        self.camera_local_pos_env = self.camera_local_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.camera_yaw_error = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self.camera_yaw_quat = torch.zeros(
            self.num_envs,
            4,
            device=self.device,
            dtype=torch.float,
        )
        self.camera_yaw_quat[:, 3] = 1.0

    def _resample_camera_randomization(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        xyz_min, xyz_max = getattr(self.cfg.camera, "xyz_error", [0.0, 0.0])
        if float(xyz_min) == 0.0 and float(xyz_max) == 0.0:
            self.camera_local_pos_error[env_ids] = 0.0
        else:
            self.camera_local_pos_error[env_ids] = torch_rand_float(
                float(xyz_min),
                float(xyz_max),
                (len(env_ids), 3),
                device=self.device,
            )
        self.camera_local_pos_env[env_ids] = self.camera_local_pos.unsqueeze(0) + self.camera_local_pos_error[env_ids]

        yaw_min_deg, yaw_max_deg = getattr(
            self.cfg.camera,
            "yaw_error_deg",
            getattr(self.cfg.camera, "yaw_error", [0.0, 0.0]),
        )
        yaw_min = math.radians(float(yaw_min_deg))
        yaw_max = math.radians(float(yaw_max_deg))
        if yaw_min == 0.0 and yaw_max == 0.0:
            self.camera_yaw_error[env_ids] = 0.0
            self.camera_yaw_quat[env_ids] = 0.0
            self.camera_yaw_quat[env_ids, 3] = 1.0
            return

        yaw_error = torch_rand_float(
            yaw_min,
            yaw_max,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.camera_yaw_error[env_ids] = yaw_error
        zeros = torch.zeros_like(yaw_error)
        self.camera_yaw_quat[env_ids] = quat_from_euler_xyz(zeros, zeros, yaw_error)

    def _get_camera_proxy_pose(self, env_ids=None):
        if env_ids is None:
            base_quat = self.base_quat
            root_states = self.root_states
            camera_local_pos = self.camera_local_pos_env
            camera_yaw_quat = self.camera_yaw_quat
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            base_quat = self.base_quat[env_ids]
            root_states = self.root_states[env_ids]
            camera_local_pos = self.camera_local_pos_env[env_ids]
            camera_yaw_quat = self.camera_yaw_quat[env_ids]

        camera_origins = root_states[:, :3] + quat_apply(base_quat, camera_local_pos)
        camera_ray_quat = quat_mul(base_quat, camera_yaw_quat)
        return camera_origins, camera_ray_quat

    def _init_warp_proxy_backend(self):
        self._proxy_backend = str(getattr(self.cfg.camera, "proxy_backend", "torch")).lower()
        self._warp_proxy_enabled = False
        self._warp_mesh = None
        self._warp_device = None
        self._warp_chunk_size = int(
            getattr(
                self.cfg.camera,
                "warp_raycast_chunk_size",
                max(self.raycast_chunk_size, 512),
            )
        )

        if self._proxy_backend != "warp":
            return
        if wp is None:
            print("Depth camera proxy backend requested 'warp' but warp is not installed. Falling back to torch.")
            return
        if not hasattr(wp, "from_torch"):
            print("Depth camera proxy backend requested 'warp' but this warp build has no torch interop. Falling back to torch.")
            return

        terrain_vertices = getattr(self.terrain, "vertices", None)
        terrain_triangles = getattr(self.terrain, "triangles", None)
        if terrain_vertices is None or terrain_triangles is None:
            print("Depth camera proxy backend requested 'warp' but terrain mesh is unavailable. Falling back to torch.")
            return

        try:
            _ensure_warp_initialized()
            self._warp_device = _torch_device_to_warp(self.device)

            vertices = np.asarray(terrain_vertices, dtype=np.float32).copy()
            triangles = np.asarray(terrain_triangles, dtype=np.int32).reshape(-1)
            border = float(self.terrain.cfg.border_size)
            vertices[:, 0] -= border
            vertices[:, 1] -= border

            self._warp_mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=self._warp_device),
                indices=wp.array(triangles, dtype=wp.int32, device=self._warp_device),
            )
            self._warp_proxy_enabled = True
            print(
                "Depth camera proxy backend: warp "
                f"(chunk_size={self._warp_chunk_size}, rays={self.camera_ray_dirs_base.shape[0]})"
            )
        except Exception as exc:
            print(f"Depth camera proxy backend failed to initialize warp mesh: {exc}. Falling back to torch.")
            self._warp_proxy_enabled = False
            self._warp_mesh = None
            self._warp_device = None

    def _sample_heightfield(self, xy_world):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(xy_world.shape[:-1], device=self.device, dtype=torch.float)

        scale = self.terrain.cfg.horizontal_scale
        border = self.terrain.cfg.border_size
        rows = self.height_samples.shape[0]
        cols = self.height_samples.shape[1]

        grid_x = (xy_world[..., 0] + border) / scale
        grid_y = (xy_world[..., 1] + border) / scale

        x0 = torch.floor(grid_x).long().clamp(0, rows - 2)
        y0 = torch.floor(grid_y).long().clamp(0, cols - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = (grid_x - x0.float()).clamp(0.0, 1.0)
        wy = (grid_y - y0.float()).clamp(0.0, 1.0)

        h00 = self.height_samples[x0, y0].float()
        h10 = self.height_samples[x1, y0].float()
        h01 = self.height_samples[x0, y1].float()
        h11 = self.height_samples[x1, y1].float()

        heights = (
            (1.0 - wx) * (1.0 - wy) * h00
            + wx * (1.0 - wy) * h10
            + (1.0 - wx) * wy * h01
            + wx * wy * h11
        )
        return heights * self.terrain.cfg.vertical_scale

    def _raycast_heightfield_chunk(self, ray_origins, ray_dirs):
        num_envs, num_rays, _ = ray_dirs.shape
        max_distance = self.cfg.camera.clipping_range
        distances = torch.full(
            (num_envs, num_rays),
            max_distance,
            device=self.device,
            dtype=torch.float,
        )

        active = torch.ones_like(ray_dirs[..., 2], dtype=torch.bool)
        left_t = torch.zeros_like(distances)
        right_t = torch.full_like(distances, max_distance)
        hit_mask = torch.zeros_like(active)

        coarse_ts = torch.linspace(
            0.0,
            max_distance,
            self.raycast_num_coarse_steps + 1,
            device=self.device,
            dtype=torch.float,
        )
        prev_t = coarse_ts[0]
        for curr_t in coarse_ts[1:]:
            sample_points = ray_origins.unsqueeze(1) + ray_dirs * curr_t
            terrain_height = self._sample_heightfield(sample_points[..., :2])
            signed_height = sample_points[..., 2] - terrain_height
            crossed = active & (signed_height <= 0.0)
            if torch.any(crossed):
                left_t[crossed] = prev_t
                right_t[crossed] = curr_t
                hit_mask |= crossed
                active &= ~crossed
            prev_t = curr_t
            if not torch.any(active):
                break

        if torch.any(hit_mask):
            for _ in range(self.raycast_num_refine_steps):
                mid_t = 0.5 * (left_t + right_t)
                sample_points = ray_origins.unsqueeze(1) + ray_dirs * mid_t.unsqueeze(-1)
                terrain_height = self._sample_heightfield(sample_points[..., :2])
                signed_height = sample_points[..., 2] - terrain_height
                go_right = hit_mask & (signed_height > 0.0)
                go_left = hit_mask & ~go_right
                left_t[go_right] = mid_t[go_right]
                right_t[go_left] = mid_t[go_left]
            distances[hit_mask] = right_t[hit_mask]

        return distances

    def _render_proxy_distance_to_camera_warp(self, env_ids=None):
        if not self._warp_proxy_enabled or self._warp_mesh is None:
            return self._render_proxy_distance_to_camera_torch(env_ids=env_ids)

        camera_origins, camera_ray_quat = self._get_camera_proxy_pose(env_ids=env_ids)
        num_envs = camera_origins.shape[0]
        num_rays = self.camera_ray_dirs_base.shape[0]

        ray_distances = []
        max_distance = float(self.cfg.camera.clipping_range)
        for start in range(0, num_rays, self._warp_chunk_size):
            end = min(start + self._warp_chunk_size, num_rays)
            chunk_size = end - start
            ray_dirs_base = self.camera_ray_dirs_base[start:end]
            ray_dirs_world = quat_apply(
                camera_ray_quat.repeat_interleave(chunk_size, dim=0),
                ray_dirs_base.unsqueeze(0).repeat(num_envs, 1, 1).reshape(-1, 3),
            ).view(num_envs, chunk_size, 3)

            flat_origins = (
                camera_origins.unsqueeze(1)
                .expand(-1, chunk_size, -1)
                .contiguous()
                .view(-1, 3)
            )
            flat_dirs = ray_dirs_world.contiguous().view(-1, 3)
            flat_distances = torch.empty(flat_origins.shape[0], device=self.device, dtype=torch.float32)
            flat_hits_found = torch.empty(flat_origins.shape[0], device=self.device, dtype=torch.int32)

            wp.launch(
                kernel=_raycast_mesh_distance_kernel,
                dim=flat_origins.shape[0],
                inputs=[
                    self._warp_mesh.id,
                    wp.from_torch(flat_origins, dtype=wp.vec3),
                    wp.from_torch(flat_dirs, dtype=wp.vec3),
                    wp.from_torch(flat_distances, dtype=wp.float32),
                    wp.from_torch(flat_hits_found, dtype=wp.int32),
                    max_distance,
                ],
                device=self._warp_device,
            )
            if hasattr(wp, "synchronize_device"):
                wp.synchronize_device(self._warp_device)
            else:  # pragma: no cover - API differs across warp versions
                wp.synchronize()

            ray_distances.append(flat_distances.view(num_envs, chunk_size))

        return torch.cat(ray_distances, dim=1).view(num_envs, *self.raw_depth_shape)

    def _render_proxy_distance_to_camera_torch(self, env_ids=None):
        camera_origins, camera_ray_quat = self._get_camera_proxy_pose(env_ids=env_ids)
        num_envs = camera_origins.shape[0]
        num_rays = self.camera_ray_dirs_base.shape[0]

        ray_distances = []
        for start in range(0, num_rays, self.raycast_chunk_size):
            end = min(start + self.raycast_chunk_size, num_rays)
            ray_dirs_base = self.camera_ray_dirs_base[start:end]
            ray_dirs_world = quat_apply(
                camera_ray_quat.repeat_interleave(end - start, dim=0),
                ray_dirs_base.unsqueeze(0).repeat(num_envs, 1, 1).reshape(-1, 3),
            ).view(num_envs, end - start, 3)
            ray_distances.append(self._raycast_heightfield_chunk(camera_origins, ray_dirs_world))

        return torch.cat(ray_distances, dim=1).view(num_envs, *self.raw_depth_shape)

    def _render_proxy_distance_to_camera(self, env_ids=None):
        if self._warp_proxy_enabled:
            return self._render_proxy_distance_to_camera_warp(env_ids=env_ids)
        return self._render_proxy_distance_to_camera_torch(env_ids=env_ids)

    def _apply_proxy_depth_noise(self, depth_images):
        noise_gaussian = float(getattr(self.cfg.camera, "noise_gaussian", 0.0))
        noise_dropout = float(getattr(self.cfg.camera, "noise_dropout", 0.0))

        if noise_gaussian > 0.0:
            depth_images.add_(torch.randn_like(depth_images) * noise_gaussian)
        if noise_dropout > 0.0:
            keep_mask = (torch.rand_like(depth_images) > noise_dropout).to(depth_images.dtype)
            depth_images.mul_(keep_mask)
        if noise_gaussian > 0.0 or noise_dropout > 0.0:
            depth_images.clamp_(0.0, self.cfg.camera.clipping_range)
        return depth_images

    def compute_observations(self):
        feet_heights = self._get_feet_heights()
        super().compute_observations()
        if hasattr(self, "terrain_ids"):
            terrain_ids = self.terrain_ids.unsqueeze(-1).to(self.privileged_obs_buf.dtype)
        else:
            terrain_ids = -torch.ones(
                self.num_envs,
                self.terrain_id_dim,
                device=self.device,
                dtype=self.privileged_obs_buf.dtype,
            )
        fall_recovery = self._fall_recovery_mask().unsqueeze(-1).to(self.privileged_obs_buf.dtype)
        self.privileged_obs_buf = torch.cat(
            (self.privileged_obs_buf, feet_heights, terrain_ids, fall_recovery),
            dim=-1,
        )

    def _capture_proxy_depth(self, env_ids=None):
        timing_store = self._active_step_timing
        timed_start = self._timing_start()
        depth_images = self._render_proxy_distance_to_camera(env_ids=env_ids)
        self._timing_stop(timed_start, "parkour_proxy_render", timing_store)

        timed_start = self._timing_start()
        top = self.cfg.camera.crop_top
        bottom = self.cfg.camera.crop_bottom
        left = self.cfg.camera.crop_left
        right = self.cfg.camera.crop_right
        h_end = depth_images.shape[1] - bottom if bottom > 0 else depth_images.shape[1]
        w_end = depth_images.shape[2] - right if right > 0 else depth_images.shape[2]
        depth_images = depth_images[:, top:h_end, left:w_end]

        depth_images = F.interpolate(
            depth_images.unsqueeze(1),
            size=self.depth_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        depth_images = self._apply_proxy_depth_noise(depth_images)
        depth_images = (depth_images / self.cfg.camera.clipping_range) - 0.5
        depth_images.clamp_(-0.5, 0.5)
        self._timing_stop(timed_start, "parkour_depth_postprocess", timing_store)
        return depth_images

    def _update_depth_camera_buffer(self):
        timing_store = self._active_step_timing
        periodic_update = self.common_step_counter % self.cfg.camera.update_interval == 0 or not self._depth_initialized
        frame = None
        if periodic_update:
            timed_start = self._timing_start()
            frame = self._capture_proxy_depth()
            self._timing_stop(timed_start, "parkour_depth_capture_full", timing_store)
            timed_start = self._timing_start()
            self.depth_camera_buf = torch.roll(self.depth_camera_buf, shifts=-1, dims=1)
            self.depth_camera_buf[:, -1] = frame
            self._timing_stop(timed_start, "parkour_depth_buffer_full", timing_store)

        if not self._depth_initialized:
            timed_start = self._timing_start()
            self.depth_camera_buf[:] = frame.unsqueeze(1).repeat(1, self.depth_buffer_len, 1, 1)
            self._depth_initialized = True
            self._timing_stop(timed_start, "parkour_depth_buffer_init", timing_store)

        if torch.any(self._force_depth_refresh):
            refresh_ids = self._force_depth_refresh.nonzero(as_tuple=False).flatten()
            if refresh_ids.numel() > 0:
                if frame is not None:
                    refresh_frame = frame[refresh_ids]
                else:
                    timed_start = self._timing_start()
                    refresh_frame = self._capture_proxy_depth(env_ids=refresh_ids)
                    self._timing_stop(timed_start, "parkour_depth_capture_reset", timing_store)
                timed_start = self._timing_start()
                repeated = refresh_frame.unsqueeze(1).repeat(1, self.depth_buffer_len, 1, 1)
                self.depth_camera_buf[refresh_ids] = repeated
                self._force_depth_refresh[refresh_ids] = False
                self._timing_stop(timed_start, "parkour_depth_buffer_reset", timing_store)

        self.additional_obs_buf["depth_camera"] = self.depth_camera_buf
