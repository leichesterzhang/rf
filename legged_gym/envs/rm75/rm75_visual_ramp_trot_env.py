"""Vision-conditioned RM75 trot task for flat ground and continuous ramps."""

import math

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse
import torch

from legged_gym.envs.go2.go2_parkour_env import Go2ParkourRobot


class RM75VisualRampTrotRobot(Go2ParkourRobot):
    """Keeps the flat 3 m/s trot while adapting to the configured ramp angle."""

    _TROT_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # The common terrain constructor initially creates actors at each block's
        # center height. Explicitly reset every RM75 onto its start platform before
        # the runner collects the first rollout.
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids)
        self.reset_buf.zero_()
        self.episode_length_buf.zero_()

    def _init_buffers(self):
        super()._init_buffers()

        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        missing_names = [name for name in self._TROT_FOOT_NAMES if name not in body_names]
        if missing_names:
            raise RuntimeError(
                "RM75 visual ramp trot requires four named feet; "
                f"missing={missing_names}, available={body_names}"
            )
        self.trot_feet_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], foot_name
                )
                for foot_name in self._TROT_FOOT_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.trot_last_contacts = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.trot_contact_ema = torch.full(
            (self.num_envs, 4),
            float(self.cfg.rewards.trot_flat_contact_duty),
            dtype=torch.float,
            device=self.device,
        )
        self.visual_slope_factor = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.local_terrain_slope = torch.zeros_like(self.visual_slope_factor)
        self.local_terrain_angle = torch.zeros_like(self.visual_slope_factor)
        self.terrain_target_speed = torch.full_like(
            self.visual_slope_factor, float(self.cfg.rewards.flat_target_speed)
        )
        self.terrain_tangent_world = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.terrain_tangent_world[:, 0] = 1.0
        self.terrain_normal_world = torch.zeros_like(self.terrain_tangent_world)
        self.terrain_normal_world[:, 2] = 1.0
        self.terrain_tangent_speed = torch.zeros_like(self.visual_slope_factor)
        self.terrain_normal_velocity = torch.zeros_like(self.visual_slope_factor)
        self.random_section_spawn = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._terrain_profile_step = -1

    def _training_iteration(self):
        return int(self.common_step_counter // max(int(self.num_steps_per_env), 1))

    def _slope_speed_target(self):
        iteration = self._training_iteration()
        curriculum = getattr(self.cfg.rewards, "slope_speed_curriculum", [])
        if not curriculum:
            return float(self.cfg.rewards.slope_target_speed)
        for stage in curriculum:
            start_iter = int(stage["start_iter"])
            end_iter = int(stage["end_iter"])
            if iteration <= end_iter:
                ratio = (iteration - start_iter) / max(end_iter - start_iter, 1)
                ratio = min(max(ratio, 0.0), 1.0)
                return float(stage["start_speed"]) + ratio * (
                    float(stage["end_speed"]) - float(stage["start_speed"])
                )
        return float(curriculum[-1]["end_speed"])

    def _max_allowed_terrain_level(self):
        iteration = self._training_iteration()
        max_level = int(self.cfg.terrain.max_init_terrain_level)
        for stage in getattr(self.cfg.terrain, "level_curriculum", []):
            if iteration >= int(stage["iter"]):
                max_level = int(stage["max_level"])
        return min(max_level, int(self.cfg.terrain.num_rows) - 1)

    def _apply_terrain_level_cap(self, env_ids):
        if len(env_ids) == 0 or not hasattr(self, "terrain_levels"):
            return
        max_level = self._max_allowed_terrain_level()
        self.terrain_levels[env_ids] = torch.clamp(
            self.terrain_levels[env_ids], min=0, max=max_level
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]
        if hasattr(self, "terrain_spawn_origins"):
            self.reset_origins[env_ids] = self.terrain_spawn_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]
            ]
            self.progress_origins[env_ids] = self.reset_origins[env_ids]
        if hasattr(self, "terrain_id_map"):
            self.terrain_ids[env_ids] = self.terrain_id_map[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]
            ]
            self._update_env_command_ranges()

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        regular_env_ids = env_ids[~self.random_section_spawn[env_ids]]
        if len(regular_env_ids) > 0:
            super()._update_terrain_curriculum(regular_env_ids)
        self._apply_terrain_level_cap(env_ids)

    def _sample_longitudinal_slope(self, xy_world, half_span=0.20):
        before = xy_world.clone()
        after = xy_world.clone()
        before[:, 0] -= float(half_span)
        after[:, 0] += float(half_span)
        height_before = self._sample_heightfield(before)
        height_after = self._sample_heightfield(after)
        return (height_after - height_before) / (2.0 * float(half_span))

    def _update_terrain_profile(self, force=False):
        if not force and self._terrain_profile_step == self.common_step_counter:
            return

        root_xy = self.root_states[:, :2]
        local_slope = self._sample_longitudinal_slope(root_xy)
        ahead_xy = root_xy.clone()
        ahead_xy[:, 0] += float(self.cfg.rewards.slope_lookahead_distance)
        ahead_slope = self._sample_longitudinal_slope(ahead_xy)

        slope_for_detection = torch.maximum(torch.abs(local_slope), torch.abs(ahead_slope))
        start_tan = math.tan(math.radians(float(self.cfg.rewards.slope_detect_start_deg)))
        full_tan = math.tan(math.radians(float(self.cfg.rewards.slope_detect_full_deg)))
        raw_factor = torch.clamp(
            (slope_for_detection - start_tan) / max(full_tan - start_tan, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        release = float(self.cfg.rewards.slope_mode_release)
        self.visual_slope_factor = torch.maximum(
            raw_factor,
            self.visual_slope_factor * release,
        )

        tangent = torch.stack(
            (torch.ones_like(local_slope), torch.zeros_like(local_slope), local_slope),
            dim=1,
        )
        normal = torch.stack(
            (-local_slope, torch.zeros_like(local_slope), torch.ones_like(local_slope)),
            dim=1,
        )
        self.terrain_tangent_world = torch.nn.functional.normalize(tangent, dim=1)
        self.terrain_normal_world = torch.nn.functional.normalize(normal, dim=1)
        world_velocity = self.root_states[:, 7:10]
        self.terrain_tangent_speed = torch.sum(
            world_velocity * self.terrain_tangent_world, dim=1
        )
        self.terrain_normal_velocity = torch.sum(
            world_velocity * self.terrain_normal_world, dim=1
        )
        flat_speed = float(self.cfg.rewards.flat_target_speed)
        slope_speed = self._slope_speed_target()
        self.terrain_target_speed = flat_speed + self.visual_slope_factor * (
            slope_speed - flat_speed
        )
        self.local_terrain_slope = local_slope
        self.local_terrain_angle = torch.atan(local_slope)
        self._terrain_profile_step = self.common_step_counter

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_terrain_profile(force=True)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.trot_last_contacts[env_ids] = False
        self.trot_contact_ema[env_ids] = float(self.cfg.rewards.trot_flat_contact_duty)
        self.visual_slope_factor[env_ids] = 0.0
        self._terrain_profile_step = -1

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return

        self.root_states[env_ids, 7:13] = 0.0
        self.random_section_spawn[env_ids] = False
        start_iteration = int(self.cfg.terrain.random_section_spawn_start_iter)
        if self._training_iteration() >= start_iteration and hasattr(self, "terrain_ids"):
            ramp_mask = self.terrain_ids[env_ids] == int(self.cfg.terrain.ramp_terrain_id)
            random_mask = (
                torch.rand(len(env_ids), device=self.device)
                < float(self.cfg.terrain.random_section_spawn_probability)
            ) & ramp_mask
            random_env_ids = env_ids[random_mask]
            if len(random_env_ids) > 0:
                block_start_x = (
                    self.env_origins[random_env_ids, 0]
                    - 0.5 * float(self.cfg.terrain.terrain_length)
                )
                section = torch.randint(
                    0, 3, (len(random_env_ids),), device=self.device
                )
                ranges = torch.tensor(
                    self.cfg.terrain.random_section_spawn_ranges,
                    device=self.device,
                    dtype=torch.float,
                )
                selected_ranges = ranges[section]
                local_x = selected_ranges[:, 0] + torch.rand(
                    len(random_env_ids), device=self.device
                ) * (selected_ranges[:, 1] - selected_ranges[:, 0])
                self.root_states[random_env_ids, 0] = block_start_x + local_x
                self.root_states[random_env_ids, 1] = self.env_origins[random_env_ids, 1]

                xy_world = self.root_states[random_env_ids, :2]
                slope = self._sample_longitudinal_slope(xy_world)
                ground_height = self._sample_heightfield(xy_world)
                normal_z = torch.rsqrt(1.0 + torch.square(slope))
                self.root_states[random_env_ids, 2] = ground_height + (
                    float(self.cfg.rewards.base_height_target) / normal_z
                )
                zero = torch.zeros_like(slope)
                pitch = -torch.atan(slope)
                self.root_states[random_env_ids, 3:7] = quat_from_euler_xyz(
                    zero, pitch, zero
                )
                self.progress_origins[random_env_ids, :2] = self.root_states[
                    random_env_ids, :2
                ]
                self.random_section_spawn[random_env_ids] = True

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def check_termination(self):
        super().check_termination()
        base_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if len(self.base_termination_contact_indices) > 0:
            threshold = float(self.cfg.rewards.fall_contact_force_threshold)
            base_contact = torch.any(
                torch.norm(
                    self.contact_forces[:, self.base_termination_contact_indices, :],
                    dim=-1,
                )
                > threshold,
                dim=1,
            )
            self.reset_buf |= base_contact
        self._record_termination_reason("base_contact", base_contact)

        # At 3 m/s an intact robot reaches the front edge of a 12 m block well
        # before the episode timer. Treat that event as successful completion so
        # PPO bootstraps it like a timeout and does not apply a terminal penalty.
        # Backward/side exits and exits coincident with any fall remain failures.
        margin = float(getattr(self.cfg.terrain, "reset_outside_block_margin", 0.0))
        half_length = 0.5 * float(self.cfg.terrain.terrain_length) + margin
        half_width = 0.5 * float(self.cfg.terrain.terrain_width) + margin
        relative_pos = self.root_states[:, :2] - self.env_origins[:, :2]
        forward_exit = relative_pos[:, 0] > half_length
        side_or_backward_exit = (torch.abs(relative_pos[:, 1]) > half_width) | (
            relative_pos[:, 0] < -half_length
        )
        other_failure = side_or_backward_exit.clone()
        for reason, mask in self.termination_reason_masks.items():
            if reason not in ("time_out", "out_of_block"):
                other_failure |= mask
        forward_completion = forward_exit & ~other_failure
        self.time_out_buf |= forward_completion
        if "time_out" in self.termination_reason_masks:
            self.termination_reason_masks["time_out"] = (
                self.time_out_buf & ~forward_completion
            )
        if "out_of_block" in self.termination_reason_masks:
            self.termination_reason_masks["out_of_block"] &= ~forward_completion
        self._record_termination_reason("forward_completion", forward_completion)

    def _get_base_height(self):
        self._update_terrain_profile()
        ground_height = self._sample_heightfield(self.root_states[:, :2])
        vertical_height = self.root_states[:, 2] - ground_height
        return vertical_height * self.terrain_normal_world[:, 2]

    def _reward_tracking_lin_vel(self):
        self._update_terrain_profile()
        tangent_error = torch.square(
            self.terrain_target_speed - self.terrain_tangent_speed
        )
        lateral_error = torch.square(self.root_states[:, 8])
        return torch.exp(
            -(tangent_error + lateral_error) / float(self.cfg.rewards.tracking_sigma)
        )

    def _reward_lin_vel_z(self):
        self._update_terrain_profile()
        return torch.square(self.terrain_normal_velocity)

    def _reward_orientation(self):
        self._update_terrain_profile()
        normal_body = quat_rotate_inverse(self.base_quat, self.terrain_normal_world)
        return torch.sum(torch.square(normal_body[:, :2]), dim=1)

    def _reward_lateral_velocity(self):
        return torch.square(self.root_states[:, 8])

    def _reward_speed_tracking_error(self):
        self._update_terrain_profile()
        return torch.square(self.terrain_target_speed - self.terrain_tangent_speed)

    def _reward_downhill_overspeed(self):
        self._update_terrain_profile()
        downhill = self.local_terrain_slope < -math.tan(math.radians(2.0))
        overspeed = torch.clamp(
            self.terrain_tangent_speed - self.terrain_target_speed,
            min=0.0,
        )
        return torch.square(overspeed) * downhill.float()

    def _reward_trot_gait(self):
        self._update_terrain_profile()
        contact_threshold = float(self.cfg.rewards.trot_contact_force_threshold)
        contacts = self.contact_forces[:, self.trot_feet_indices, 2] > contact_threshold
        contact_filt = torch.logical_or(contacts, self.trot_last_contacts)
        self.trot_last_contacts[:] = contacts

        alpha = float(self.cfg.rewards.trot_contact_ema_alpha)
        self.trot_contact_ema.mul_(alpha).add_(contact_filt.float(), alpha=1.0 - alpha)

        contact = contact_filt.float()
        diagonal_a = 0.5 * (contact[:, 0] + contact[:, 3])
        diagonal_b = 0.5 * (contact[:, 1] + contact[:, 2])
        diagonal_sync_error = torch.abs(contact[:, 0] - contact[:, 3])
        diagonal_sync_error += torch.abs(contact[:, 1] - contact[:, 2])
        diagonal_sync_score = (1.0 - 0.5 * diagonal_sync_error).clamp(0.0, 1.0)
        diagonal_opposition_score = torch.abs(diagonal_a - diagonal_b)

        mean_duty = self.trot_contact_ema.mean(dim=1, keepdim=True)
        duty_spread = torch.abs(self.trot_contact_ema - mean_duty).mean(dim=1)
        duty_balance_score = (
            1.0
            - duty_spread / float(self.cfg.rewards.trot_duty_balance_tolerance)
        ).clamp(0.0, 1.0)

        flat_duty = float(self.cfg.rewards.trot_flat_contact_duty)
        uphill_duty = float(self.cfg.rewards.trot_uphill_contact_duty)
        downhill_duty = float(self.cfg.rewards.trot_downhill_contact_duty)
        slope_duty = torch.where(
            self.local_terrain_slope >= 0.0,
            torch.full_like(self.local_terrain_slope, uphill_duty),
            torch.full_like(self.local_terrain_slope, downhill_duty),
        )
        target_duty = flat_duty + self.visual_slope_factor * (slope_duty - flat_duty)
        target_duty_score = torch.exp(
            -torch.square(mean_duty.squeeze(1) - target_duty)
            / float(self.cfg.rewards.trot_contact_duty_sigma)
        )
        active = self.terrain_target_speed > float(self.cfg.rewards.trot_min_command_speed)
        return (
            diagonal_sync_score
            * diagonal_opposition_score
            * duty_balance_score
            * target_duty_score
            * active.float()
        )

    def _reward_feet_slip(self):
        contact_threshold = float(self.cfg.rewards.trot_contact_force_threshold)
        contacts = self.contact_forces[:, self.trot_feet_indices, 2] > contact_threshold
        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        feet_world_velocity = body_states[:, self.trot_feet_indices, 7:10]
        return torch.sum(
            torch.sum(torch.square(feet_world_velocity), dim=2) * contacts.float(),
            dim=1,
        )
