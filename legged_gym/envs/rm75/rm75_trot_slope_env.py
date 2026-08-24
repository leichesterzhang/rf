import math

import torch

from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz

from legged_gym.envs.go2.go2_parkour_env import Go2ParkourRobot
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi


class RM75TrotSlopeRobot(Go2ParkourRobot):
    """Forward RM75 diagonal-trot task on flat ground and a 35-degree ramp."""

    MODE_STAND = 0
    MODE_FORWARD = 1
    MODE_SIDE_LEFT = 2
    MODE_SIDE_RIGHT = 3

    RAMP_TERRAIN_ID = 8

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._ensure_trot_buffers()

    def _ensure_trot_buffers(self):
        if not hasattr(self, "trot_feet_indices") and hasattr(self, "gym") and self.envs:
            foot_names = getattr(
                self.cfg.rewards,
                "trot_foot_names",
                ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            )
            foot_handles = [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0],
                    self.actor_handles[0],
                    foot_name,
                )
                for foot_name in foot_names
            ]
            if len(foot_handles) != 4 or any(handle < 0 for handle in foot_handles):
                raise ValueError(
                    "trot_foot_names must resolve to FL, FR, RL, RR rigid bodies"
                )
            if len(set(foot_handles)) != 4:
                raise ValueError("trot_foot_names must resolve to four distinct rigid bodies")
            self.trot_feet_indices = torch.tensor(
                foot_handles,
                device=self.device,
                dtype=torch.long,
            )

        if not hasattr(self, "locomotion_mode"):
            self.locomotion_mode = torch.full(
                (self.num_envs,),
                self.MODE_FORWARD,
                device=self.device,
                dtype=torch.long,
            )
            self.trot_target_heading = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            )
            self.trot_terrain_ids = torch.tensor(
                getattr(self.cfg.rewards, "trot_terrain_ids", [8, 11, 12]),
                device=self.device,
                dtype=torch.long,
            )

        if hasattr(self, "height_points") and not hasattr(self, "trot_plane_x"):
            plane_x = self.height_points[0, :, 0]
            plane_y = self.height_points[0, :, 1]
            central_mask = (
                (torch.abs(plane_x) <= float(getattr(self.cfg.rewards, "terrain_normal_scan_x", 0.5)))
                & (torch.abs(plane_y) <= float(getattr(self.cfg.rewards, "terrain_normal_scan_y", 0.4)))
            )
            if torch.any(central_mask):
                plane_x = plane_x[central_mask]
                plane_y = plane_y[central_mask]
                self.trot_plane_height_indices = central_mask.nonzero(as_tuple=False).flatten()
            else:
                self.trot_plane_height_indices = torch.arange(
                    plane_x.numel(), device=self.device, dtype=torch.long
                )
            self.trot_plane_x = plane_x - plane_x.mean()
            self.trot_plane_y = plane_y - plane_y.mean()
            self.trot_plane_x_denom = torch.sum(torch.square(self.trot_plane_x)).clamp_min(1.0e-6)
            self.trot_plane_y_denom = torch.sum(torch.square(self.trot_plane_y)).clamp_min(1.0e-6)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return super().reset_idx(env_ids)
        self._ensure_trot_buffers()
        self._sample_locomotion_modes(env_ids)
        super().reset_idx(env_ids)

    def _sample_locomotion_modes(self, env_ids):
        if bool(getattr(self.cfg.commands, "forward_only", False)):
            self.locomotion_mode[env_ids] = self.MODE_FORWARD
            self.trot_target_heading[env_ids] = 0.0
            return

        probabilities = torch.tensor(
            getattr(self.cfg.commands, "locomotion_mode_probabilities", [0.0, 1.0, 0.0, 0.0]),
            device=self.device,
            dtype=torch.float,
        )
        if probabilities.numel() != 4 or bool(torch.any(probabilities < 0.0)):
            raise ValueError("locomotion_mode_probabilities must contain four non-negative values")
        probability_sum = probabilities.sum()
        if float(probability_sum) <= 0.0:
            raise ValueError("locomotion_mode_probabilities must have a positive sum")
        probabilities = probabilities / probability_sum
        self.locomotion_mode[env_ids] = torch.multinomial(
            probabilities,
            len(env_ids),
            replacement=True,
        )

        target_heading = torch.zeros(len(env_ids), device=self.device, dtype=torch.float)
        mode = self.locomotion_mode[env_ids]
        # At yaw=-pi/2, positive body Y points along world +X. At yaw=+pi/2,
        # negative body Y points along world +X.
        target_heading[mode == self.MODE_SIDE_LEFT] = -0.5 * math.pi
        target_heading[mode == self.MODE_SIDE_RIGHT] = 0.5 * math.pi
        self.trot_target_heading[env_ids] = target_heading

    def _randomize_omni_yaw(self, env_ids):
        """Set the reset heading selected by the episode locomotion mode."""
        if len(env_ids) == 0:
            return
        self._ensure_trot_buffers()
        yaw = self.trot_target_heading[env_ids]
        zeros = torch.zeros_like(yaw)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)

    def _current_trot_speed_stage(self):
        stages = getattr(self.cfg.commands, "trot_speed_curriculum", None)
        if not stages:
            return {
                "flat_forward": 3.0,
                "slope_forward": 1.5,
                "flat_lateral": 0.8,
                "slope_lateral": 0.5,
            }

        current_iter = self.common_step_counter // self.num_steps_per_env
        selected = stages[0]
        for stage in stages:
            if current_iter >= int(stage["iter"]):
                selected = stage
            else:
                break
        return selected

    def _sample_mode_speed(self, maximum_speed):
        if bool(getattr(self.cfg.commands, "fixed_target_speed", False)):
            return maximum_speed

        minimum_ratio = float(getattr(self.cfg.commands, "minimum_speed_ratio", 0.35))
        minimum_absolute = float(getattr(self.cfg.commands, "minimum_moving_speed", 0.10))
        minimum_speed = torch.minimum(
            maximum_speed,
            torch.maximum(
                maximum_speed * minimum_ratio,
                torch.full_like(maximum_speed, minimum_absolute),
            ),
        )
        speed = minimum_speed + torch.rand_like(maximum_speed) * (maximum_speed - minimum_speed)

        target_probability = float(getattr(self.cfg.commands, "target_speed_probability", 0.35))
        if target_probability > 0.0:
            use_target = torch.rand_like(speed) < target_probability
            speed = torch.where(use_target, maximum_speed, speed)
        return speed

    def _resample_commands(self, env_ids):
        """Sample mutually exclusive forward or lateral velocity commands."""
        if len(env_ids) == 0:
            return
        self._ensure_trot_buffers()

        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        self.commands[env_ids, :4] = 0.0
        self.stop_heading[env_ids] = False

        stage = self._current_trot_speed_stage()
        if hasattr(self, "terrain_ids"):
            slope_mask = self.terrain_ids[env_ids] == self.RAMP_TERRAIN_ID
        else:
            slope_mask = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)

        forward_max = torch.where(
            slope_mask,
            torch.full((len(env_ids),), float(stage["slope_forward"]), device=self.device),
            torch.full((len(env_ids),), float(stage["flat_forward"]), device=self.device),
        )
        lateral_max = torch.where(
            slope_mask,
            torch.full((len(env_ids),), float(stage["slope_lateral"]), device=self.device),
            torch.full((len(env_ids),), float(stage["flat_lateral"]), device=self.device),
        )

        mode = self.locomotion_mode[env_ids]
        moving_forward = mode == self.MODE_FORWARD
        side_left = mode == self.MODE_SIDE_LEFT
        side_right = mode == self.MODE_SIDE_RIGHT
        moving_lateral = side_left | side_right

        forward_speed = self._sample_mode_speed(forward_max)
        lateral_speed = self._sample_mode_speed(lateral_max)
        self.commands[env_ids[moving_forward], 0] = forward_speed[moving_forward]
        self.commands[env_ids[side_left], 1] = lateral_speed[side_left]
        self.commands[env_ids[side_right], 1] = -lateral_speed[side_right]
        self.commands[env_ids, 3] = self.trot_target_heading[env_ids]

        stand_ids = env_ids[~(moving_forward | moving_lateral)]
        self.commands[stand_ids, :3] = 0.0
        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

    def check_termination(self):
        super().check_termination()

        base_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if len(self.base_termination_contact_indices) > 0:
            base_contact = torch.any(
                torch.norm(
                    self.contact_forces[:, self.base_termination_contact_indices, :],
                    dim=-1,
                )
                > 1.0,
                dim=1,
            )
        self._record_termination_reason("trot_base_contact", base_contact)
        self.reset_buf |= base_contact

        excessive_tilt = (
            torch.abs(self.rpy[:, 0]) > float(getattr(self.cfg.termination, "trot_max_roll", 1.0))
        ) | (
            torch.abs(self.rpy[:, 1]) > float(getattr(self.cfg.termination, "trot_max_pitch", 1.15))
        )
        self._record_termination_reason("trot_excessive_tilt", excessive_tilt)
        self.reset_buf |= excessive_tilt

        timeout_s = float(getattr(self.cfg.rewards, "trot_no_progress_timeout_s", 3.0))
        if timeout_s <= 0.0:
            return

        command_speed = torch.norm(self.commands[:, :2], dim=1)
        monitor = command_speed > float(getattr(self.cfg.rewards, "trot_command_threshold", 0.15))
        progress = self._forward_progress()
        progress_step = float(getattr(self.cfg.rewards, "trot_no_progress_min_delta", 0.08))
        improved = progress > (self.max_forward_progress + progress_step)
        self.max_forward_progress = torch.where(improved, progress, self.max_forward_progress)
        self.no_progress_steps = torch.where(
            monitor,
            torch.where(improved, torch.zeros_like(self.no_progress_steps), self.no_progress_steps + 1),
            torch.zeros_like(self.no_progress_steps),
        )
        timeout_steps = max(1, int(math.ceil(timeout_s / self.dt)))
        no_progress = monitor & (self.no_progress_steps >= timeout_steps)
        self._record_termination_reason("trot_no_progress", no_progress)
        self.reset_buf |= no_progress

    def _trot_active_mask(self):
        command_speed = torch.norm(self.commands[:, :2], dim=1)
        active = command_speed > float(getattr(self.cfg.rewards, "trot_command_threshold", 0.15))
        if self.trot_terrain_ids.numel() > 0 and hasattr(self, "terrain_ids"):
            active &= self._terrain_id_mask(self.trot_terrain_ids)
        return active

    def _foot_contacts(self):
        return (self.contact_forces[:, self.trot_feet_indices, 2] > 1.0).float()

    def _trot_foot_heights(self):
        base_height = self._get_base_height()
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[
            :, self.trot_feet_indices, 0:3
        ]
        delta_feet = feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        feet_to_base_height = (delta_feet * self.projected_gravity.unsqueeze(1)).sum(-1)
        return torch.clamp(base_height.unsqueeze(1) - feet_to_base_height, min=0.0)

    def _reward_tracking_lin_vel(self):
        error_sq = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        mode = self.locomotion_mode
        lateral_mode = (mode == self.MODE_SIDE_LEFT) | (mode == self.MODE_SIDE_RIGHT)

        forward_sigma = float(getattr(self.cfg.rewards, "tracking_forward_sigma", 0.50))
        lateral_sigma = float(getattr(self.cfg.rewards, "tracking_lateral_sigma", 0.35))
        cross_sigma = float(getattr(self.cfg.rewards, "tracking_cross_sigma", 0.20))
        stand_sigma = float(getattr(self.cfg.rewards, "tracking_stand_sigma", 0.12))

        sigma_x = torch.full((self.num_envs,), forward_sigma, device=self.device)
        sigma_y = torch.full((self.num_envs,), cross_sigma, device=self.device)
        sigma_x = torch.where(lateral_mode, torch.full_like(sigma_x, cross_sigma), sigma_x)
        sigma_y = torch.where(lateral_mode, torch.full_like(sigma_y, lateral_sigma), sigma_y)
        stand_mode = mode == self.MODE_STAND
        sigma_x = torch.where(stand_mode, torch.full_like(sigma_x, stand_sigma), sigma_x)
        sigma_y = torch.where(stand_mode, torch.full_like(sigma_y, stand_sigma), sigma_y)

        scaled_error = error_sq[:, 0] / sigma_x.clamp_min(1.0e-6)
        scaled_error += error_sq[:, 1] / sigma_y.clamp_min(1.0e-6)
        return torch.exp(-scaled_error)

    def _reward_trot_diagonal_sync(self):
        if self.trot_feet_indices.numel() < 4:
            return torch.zeros(self.num_envs, device=self.device)
        contact = self._foot_contacts()
        penalty = 0.5 * (
            torch.abs(contact[:, 0] - contact[:, 3])
            + torch.abs(contact[:, 1] - contact[:, 2])
        )
        return penalty * self._trot_active_mask().float()

    def _reward_trot_diagonal_opposition(self):
        if self.trot_feet_indices.numel() < 4:
            return torch.zeros(self.num_envs, device=self.device)
        contact = self._foot_contacts()
        diagonal_a = 0.5 * (contact[:, 0] + contact[:, 3])
        diagonal_b = 0.5 * (contact[:, 1] + contact[:, 2])
        # Penalize simultaneous support by both diagonal pairs while allowing
        # the short all-feet-air phase that can appear in a 3 m/s trot.
        penalty = diagonal_a * diagonal_b
        return penalty * self._trot_active_mask().float()

    def _reward_trot_foot_height_sync(self):
        if self.trot_feet_indices.numel() < 4:
            return torch.zeros(self.num_envs, device=self.device)
        foot_height = self._trot_foot_heights()
        sigma = float(getattr(self.cfg.rewards, "trot_foot_height_sigma", 0.05))
        penalty = (
            torch.square(foot_height[:, 0] - foot_height[:, 3])
            + torch.square(foot_height[:, 1] - foot_height[:, 2])
        ) / max(sigma * sigma, 1.0e-6)
        return penalty * self._trot_active_mask().float()

    def _reward_feet_slip(self):
        feet_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[
            :, self.trot_feet_indices
        ]
        contact = self._foot_contacts()
        tangential_speed_sq = torch.sum(torch.square(feet_state[:, :, 7:9]), dim=-1)
        return torch.sum(tangential_speed_sq * contact, dim=1)

    def _reward_trot_heading_error(self):
        heading_error = wrap_to_pi(self.trot_target_heading - self._root_yaw())
        return torch.square(heading_error)

    def _reward_terrain_normal_alignment(self):
        if not torch.is_tensor(self.measured_heights) or not hasattr(self, "trot_plane_x"):
            return torch.zeros(self.num_envs, device=self.device)

        heights = self.measured_heights[:, self.trot_plane_height_indices]
        heights = heights - heights.mean(dim=1, keepdim=True)
        slope_x = torch.sum(heights * self.trot_plane_x.unsqueeze(0), dim=1) / self.trot_plane_x_denom
        slope_y = torch.sum(heights * self.trot_plane_y.unsqueeze(0), dim=1) / self.trot_plane_y_denom
        normal_yaw_frame = torch.stack((-slope_x, -slope_y, torch.ones_like(slope_x)), dim=1)
        normal_yaw_frame = normal_yaw_frame / torch.norm(
            normal_yaw_frame, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        terrain_normal_world = quat_apply_yaw(self.base_quat, normal_yaw_frame)

        body_up = torch.zeros(self.num_envs, 3, device=self.device)
        body_up[:, 2] = 1.0
        body_up_world = quat_apply(self.base_quat, body_up)
        alignment = torch.sum(body_up_world * terrain_normal_world, dim=1).clamp(-1.0, 1.0)
        return 1.0 - alignment
