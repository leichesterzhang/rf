"""RM75 flat-ground trot task environment."""

from isaacgym import gymtorch
import torch

from legged_gym.envs.go2.go2_env import Go2Robot


class RM75FlatTrotRobot(Go2Robot):
    """Adds an explicit diagonal-trot objective to the regular locomotion task."""

    _TROT_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    def _init_buffers(self):
        super()._init_buffers()

        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        missing_names = [name for name in self._TROT_FOOT_NAMES if name not in body_names]
        if missing_names:
            raise RuntimeError(
                "RM75 trot reward requires the four named feet; "
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
            float(self.cfg.rewards.trot_target_contact_duty),
            dtype=torch.float,
            device=self.device,
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.trot_last_contacts[env_ids] = False
        self.trot_contact_ema[env_ids] = float(
            self.cfg.rewards.trot_target_contact_duty
        )

    def _reset_root_states(self, env_ids):
        """Keep cold-start resets stable instead of injecting a random root impulse."""
        super()._reset_root_states(env_ids)
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def check_termination(self):
        """Terminate every base contact; this task does not include self-righting."""
        super().check_termination()
        if len(self.base_termination_contact_indices) == 0:
            return
        contact_threshold = float(self.cfg.rewards.fall_contact_force_threshold)
        base_contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.base_termination_contact_indices, :],
                dim=-1,
            )
            > contact_threshold,
            dim=1,
        )
        self.reset_buf |= base_contact
        self._record_termination_reason("base_contact", base_contact)

    def _reward_trot_gait(self):
        """Reward alternating diagonal support with balanced per-foot duty cycles.

        Foot order is FL, FR, RL, RR. A trot support phase therefore has either
        FL+RR or FR+RL in contact. The EMA term prevents the policy from keeping
        only one diagonal pair in stance indefinitely.
        """
        contact_threshold = float(self.cfg.rewards.trot_contact_force_threshold)
        contacts = self.contact_forces[:, self.trot_feet_indices, 2] > contact_threshold
        contact_filt = torch.logical_or(contacts, self.trot_last_contacts)
        self.trot_last_contacts[:] = contacts

        alpha = float(self.cfg.rewards.trot_contact_ema_alpha)
        self.trot_contact_ema.mul_(alpha).add_(contact_filt.float(), alpha=1.0 - alpha)

        contact = contact_filt.float()
        diagonal_a = 0.5 * (contact[:, 0] + contact[:, 3])  # FL + RR
        diagonal_b = 0.5 * (contact[:, 1] + contact[:, 2])  # FR + RL

        diagonal_sync_error = torch.abs(contact[:, 0] - contact[:, 3])
        diagonal_sync_error += torch.abs(contact[:, 1] - contact[:, 2])
        diagonal_sync_score = (1.0 - 0.5 * diagonal_sync_error).clamp(0.0, 1.0)
        diagonal_opposition_score = torch.abs(diagonal_a - diagonal_b)

        mean_duty = self.trot_contact_ema.mean(dim=1, keepdim=True)
        duty_spread = torch.abs(self.trot_contact_ema - mean_duty).mean(dim=1)
        balance_tolerance = float(self.cfg.rewards.trot_duty_balance_tolerance)
        duty_balance_score = (1.0 - duty_spread / balance_tolerance).clamp(0.0, 1.0)

        target_duty = float(self.cfg.rewards.trot_target_contact_duty)
        duty_sigma = float(self.cfg.rewards.trot_contact_duty_sigma)
        target_duty_score = torch.exp(
            -torch.square(mean_duty.squeeze(1) - target_duty) / duty_sigma
        )

        active_command = self.commands[:, 0] > float(
            self.cfg.rewards.trot_min_command_speed
        )
        return (
            diagonal_sync_score
            * diagonal_opposition_score
            * duty_balance_score
            * target_duty_score
            * active_command.float()
        )

    def _reward_feet_slip(self):
        """Penalize horizontal foot motion while a foot is supporting the body."""
        contact_threshold = float(self.cfg.rewards.trot_contact_force_threshold)
        contacts = self.contact_forces[:, self.trot_feet_indices, 2] > contact_threshold
        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        feet_xy_velocity = body_states[:, self.trot_feet_indices, 7:9]
        return torch.sum(
            torch.sum(torch.square(feet_xy_velocity), dim=2) * contacts.float(),
            dim=1,
        )

    def _reward_trot_duty_balance(self):
        """Penalize each foot's duty error relative to the configured target.

        This is intentionally separate from the multiplicative trot score.  A
        policy can otherwise keep a clean diagonal rhythm while consistently
        shortening one diagonal pair, which produces the lateral wobble seen in
        the hip84 policy.
        """
        target_duty = float(self.cfg.rewards.trot_target_contact_duty)
        return torch.mean(
            torch.square(self.trot_contact_ema - target_duty), dim=1
        )
