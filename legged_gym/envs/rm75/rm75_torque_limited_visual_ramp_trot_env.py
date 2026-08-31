"""Torque-limited RM75 visual continuous-ramp environment."""

from collections import Counter

from legged_gym.envs.rm75.rm75_visual_ramp_trot_env import (
    RM75VisualRampTrotRobot,
)


class RM75TorqueLimitedVisualRampTrotRobot(RM75VisualRampTrotRobot):
    """Apply task-local 100/160 N m limits without modifying the RM75 URDF."""

    _EXPECTED_JOINTS_PER_GROUP = 4

    def _process_dof_props(self, props, env_id):
        configured_limits = self.cfg.control.joint_torque_limits
        group_counts = Counter()

        for dof_index, dof_name in enumerate(self.dof_names):
            matching_groups = [
                group_name
                for group_name in configured_limits
                if dof_name.endswith(f"_{group_name}_joint")
            ]
            if len(matching_groups) != 1:
                raise RuntimeError(
                    "Every RM75 leg DOF must match exactly one configured torque "
                    f"group; dof={dof_name}, matches={matching_groups}"
                )
            group_name = matching_groups[0]
            torque_limit = float(configured_limits[group_name])
            if torque_limit <= 0.0:
                raise ValueError(
                    f"Torque limit for {group_name} must be positive, got {torque_limit}"
                )
            props["effort"][dof_index] = torque_limit
            group_counts[group_name] += 1

        expected_counts = {
            group_name: self._EXPECTED_JOINTS_PER_GROUP
            for group_name in configured_limits
        }
        if dict(group_counts) != expected_counts:
            raise RuntimeError(
                "Unexpected RM75 torque-limit joint mapping: "
                f"actual={dict(group_counts)}, expected={expected_counts}"
            )

        return super()._process_dof_props(props, env_id)
