"""Cold-start RM75 training configuration for a 3 m/s flat-ground trot."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.rm75.rm75_config_parkour_moe import RM75ParkourCfg


class RM75FlatTrotCfg(RM75ParkourCfg):
    class init_state(RM75ParkourCfg.init_state):
        turn_over = False
        zero_reset_root_velocities = True

    class env(RM75ParkourCfg.env):
        num_envs = 4096
        num_observations = 45
        # Actor observations (45) plus base velocity, foot forces, motor states,
        # and one base-height value for the asymmetric critic.
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 1
        episode_length_s = 20
        env_spacing = 4.0
        enable_camera_sensors = False

    class terrain(RM75ParkourCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        selected = False
        measure_heights = False
        reset_when_outside_block = False

    class commands(RM75ParkourCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 5.0
        heading_command = False
        zero_command = False
        zero_command_curriculum = None
        limit_ang_vel_at_zero_command_prob = 0.0
        limit_vel_prob = 0.0
        dynamic_resample_commands = False

        # One cold-start run: first learn stable low-speed locomotion, then
        # progressively narrow the task to the requested constant 3 m/s.
        command_range_curriculum = [
            {
                "iter": 1500,
                "lin_vel_x": [0.8, 1.3],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 3500,
                "lin_vel_x": [1.2, 1.7],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 6000,
                "lin_vel_x": [1.6, 2.1],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 9000,
                "lin_vel_x": [2.0, 2.5],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 12000,
                "lin_vel_x": [2.4, 2.8],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 15000,
                "lin_vel_x": [2.7, 3.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 18000,
                "lin_vel_x": [3.0, 3.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
        ]
        play_command = [3.0, 0.0, 0.0]

        class ranges(RM75ParkourCfg.commands.ranges):
            lin_vel_x = [0.5, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
            new_lin_vel_x = [0.5, 1.0]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [0.0, 0.0]
            new_heading = [0.0, 0.0]

    class camera(RM75ParkourCfg.camera):
        enabled = False

    class domain_rand(RM75ParkourCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.4]
        randomize_restitution = True
        restitution_range = [0.0, 0.2]
        randomize_base_mass = True
        added_mass_range = [-4.0, 4.0]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.95, 1.05]
        randomize_base_com = True
        added_base_com_range = [-0.02, 0.02]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.02, 0.02]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_action_delay = True
        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 0.25
        max_push_ang_vel = 0.30

    class rewards(RM75ParkourCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.20
        dynamic_sigma = None
        base_height_target = 0.53
        max_contact_force = 650.0
        soft_dof_pos_limit = 0.90
        soft_torque_limit = 0.90
        fall_contact_force_threshold = 1.0

        trot_min_command_speed = 0.3
        trot_contact_force_threshold = 5.0
        trot_contact_ema_alpha = 0.99
        trot_target_contact_duty = 0.50
        trot_contact_duty_sigma = 0.04
        trot_duty_balance_tolerance = 0.15

        curriculum_rewards = [
            {
                "reward_name": "tracking_lin_vel",
                "start_iter": 0,
                "end_iter": 6000,
                "start_value": 0.50,
                "end_value": 1.0,
            },
            {
                "reward_name": "trot_gait",
                "start_iter": 0,
                "end_iter": 8000,
                "start_value": 0.25,
                "end_value": 1.0,
            },
            {
                "reward_name": "feet_slip",
                "start_iter": 0,
                "end_iter": 8000,
                "start_value": 0.25,
                "end_value": 1.0,
            },
        ]

        class scales:
            termination = -2.0
            tracking_lin_vel = 4.0
            tracking_ang_vel = 1.0
            trot_gait = 1.5
            lin_vel_z = -1.5
            ang_vel_xy = -0.10
            orientation = -1.0
            correct_base_height = -2.0
            torques = -2.0e-6
            dof_acc = -1.0e-8
            dof_power = -1.0e-6
            action_rate = -0.010
            action_smoothness = -0.005
            collision = -2.0
            dof_pos_limits = -3.0
            torque_limits = -0.01
            feet_contact_forces = -0.002
            feet_slip = -0.10
            feet_regulation = -0.02
            hip_to_default = -0.02


class RM75FlatTrotCfgPPO(LeggedRobotCfgPPO):
    seed = 42

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 5.0e-4
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "RM75_flat_trot_3ms"
        run_name = "cold_start"
        max_iterations = 30000
        save_interval = 100
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
