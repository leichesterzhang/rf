"""Earth-gravity RM75 3 m/s trot with 100/160 N m hard torque limits."""

from legged_gym.envs.rm75.rm75_config_flat_trot import (
    RM75FlatTrotCfg,
    RM75FlatTrotCfgPPO,
)


class RM75TorqueLimitedFlatTrotCfg(RM75FlatTrotCfg):
    class control(RM75FlatTrotCfg.control):
        # Four hip-abduction and four hip-pitch motors are limited to 100 N m;
        # the four knee (calf) motors are limited to 160 N m.
        joint_torque_limits = {
            "hip": 100.0,
            "thigh": 100.0,
            "calf": 160.0,
        }

    class domain_rand(RM75FlatTrotCfg.domain_rand):
        # Motor strength is applied after the controller's hard clipping in the
        # common environment.  Restricting the multiplier to <= 1.0 guarantees
        # the final force sent to PhysX cannot exceed the requested hard limits.
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.0]

    class commands(RM75FlatTrotCfg.commands):
        # Re-adapt the transferred gait at moderate speed before asking the
        # torque-limited robot to maintain the final 3 m/s command.
        command_range_curriculum = [
            {
                "iter": 1000,
                "lin_vel_x": [1.0, 1.5],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 3000,
                "lin_vel_x": [1.3, 1.8],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 5000,
                "lin_vel_x": [1.7, 2.2],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 7500,
                "lin_vel_x": [2.1, 2.6],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 10000,
                "lin_vel_x": [2.5, 2.9],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 13000,
                "lin_vel_x": [2.8, 3.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 15000,
                "lin_vel_x": [3.0, 3.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            },
        ]

        class ranges(RM75FlatTrotCfg.commands.ranges):
            lin_vel_x = [0.8, 1.2]
            new_lin_vel_x = [0.8, 1.2]

    class rewards(RM75FlatTrotCfg.rewards):
        curriculum_rewards = [
            {
                "reward_name": "tracking_lin_vel",
                "start_iter": 0,
                "end_iter": 3000,
                "start_value": 0.75,
                "end_value": 1.0,
            },
            {
                "reward_name": "trot_gait",
                "start_iter": 0,
                "end_iter": 5000,
                "start_value": 0.50,
                "end_value": 1.0,
            },
            {
                "reward_name": "feet_slip",
                "start_iter": 0,
                "end_iter": 5000,
                "start_value": 0.50,
                "end_value": 1.0,
            },
        ]


class RM75TorqueLimitedFlatTrotCfgPPO(RM75FlatTrotCfgPPO):
    class algorithm(RM75FlatTrotCfgPPO.algorithm):
        learning_rate = 2.0e-4
        schedule = "fixed"
        entropy_coef = 0.003

    class runner(RM75FlatTrotCfgPPO.runner):
        experiment_name = "RM75_flat_trot_3ms_hip100_knee160"
        run_name = "warmstart_earth_model25000"
        max_iterations = 30000
        save_interval = 100
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        warm_start_path = (
            "{LEGGED_GYM_ROOT_DIR}/logs/RM75_flat_trot_3ms/"
            "Aug25_16-23-22_cold_start/model_25000.pt"
        )
        warm_start_max_action_std = 0.35
