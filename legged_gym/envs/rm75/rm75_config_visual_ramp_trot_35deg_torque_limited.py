"""Visual RM75 flat/ramp trot with 100/160 N m hard torque limits."""

from legged_gym.envs.rm75.rm75_config_visual_ramp_trot_35deg import (
    RM75VisualRampTrot35DegCfg,
    RM75VisualRampTrot35DegCfgPPO,
)


class RM75TorqueLimitedVisualRampTrot35DegCfg(RM75VisualRampTrot35DegCfg):
    """Adapt the successful limited flat trot to continuous 35 degree ramps."""

    class control(RM75VisualRampTrot35DegCfg.control):
        # hip includes ab/adduction, thigh is hip pitch, and calf is knee pitch.
        joint_torque_limits = {
            "hip": 100.0,
            "thigh": 100.0,
            "calf": 160.0,
        }

    class domain_rand(RM75VisualRampTrot35DegCfg.domain_rand):
        # Motor strength is applied after controller clipping. Keeping the
        # multiplier <= 1 guarantees that randomized rollouts also respect the
        # requested physical torque ceilings.
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.0]

    class terrain(RM75VisualRampTrot35DegCfg.terrain):
        # Add an intermediate level and delay the steeper ramps. The final
        # 35-degree distribution is still active for the last 8000 updates.
        level_curriculum = [
            {"iter": 0, "max_level": 2},
            {"iter": 1500, "max_level": 3},
            {"iter": 3500, "max_level": 4},
            {"iter": 6500, "max_level": 6},
            {"iter": 10500, "max_level": 7},
            {"iter": 16000, "max_level": 8},
            {"iter": 22000, "max_level": 9},
        ]
        random_section_spawn_start_iter = 1500
        random_section_spawn_probability = 0.50

    class rewards(RM75VisualRampTrot35DegCfg.rewards):
        # Under a hard torque ceiling, learn a lower-speed support pattern
        # before demanding the final 1.5 m/s on ramps.
        slope_speed_curriculum = [
            {
                "start_iter": 0,
                "end_iter": 1500,
                "start_speed": 0.9,
                "end_speed": 1.1,
            },
            {
                "start_iter": 1500,
                "end_iter": 6000,
                "start_speed": 1.1,
                "end_speed": 1.35,
            },
            {
                "start_iter": 6000,
                "end_iter": 10000,
                "start_speed": 1.35,
                "end_speed": 1.5,
            },
        ]

        # Longer support phases share the required slope force over more feet
        # and reduce dependence on short saturated torque bursts.
        trot_uphill_contact_duty = 0.62
        trot_downhill_contact_duty = 0.56

        # The hard clip remains 100/160 N m. Move the soft warning threshold
        # from 90% to 95% so the policy may use the available actuator envelope
        # on steep ramps without being rewarded for sustained saturation.
        soft_torque_limit = 0.95

        curriculum_rewards = [
            {
                "reward_name": "tracking_lin_vel",
                "start_iter": 0,
                "end_iter": 4000,
                "start_value": 0.80,
                "end_value": 1.0,
            },
            {
                "reward_name": "trot_gait",
                "start_iter": 0,
                "end_iter": 6000,
                "start_value": 0.60,
                "end_value": 1.0,
            },
            {
                "reward_name": "feet_slip",
                "start_iter": 0,
                "end_iter": 6000,
                "start_value": 0.60,
                "end_value": 1.0,
            },
        ]

        class scales(RM75VisualRampTrot35DegCfg.rewards.scales):
            # Preserve the same reward definitions as the proven visual task,
            # with small changes for torque-constrained climbing stability.
            speed_tracking_error = -0.65
            downhill_overspeed = -1.20
            torques = -1.5e-6
            dof_power = -7.5e-7
            torque_limits = -0.006
            action_rate = -0.012
            action_smoothness = -0.006


class RM75TorqueLimitedVisualRampTrot35DegCfgPPO(RM75VisualRampTrot35DegCfgPPO):
    class policy(RM75VisualRampTrot35DegCfgPPO.policy):
        # Let the zero-initialized visual residual adapt first and keep the
        # successful limited flat actor conservative when it is unfrozen.
        visual_residual_freeze_iterations = 250
        base_actor_freeze_iterations = 1500
        base_actor_lr_scale = 0.20
        init_noise_std = 0.35

    class algorithm(RM75VisualRampTrot35DegCfgPPO.algorithm):
        learning_rate = 1.5e-4
        entropy_coef = 0.003
        schedule = "fixed"

    class runner(RM75VisualRampTrot35DegCfgPPO.runner):
        experiment_name = "RM75_visual_ramp_trot_3ms_35deg_hip100_knee160"
        run_name = "warmstart_torque_model30000"
        max_iterations = 30000
        save_interval = 100
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        warm_start_path = (
            "{LEGGED_GYM_ROOT_DIR}/logs/RM75_flat_trot_3ms_hip100_knee160/"
            "Aug27_14-56-04_warmstart_earth_model25000/model_30000.pt"
        )
        warm_start_max_action_std = 0.35
        warm_start_critic_common_dim = 76
