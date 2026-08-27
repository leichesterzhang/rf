"""Warm-start visual RM75 trot training on flat ground and continuous ramps."""

from legged_gym.envs.rm75.rm75_config_parkour_moe import (
    RM75CfgParkourMoE,
    RM75ParkourCfg,
)


class RM75VisualRampTrotCfg(RM75ParkourCfg):
    class init_state(RM75ParkourCfg.init_state):
        turn_over = False
        zero_reset_root_velocities = True

    class env(RM75ParkourCfg.env):
        num_envs = 2048
        num_observations = 45
        num_privileged_obs = 269
        episode_length_s = 12
        env_spacing = 4.0
        # Proxy depth is produced by the height-field ray caster and therefore
        # does not require one Isaac Gym graphics camera per environment.
        enable_camera_sensors = False

    class terrain(RM75ParkourCfg.terrain):
        mesh_type = "trimesh"
        terrain_style = "mgdp_parkour"
        curriculum = True
        selected = False
        measure_heights = True
        measure_heights_interval = 5
        reset_when_outside_block = True
        reset_outside_block_margin = 0.35
        terrain_length = 12.0
        terrain_width = 5.0
        terrain_spacing = 0.0
        num_rows = 10
        # More terrain columns keep thousands of isolated Isaac Gym actors from
        # sharing too few broadphase locations, which otherwise overflows the
        # GPU found/lost aggregate-pair buffer.
        num_cols = 20
        max_init_terrain_level = 2
        ramp_max_angle_deg = 25.0
        mgdp_max_difficulty = 0.9
        mgdp_add_roughness = False
        start_platform_length = 2.0
        start_platform_width = 2.4
        reset_platform_x_jitter = 0.15
        reset_platform_y_jitter = 0.0
        edge_width_thresh = 0.08

        # MGDP order: gap, step stone, two-row stones, one-row stones,
        # bridge, beams, air stones, hurdle, ramp, corridor, stairs,
        # flat, rough flat. Keep 70% continuous ramps and 30% pure planes.
        terrain_proportions = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.70,
            0.0,
            0.0,
            0.30,
            0.0,
        ]
        level_curriculum = [
            {"iter": 0, "max_level": 2},
            {"iter": 1000, "max_level": 4},
            {"iter": 5000, "max_level": 6},
            {"iter": 10000, "max_level": 7},
            {"iter": 17000, "max_level": 9},
        ]
        ramp_terrain_id = 8
        random_section_spawn_start_iter = 1000
        random_section_spawn_probability = 0.50
        # Uphill, upper platform, and downhill local-x intervals.
        random_section_spawn_ranges = [
            [2.20, 3.80],
            [4.20, 5.80],
            [6.20, 7.80],
        ]

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
        command_range_curriculum = []
        omni_enabled = False
        omni_blind = False
        flat_speed_search_enabled = False
        play_command = [3.0, 0.0, 0.0]
        terrain_max_command_ranges = [
            {
                "lin_vel_x": [3.0, 3.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "heading": [0.0, 0.0],
            }
            for _ in range(13)
        ]

        class ranges(RM75ParkourCfg.commands.ranges):
            lin_vel_x = [3.0, 3.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
            new_lin_vel_x = [3.0, 3.0]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [0.0, 0.0]
            new_heading = [0.0, 0.0]

    class camera(RM75ParkourCfg.camera):
        enabled = True
        source = "proxy"
        proxy_backend = "warp"
        update_interval = 5
        buffer_len = 1
        clipping_range = 2.0
        raw_height = 60
        raw_width = 106
        output_height = 58
        output_width = 87
        xyz_error = [-0.01, 0.01]
        yaw_error_deg = [-5.0, 5.0]
        noise_gaussian = 0.05
        noise_dropout = 0.0

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

        flat_target_speed = 3.0
        slope_target_speed = 1.5
        slope_lookahead_distance = 1.20
        slope_detect_start_deg = 1.5
        slope_detect_full_deg = 4.0
        # 0.98 at 50 Hz retains slope mode across the short upper platform.
        slope_mode_release = 0.98
        slope_speed_curriculum = [
            {
                "start_iter": 0,
                "end_iter": 1000,
                "start_speed": 1.0,
                "end_speed": 1.2,
            },
            {
                "start_iter": 1000,
                "end_iter": 5000,
                "start_speed": 1.2,
                "end_speed": 1.5,
            },
        ]

        trot_min_command_speed = 0.3
        trot_contact_force_threshold = 5.0
        trot_contact_ema_alpha = 0.99
        trot_flat_contact_duty = 0.50
        trot_uphill_contact_duty = 0.58
        trot_downhill_contact_duty = 0.54
        trot_contact_duty_sigma = 0.04
        trot_duty_balance_tolerance = 0.15

        heading_conditioned_terrain_ids = []
        tracking_lin_vel_full_reward_terrain_ids = []
        clearance_excluded_terrain_ids = []
        base_height_excluded_terrain_ids = []
        hip_to_default_excluded_terrain_ids = []
        target_posture_terrain_ids = []
        stand_still_terrain_ids = []
        targeted_foothold_terrain_ids = []
        no_progress_excluded_terrain_ids = []
        no_progress_min_terrain_level = 0
        no_progress_timeout_s = 3.0

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

        class scales:
            termination = -2.0
            tracking_lin_vel = 4.0
            speed_tracking_error = -0.50
            tracking_ang_vel = 1.0
            trot_gait = 1.5
            lin_vel_z = -1.5
            lateral_velocity = -0.40
            downhill_overspeed = -1.0
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
            hip_to_default = -0.02

    class termination(RM75ParkourCfg.termination):
        use_fall_height = False
        fall_terrain_ids = [8, 11]

    class sim(RM75ParkourCfg.sim):
        class physx(RM75ParkourCfg.sim.physx):
            max_gpu_contact_pairs = 2**24
            default_buffer_size_multiplier = 5


class RM75VisualRampTrotCfgPPO(RM75CfgParkourMoE):
    seed = 42

    class policy(RM75CfgParkourMoE.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        residual_hidden_dims = [256, 128]
        residual_scale = 1.0
        visual_residual_freeze_iterations = 250
        base_actor_freeze_iterations = 1000
        base_actor_lr_scale = 0.25
        init_noise_std = 0.45
        num_actor_obs_now = 45
        mcp_dim = 69

    class estimator(RM75CfgParkourMoE.estimator):
        expert_num = 3
        history_length = 10
        depth_frame_count = 1
        num_terrain_types = 13
        terrain_map_dim = 187
        feet_height_dim = 4
        token_dropout_min = 0.00
        token_dropout_max = 0.15
        use_active_token_sampler = True
        active_token_count = 16

    class algorithm(RM75CfgParkourMoE.algorithm):
        learning_rate = 2.0e-4
        estimator_learning_rate = 1.0e-3
        entropy_coef = 0.005
        schedule = "fixed"
        load_balance_coef = 0.0
        terrain_swav_weight_max = 0.02
        terrain_swav_warmup_start = 2000
        terrain_swav_warmup_steps = 8000

    class runner(RM75CfgParkourMoE.runner):
        policy_class_name = "ActorCriticVisualResidual"
        algorithm_class_name = "PPOParkourMoE"
        experiment_name = "RM75_visual_ramp_trot_3ms_25deg"
        run_name = "warmstart_model25000"
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
        warm_start_max_action_std = 0.45
        warm_start_critic_common_dim = 76
        # Vision is always available for this task; do not toggle it off on
        # ramp or flat environments during the initial visual curriculum.
        easy_terrain_names = []
        vision_toggle_interval = 20
        enable_timing = False
        timing_sync_cuda = False
