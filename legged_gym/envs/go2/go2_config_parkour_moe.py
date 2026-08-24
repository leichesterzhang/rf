import math

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgParkourMoE
from legged_gym.envs.go2.go2_config_vanilla import GO2Cfg


MGDP_NUM_TERRAIN_TYPES = 13
MGDP_FLAT_TERRAIN_ID = 11
MGDP_ROUGH_FLAT_TERRAIN_ID = 12
MGDP_FLAT_TERRAIN_IDS = [MGDP_FLAT_TERRAIN_ID, MGDP_ROUGH_FLAT_TERRAIN_ID]


class GO2ParkourCfg(GO2Cfg):
    class init_state(GO2Cfg.init_state):
        turn_over = True
        turn_over_terrain_ids = [MGDP_FLAT_TERRAIN_ID]
        turn_over_proportions = [0.4, 0.4, 0.2] # proportions for backflip, sideflip, noflip
        turn_over_init_heights = { # initial heights range for each flip type
            'backflip': [0.20, 0.21],
            'sideflip': [0.34, 0.35],
        }
        turn_over_roll_ranges = {
            'backflip': [math.pi / 2, math.pi],
            'sideflip': [0.0, math.pi / 2],
        }

    class env(GO2Cfg.env):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 269
        enable_camera_sensors = False
    
    class control(GO2Cfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain(GO2Cfg.terrain):
        terrain_style = "mgdp_parkour"
        measure_heights = True
        measure_heights_interval = 5
        reset_when_outside_block = True
        reset_outside_block_margin = 0.8
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 5
        terrain_length = 10.0
        terrain_width = 4.0
        terrain_spacing = 0.0
        num_rows = 10
        num_cols = 45
        edge_width_thresh = 0.05

        max_init_terrain_level = 2
        # MGDP stage-2 active terrain columns:
        # [single_gap, step_stone, two_row_stones, one_row_stones, single_bridge,
        #  air_beams, air_stones, hurdle, ramp, corridor, stairs_up, flat, rough_flat]
        terrain_proportions = [0.15, 0.0, 0.05, 0.1, 0.1, 0.1, 0.05, 0.15, 0.05, 0.05, 0.1, 0.05, 0.05]
        #terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        mgdp_add_roughness = False
        add_air_beam = True
        add_air_stone = True
        mgdp_air_stone_x_shift = 0.0
        mgdp_roughness_height = [0.01, 0.04]
        mgdp_roughness_downsampled_scale = 0.5
        mgdp_rough_flat_spike_height = [0.015, 0.055]
        mgdp_rough_flat_spike_density = [5.0, 18.0]
        mgdp_rough_flat_spike_radius = [0.025, 0.075]
        mgdp_depth = 0.6
        mgdp_gap_width_range = [0.1, 1.0]
        ramp_max_angle_deg = 30.0
        one_row_stone_gap_range = [0.1, 0.35]
        one_row_stone_x_size = 0.1  # x full length at max difficulty; centers stay fixed across difficulty
        start_platform_length = 2.0
        start_platform_width = 2.0
        reset_platform_x_jitter = 0.15
        reset_platform_y_jitter = 0.0
        measured_points_x = [round(-0.4 + 0.1 * i, 2) for i in range(17)]
        measured_points_y = [round(-0.5 + 0.1 * i, 2) for i in range(11)]
        move_down_by_accumulated_xy_command = False


    class commands(GO2Cfg.commands):
        heading_command = True
        zero_command = True
        zero_command_threshold = 0.2
        easy_command_terrain_ids = [6, 8, 10, MGDP_FLAT_TERRAIN_ID, MGDP_ROUGH_FLAT_TERRAIN_ID]
        omni_enabled = True
        omni_terrain_ids = [8, 10, MGDP_FLAT_TERRAIN_ID, MGDP_ROUGH_FLAT_TERRAIN_ID]  # air_stones暂时去掉, ramp, stairs_up, flat, rough_flat
        omni_ratio = 0.5
        omni_blind = True
        omni_random_yaw = True
        omni_yaw_range = [-math.pi, math.pi]
        omni_stairs_discrete_yaw = False
        omni_stairs_terrain_id = 10
        omni_stairs_yaw_values = [0.0, math.pi, math.pi / 2, -math.pi / 2]
        omni_stairs_max_speed = 0.6
        omni_disable_forward_rewards = False
        turning_lin_vel_threshold = 0.3
        turn_in_place_ang_vel_yaw = [-1.2, 1.2]
        ang_vel_sampling_terrain_ids = []
        flat_speed_search_enabled = True
        flat_speed_search_terrain_ids = MGDP_FLAT_TERRAIN_IDS
        flat_speed_search_disable_omni = True
        terrain_max_command_ranges = [
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # single_gap
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # step_stone
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # two_row_stones
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # one_row_stones
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # single_bridge
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # air_beams
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # air_stones
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # hurdle
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # ramp
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # corridor
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # stairs_up
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # flat
            {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [0.0, 0.0], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [0.0, 0.0]},  # rough_flat
        ]
        turn_over_zero_time = { # if turn_over is true, time robot must be stable before sampling new commands after a turn over
            "backflip": 5.0,
            "sideflip": 3.0,
        }

        class ranges:
            lin_vel_x = [0.0, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]
            new_lin_vel_x = [0.0, 1.5]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [-1.0, 1.0]
            new_heading = [0.0, 0.0]

    class camera(GO2Cfg.camera):
        enabled = True
        source = "proxy"
        proxy_backend = "warp"  # torch or warp
        update_interval = 5
        buffer_len = 1
        convention = "ros"
        focal_length = 11.041
        horizontal_aperture = 20.955
        vertical_aperture = 12.240
        raw_height = 60
        raw_width = 106
        crop_top = 0
        crop_bottom = 2
        crop_left = 4
        crop_right = 4
        clipping_range = 2.0
        attach_body = "base"
        local_pos = [0.33, 0.0, 0.08]
        local_euler_xyz = [math.pi, math.radians(65.0), math.radians(-90.0)]
        xyz_error = [-0.01, 0.01]
        yaw_error_deg = [-5.0, 5.0]
        noise_gaussian = 0.05 # 高斯噪声
        noise_dropout = 0.0  # 坏点
        raycast_chunk_size = 128
        warp_raycast_chunk_size = 512
        raycast_num_coarse_steps = 8
        raycast_num_refine_steps = 4

    class parkour:
        obs_now_dim = 45
        feet_height_dim = 4
        height_map_dim = 187
        history_length = 10

    class termination:
        use_fall_height = True
        fall_base_height = -0.1
        fall_foot_height = -0.1
        fall_foot_height_overrides = {
            0: -0.3,  # single_gap
        }
        fall_terrain_ids = list(range(MGDP_NUM_TERRAIN_TYPES))

    class rewards(GO2Cfg.rewards):
        low_speed_cmd_threshold = 0.2
        low_speed_min_xy_speed = 0.25
        no_progress_cmd_threshold = 0.2
        no_progress_excluded_terrain_ids = MGDP_FLAT_TERRAIN_IDS
        no_progress_min_terrain_level = 0
        no_progress_timeout_s = 2.0
        omni_no_progress_timeout_s = 8.0
        no_progress_min_delta = 0.1
        turn_over_success_base_height = 0.3 # 翻身base判断条件
        turn_over_base_height_target = 0.38 # 翻身base奖励
        # omni 下允许小幅摆身恢复，超过 deadzone 后才惩罚偏离出生 yaw
        omni_target_yaw_deadzone = 0.25
        heading_conditioned_terrain_ids = list(range(MGDP_NUM_TERRAIN_TYPES))
        # 线速度最大值放开地形
        tracking_lin_vel_full_reward_terrain_ids = [0, 7]  # single_gap, hurdle
        heading_conditioned_cmd_threshold = 0.0
        clearance_lookahead_distance = 0.06
        omni_stairs_clearance_lookahead_distance = 0.08
        clearance_active_step_height = 0.02
        clearance_base_height = 0.03
        clearance_rise_gain = 1.0
        clearance_max_height = 0.7
        clearance_min_foot_speed = 0.1
        clearance_min_command_x = 0.2
        clearance_min_base_vel = 0.08
        clearance_min_base_vel_ratio = 0.2
        clearance_max_air_time = 1.0
        clearance_overtime_penalty = 1.25
        clearance_tracking_sigma = 0.04
        # 抬脚奖励排除： single_gap, air_stones, corridor, flat
        clearance_excluded_terrain_ids = [0, 6, 9]  
        # base高度奖励排除：single_gap, two_row_stones, one_row_stones, air_beams, air_stones， hurdle
        base_height_excluded_terrain_ids = [0, 2, 3, 5, 6, 7]  
        # single_bridge 上不计算 hip_to_default
        hip_to_default_excluded_terrain_ids = [4]
        # target_posture 只在两类平地上约束，避免限制障碍地形的关节姿态
        target_posture_terrain_ids = MGDP_FLAT_TERRAIN_IDS
        # 落脚点奖励用于需要主动选点的地形以及两类平地
        targeted_foothold_terrain_ids = [1, 2, 3, 5, 7, 8, 10, MGDP_FLAT_TERRAIN_ID, MGDP_ROUGH_FLAT_TERRAIN_ID]  
        targeted_foothold_cmd_threshold = 0.2
        targeted_foothold_front_center_x = 0.25
        targeted_foothold_rear_center_x = 0.15
        targeted_foothold_lateral_center_y = 0.12
        targeted_foothold_front_lookahead = 0.12
        targeted_foothold_rear_lookahead = 0.05
        targeted_foothold_search_x_half = 0.22
        targeted_foothold_search_y_half = 0.22
        targeted_foothold_height_gain = 1.0
        targeted_foothold_anchor_penalty = 6.0
        targeted_foothold_edge_penalty = 1.0
        targeted_foothold_distance_sigma = 0.03
        targeted_foothold_min_air_time = 0.12
        targeted_foothold_min_target_z = -0.1
        targeted_foothold_retry_step_x = 0.25
        single_bridge_terrain_id = 4
        single_bridge_narrow_stance_cmd_threshold = 0.2
        single_bridge_narrow_stance_lateral_target = 0.06
        single_bridge_narrow_stance_lateral_tolerance = 0.01
        single_bridge_narrow_stance_sigma = 0.0025
        flat_gait_terrain_ids = MGDP_FLAT_TERRAIN_IDS
        flat_gait_cmd_threshold = 0.2
        flat_gait_min_stance_time = 0.10
        flat_gait_contact_ema_alpha = 0.98
        flat_gait_min_contact_duty = 0.20
        flat_gait_min_rear_contact_duty = 0.30
        flat_gait_rear_balance_tolerance = 0.20
        flat_gait_early_liftoff_weight = 1.0
        flat_gait_contact_duty_weight = 0.5
        flat_gait_rear_duty_weight = 1.0
        flat_gait_rear_balance_weight = 0.5
        stairs_omni_terrain_id = 10
        stairs_omni_cmd_threshold = 0.2
        stairs_omni_height_probe_distance = 0.08
        stairs_omni_touch_height = 0.03
        stairs_omni_followup_window = 0.55
        stairs_omni_support_progress_step = 0.04
        omni_stairs_clearance_requires_progress = False
        omni_stairs_clearance_min_foot_speed = 0.04
        flat_speed_search_max_speed_reward = 2.5

        class scales(GO2Cfg.rewards.scales):
            #feet_air_time = 0.0
            stand_still = -0.5
            #target_posture = -0.01
            low_speed_when_commanded = -0.5
            # feet_stumble = -1.0
            # feet_edge = -1.0
            hard_terrain_yaw_penalty = -0.2
            omni_target_yaw_error = -0.05
            upward_foothold_clearance = 0.5
            targeted_foothold_touchdown = 0.5
            single_bridge_narrow_stance = 0.5
            flat_forward_speed = 1.0
            #flat_gait = -0.5
            stairs_omni_lead_touch_above = 0.6
            stairs_omni_trail_followup = 1.0
            stairs_omni_support_transfer_progress = 0.8
            #collision = -10.0

        class turn_over_scales(GO2Cfg.rewards.turn_over_scales):
            upright = 1.0
            turn_over_base_height = -1.0
            #orientation = -1.0
            #target_posture = -0.05


class GO2CfgParkourMoE(LeggedRobotCfgParkourMoE):
    class policy(LeggedRobotCfgParkourMoE.policy):
        mcp_dim = 69

    class estimator(LeggedRobotCfgParkourMoE.estimator):
        expert_num = 3
        history_length = 10
        depth_frame_count = 1
        num_terrain_types = MGDP_NUM_TERRAIN_TYPES
        transformer_num_layers = 1
        transformer_nhead = 4
        local_patch_size = 15
        local_patch_stride = 8

        token_dropout_min = 0.00
        token_dropout_max = 0.15
        token_dropout_mode = "spatial_block"
        token_dropout_block_min_size = [1, 2]
        token_dropout_block_max_size = [2, 3]
        token_dropout_block_min_count = 1
        token_dropout_block_max_count = 3
        token_dropout_uniform_weight = 0.10
        token_dropout_ground_weight = 3.5
        token_dropout_far_weight = 2.5
        token_dropout_center_weight = 0.2
        token_dropout_center_sigma = 0.30
        token_dropout_ground_row_start = 0.50
        token_dropout_far_row_end = 0.30
        
        use_active_token_sampler = True
        active_token_count = 16
        active_token_noise_std = 0.02
        active_token_min_valid_mass = 0.05

    class runner(LeggedRobotCfgParkourMoE.runner):
        experiment_name = "go2_parkour_moe"
        run_name = ""
        max_iterations = 30000
        save_interval = 100
        easy_terrain_names = ["ramp", "stairs_up", "air_stones", "flat", "rough_flat"]
        enable_timing = False
        timing_sync_cuda = False
