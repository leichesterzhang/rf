import math

from legged_gym.envs.go2.go2_config_parkour_moe import (
    GO2CfgParkourMoE,
    GO2ParkourCfg,
    MGDP_FLAT_TERRAIN_IDS,
)


class RM75ParkourCfg(GO2ParkourCfg):
    """Parkour training configuration for the 80 kg RM75 with fixed Z1 arm."""

    class init_state(GO2ParkourCfg.init_state):
        # Z1_NOLIDAR.urdf keeps the complete arm rigidly attached at its URDF
        # zero pose, while only these 12 leg joints remain policy-controlled.
        # Start at the four-foot contact height.  The old 0.56 m start
        # dropped the heavy RM75 onto the front feet before the rear feet.
        pos = [0.0, 0.0, 0.53]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FL_calf_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "FR_calf_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.0,
            "RL_calf_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.0,
            "RR_calf_joint": 0.0,
        }
        # RM75 walking curriculum does not include self-righting by default.
        # The fixed arm is still present in the rigid-body dynamics.
        turn_over = False
        # Do not inject a random root velocity into the heavy robot at reset.
        # The common Go2 parkour reset uses +/-0.5 m/s and rad/s, which makes
        # the RM75 fall before its first stable support phase.
        zero_reset_root_velocities = True

    class domain_rand(GO2ParkourCfg.domain_rand):
        added_mass_range = [-4.0, 4.0]
        multiplied_link_mass_range = [0.95, 1.05]
        added_base_com_range = [-0.02, 0.02]
        stiffness_multiplier_range = [0.85, 1.15]
        damping_multiplier_range = [0.85, 1.15]
        motor_strength_range = [0.9, 1.1]
        max_push_vel_xy = 0.30
        max_push_ang_vel = 0.40

    class control(GO2ParkourCfg.control):
        # RM75 with the fixed arm is about 95.9 kg.  300/10 lets the body
        # sag and roll after contact; 500/18 keeps the baked zero pose near
        # the intended four-foot support height.
        stiffness = {"joint": 500.0}
        damping = {"joint": 18.0}
        action_scale = 0.20

    class terrain(GO2ParkourCfg.terrain):
        terrain_length = 12.0
        terrain_width = 5.0
        start_platform_length = 2.5
        start_platform_width = 2.4
        reset_outside_block_margin = 1.0
        reset_platform_x_jitter = 0.20
        reset_platform_y_jitter = 0.0
        edge_width_thresh = 0.08
        ramp_max_angle_deg = 25.0
        measured_points_x = [round(-0.60 + 0.125 * i, 3) for i in range(17)]
        measured_points_y = [round(-0.60 + 0.12 * i, 3) for i in range(11)]
        # [single_gap, step_stone, two_row_stones, one_row_stones,
        #  single_bridge, air_beams, air_stones, hurdle, ramp, corridor,
        #  stairs_up, flat, rough_flat]
        terrain_proportions = [
            0.13,
            0.00,
            0.05,
            0.10,
            0.08,
            0.08,
            0.05,
            0.13,
            0.08,
            0.05,
            0.09,
            0.08,
            0.08,
        ]

    class commands(GO2ParkourCfg.commands):
        turning_lin_vel_threshold = 0.25
        flat_speed_search_enabled = False
        terrain_max_command_ranges = [
            {
                "lin_vel_x": [0.0, 1.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [-0.8, 0.8],
                "heading": [0.0, 0.0],
            }
            for _ in range(13)
        ]

        class ranges(GO2ParkourCfg.commands.ranges):
            lin_vel_x = [0.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [0.0, 0.0]
            new_lin_vel_x = [0.0, 1.0]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [-0.8, 0.8]
            new_heading = [0.0, 0.0]

    class asset(GO2ParkourCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/RM75/urdf/Z1_NOLIDARV11_locked_arm.urdf"
        name = "RM75"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        flip_visual_attachments = False

    class camera(GO2ParkourCfg.camera):
        attach_body = "base_link"
        local_pos = [0.45, 0.0, 0.22]
        local_euler_xyz = [math.pi, math.radians(65.0), math.radians(-90.0)]

    class rewards(GO2ParkourCfg.rewards):
        base_height_target = 0.53
        turn_over_success_base_height = 0.46
        turn_over_base_height_target = 0.53
        max_contact_force = 420.0
        min_legs_distance = 0.15
        clearance_max_height = 0.80
        targeted_foothold_front_center_x = 0.35
        targeted_foothold_rear_center_x = 0.20
        targeted_foothold_lateral_center_y = 0.20
        targeted_foothold_front_lookahead = 0.15
        targeted_foothold_rear_lookahead = 0.08
        targeted_foothold_search_x_half = 0.30
        targeted_foothold_search_y_half = 0.30
        targeted_foothold_distance_sigma = 0.05
        flat_gait_min_contact_duty = 0.25
        flat_gait_min_rear_contact_duty = 0.45
        flat_gait_rear_balance_tolerance = 0.12
        flat_gait_contact_duty_weight = 0.75
        flat_gait_rear_duty_weight = 1.60
        flat_gait_rear_balance_weight = 0.80
        flat_speed_search_max_speed_reward = 1.5
        stand_still_terrain_ids = MGDP_FLAT_TERRAIN_IDS

        class scales(GO2ParkourCfg.rewards.scales):
            correct_base_height = -1.5
            torques = -2.0e-6
            dof_acc = -1.0e-8
            feet_regulation = -0.025
            hip_to_default = -0.03
            action_rate = -0.008
            action_smoothness = -0.008
            collision = -1.5
            flat_gait = -0.8


class RM75CfgParkourMoE(GO2CfgParkourMoE):
    class runner(GO2CfgParkourMoE.runner):
        experiment_name = "RM75_parkour_moe"
        run_name = ""
        max_iterations = 30000
