import math

from legged_gym.envs.go2.go2_config_parkour_moe import (
    GO2CfgParkourMoE,
    GO2ParkourCfg,
    MGDP_FLAT_TERRAIN_IDS,
)


class B2ParkourCfg(GO2ParkourCfg):
    """Unitree B2 parkour-MoE config.

    B2 keeps the same 12-DOF leg joint naming convention as Go2, so the Go2
    parkour environment, observation layout, terrain generator, and MoE policy
    structure can be reused. Robot-specific values below are sized for the
    larger and heavier B2 body.
    """

    class init_state(GO2ParkourCfg.init_state):
        pos = [0.0, 0.0, 0.58]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
        turn_over_init_heights = {
            "backflip": [0.26, 0.32],
            "sideflip": [0.36, 0.44],
        }

    class asset(GO2ParkourCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/b2_description/urdf/b2_isaacgym.urdf"
        name = "b2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base", "head"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        flip_visual_attachments = True

    class control(GO2ParkourCfg.control):
        stiffness = {"joint": 80.0}
        damping = {"joint": 2.0}
        action_scale = 0.25

    class terrain(GO2ParkourCfg.terrain):
        terrain_length = 12.0
        terrain_width = 5.0
        start_platform_length = 2.5
        start_platform_width = 2.4
        reset_outside_block_margin = 1.0
        reset_platform_x_jitter = 0.20
        reset_platform_y_jitter = 0.0
        measured_points_x = [round(-0.5 + 0.125 * i, 3) for i in range(17)]
        measured_points_y = [round(-0.625 + 0.125 * i, 3) for i in range(11)]

    class commands(GO2ParkourCfg.commands):
        turning_lin_vel_threshold = 0.25
        flat_speed_search_enabled = False

        class ranges(GO2ParkourCfg.commands.ranges):
            lin_vel_x = [0.0, 1.2]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [0.0, 0.0]
            new_lin_vel_x = [0.0, 1.2]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [-0.8, 0.8]
            new_heading = [0.0, 0.0]

    class camera(GO2ParkourCfg.camera):
        attach_body = "base_link"
        local_pos = [0.42, 0.0, 0.10]
        local_euler_xyz = [math.pi, math.radians(65.0), math.radians(-90.0)]

    class rewards(GO2ParkourCfg.rewards):
        base_height_target = 0.55
        turn_over_success_base_height = 0.46
        turn_over_base_height_target = 0.55
        max_contact_force = 260.0
        clearance_max_height = 0.9
        stand_still_terrain_ids = MGDP_FLAT_TERRAIN_IDS

        class scales(GO2ParkourCfg.rewards.scales):
            torques = -0.000003
            dof_acc = -1.0e-8


class B2CfgParkourMoE(GO2CfgParkourMoE):
    class runner(GO2CfgParkourMoE.runner):
        experiment_name = "b2_parkour_moe"
        run_name = ""
        max_iterations = 30000
