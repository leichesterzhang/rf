import math

from legged_gym.envs.go2.go2_config_parkour_moe import GO2CfgParkourMoE, GO2ParkourCfg


class DOG502ParkourCfg(GO2ParkourCfg):
    """dog_502 parkour-MoE training config.

    The robot keeps the same 12-DOF joint names as Go2, so the Go2 parkour
    environment, observations, rewards, and MoE policy layout can be reused.
    Only robot-body-specific parameters are overridden here.
    """

    class init_state(GO2ParkourCfg.init_state):
        pos = [0.0, 0.0, 0.31]
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
            "backflip": [0.10, 0.14],
            "sideflip": [0.15, 0.20],
        }

    class asset(GO2ParkourCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/dog_502_description/urdf/dog_502_description.urdf"
        name = "dog_502"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        flip_visual_attachments = False

    class control(GO2ParkourCfg.control):
        stiffness = {"joint": 20.0}
        damping = {"joint": 0.5}
        action_scale = 0.25

    class camera(GO2ParkourCfg.camera):
        attach_body = "base"
        local_pos = [0.255, 0.0, 0.04185]
        local_euler_xyz = [math.pi, math.radians(65.0), math.radians(-90.0)]

    class rewards(GO2ParkourCfg.rewards):
        base_height_target = 0.315
        turn_over_success_base_height = 0.26
        turn_over_base_height_target = 0.31
        max_contact_force = 80.0


class DOG502CfgParkourMoE(GO2CfgParkourMoE):
    class runner(GO2CfgParkourMoE.runner):
        experiment_name = "dog_502_parkour_moe"
        run_name = ""

