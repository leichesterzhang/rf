"""Staged Earth-gravity torque-limit training for the RM75 V1.1 model."""

from legged_gym.envs.rm75.rm75_config_torque_limited_flat_trot import (
    RM75TorqueLimitedFlatTrotCfg,
    RM75TorqueLimitedFlatTrotCfgPPO,
)
from legged_gym.envs.rm75.rm75_config_visual_ramp_trot_35deg_torque_limited import (
    RM75TorqueLimitedVisualRampTrot35DegCfg,
    RM75TorqueLimitedVisualRampTrot35DegCfgPPO,
)


SOURCE_FLAT_CHECKPOINT = (
    "{LEGGED_GYM_ROOT_DIR}/logs/RM75_newmodel_source/"
    "hip100_knee160/model_30000.pt"
)
HIP100_RECOVERY_CHECKPOINT = (
    "{LEGGED_GYM_ROOT_DIR}/logs/RM75_newmodel_earth_flat_hip100_knee150/"
    "Sep02_16-28-21_stage_a_flat_adaptation_v2_zero_calibrated/model_10000.pt"
)
HIP84_RECOVERY_CHECKPOINT = (
    "{LEGGED_GYM_ROOT_DIR}/logs/RM75_newmodel_earth_flat_hip84_knee150/"
    "Sep02_16-28-21_stage_a_flat_adaptation_v2_zero_calibrated/model_10000.pt"
)
HIP84_V4_BEST_CHECKPOINT = (
    "{LEGGED_GYM_ROOT_DIR}/logs/RM75_newmodel_earth_flat_hip84_knee150/"
    "Sep03_11-22-30_flat_trot_contact_heading_recovery_v4_continue/model_2750.pt"
)
HIP84_REFINE_V1_BEST_CHECKPOINT = (
    "{LEGGED_GYM_ROOT_DIR}/logs/RM75_newmodel_earth_flat_hip84_knee150_refine/"
    "Sep03_12-04-30_duty_balance_refine_v1/model_100.pt"
)

# The established RM75 policy was trained with these offsets baked into the
# legacy URDF joint origins. In the clean V1.1 export they are represented as
# actual joint coordinates, which preserves both the physical pose and the
# policy's zero-centered observations.
V11_DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.06,
    "FL_thigh_joint": 0.70,
    "FL_calf_joint": -1.20,
    "FR_hip_joint": -0.06,
    "FR_thigh_joint": 0.70,
    "FR_calf_joint": -1.20,
    "RL_hip_joint": 0.06,
    "RL_thigh_joint": 0.75,
    "RL_calf_joint": -1.175,
    "RR_hip_joint": -0.06,
    "RR_thigh_joint": 0.75,
    "RR_calf_joint": -1.175,
}


class RM75NewModelEarthFlatAdaptCfg(RM75TorqueLimitedFlatTrotCfg):
    """Adapt the established flat policy to the V1.1 mass distribution."""

    class init_state(RM75TorqueLimitedFlatTrotCfg.init_state):
        default_joint_angles = dict(V11_DEFAULT_JOINT_ANGLES)

    class commands(RM75TorqueLimitedFlatTrotCfg.commands):
        # Hold an absolute zero heading.  A zero yaw-rate command alone lets a
        # small left/right policy bias integrate into metres of lateral drift.
        heading_command = True
        command_range_curriculum = [
            {
                "iter": 1000,
                "lin_vel_x": [2.8, 3.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [-1.0, 1.0],
                "heading": [0.0, 0.0],
            },
            {
                "iter": 2500,
                "lin_vel_x": [3.0, 3.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [-1.0, 1.0],
                "heading": [0.0, 0.0],
            },
        ]

        class ranges(RM75TorqueLimitedFlatTrotCfg.commands.ranges):
            lin_vel_x = [2.4, 2.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]
            new_lin_vel_x = [2.4, 2.8]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [-1.0, 1.0]
            new_heading = [0.0, 0.0]

    class domain_rand(RM75TorqueLimitedFlatTrotCfg.domain_rand):
        # Start with mild deployment variation while the actor adapts to the
        # changed arm/frame mass distribution.
        friction_range = [0.8, 1.2]
        restitution_range = [0.0, 0.1]
        added_mass_range = [-2.0, 2.0]
        multiplied_link_mass_range = [0.98, 1.02]
        added_base_com_range = [-0.01, 0.01]
        stiffness_multiplier_range = [0.95, 1.05]
        damping_multiplier_range = [0.95, 1.05]
        motor_zero_offset_range = [-0.01, 0.01]
        motor_strength_range = [0.95, 1.0]
        randomize_action_delay = False
        push_robots = False

    class rewards(RM75TorqueLimitedFlatTrotCfg.rewards):
        # Now that contact impulses are measured on the actual foot bodies,
        # explicitly refine the inherited diagonal gait while reducing slip
        # and the prolonged knee saturation seen in deterministic evaluation.
        soft_torque_limit = 0.85
        trot_target_contact_duty = 0.47

        class scales(RM75TorqueLimitedFlatTrotCfg.rewards.scales):
            tracking_ang_vel = 2.0
            trot_gait = 2.0
            lin_vel_z = -2.0
            torques = -3.0e-6
            dof_power = -1.5e-6
            torque_limits = -0.02
            feet_slip = -0.25


class RM75NewModelEarthFlatAdaptCfgPPO(RM75TorqueLimitedFlatTrotCfgPPO):
    class algorithm(RM75TorqueLimitedFlatTrotCfgPPO.algorithm):
        learning_rate = 7.5e-5
        entropy_coef = 0.0005
        schedule = "fixed"

    class runner(RM75TorqueLimitedFlatTrotCfgPPO.runner):
        max_iterations = 8000
        save_interval = 250
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        warm_start_path = SOURCE_FLAT_CHECKPOINT
        warm_start_max_action_std = 0.12


class RM75NewModelEarthFlatHip100Knee150Cfg(RM75NewModelEarthFlatAdaptCfg):
    class control(RM75NewModelEarthFlatAdaptCfg.control):
        joint_torque_limits = {"hip": 100.0, "thigh": 100.0, "calf": 150.0}


class RM75NewModelEarthFlatHip100Knee150CfgPPO(
    RM75NewModelEarthFlatAdaptCfgPPO
):
    class runner(RM75NewModelEarthFlatAdaptCfgPPO.runner):
        experiment_name = "RM75_newmodel_earth_flat_hip100_knee150"
        run_name = "flat_trot_contact_heading_recovery_v3"
        warm_start_path = HIP100_RECOVERY_CHECKPOINT


class RM75NewModelEarthFlatHip84Knee150Cfg(RM75NewModelEarthFlatAdaptCfg):
    class control(RM75NewModelEarthFlatAdaptCfg.control):
        joint_torque_limits = {"hip": 84.0, "thigh": 84.0, "calf": 150.0}


class RM75NewModelEarthFlatHip84Knee150CfgPPO(
    RM75NewModelEarthFlatAdaptCfgPPO
):
    class runner(RM75NewModelEarthFlatAdaptCfgPPO.runner):
        experiment_name = "RM75_newmodel_earth_flat_hip84_knee150"
        run_name = "flat_trot_contact_heading_recovery_v3"
        warm_start_path = HIP84_RECOVERY_CHECKPOINT


class RM75NewModelEarthFlatHip84Knee150RefineCfg(
    RM75NewModelEarthFlatHip84Knee150Cfg
):
    """Low-rate refinement of the converged 3 m/s policy's duty-cycle bias."""

    class commands(RM75NewModelEarthFlatHip84Knee150Cfg.commands):
        command_range_curriculum = []

        class ranges(RM75NewModelEarthFlatHip84Knee150Cfg.commands.ranges):
            lin_vel_x = [3.0, 3.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]
            new_lin_vel_x = [3.0, 3.0]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [-1.0, 1.0]
            new_heading = [0.0, 0.0]

    class rewards(RM75NewModelEarthFlatHip84Knee150Cfg.rewards):
        # Apply the full objective immediately: this task warm-starts from a
        # stable policy and only needs to remove its persistent diagonal bias.
        curriculum_rewards = []
        trot_duty_balance_tolerance = 0.08

        class scales(RM75NewModelEarthFlatHip84Knee150Cfg.rewards.scales):
            tracking_ang_vel = 2.5
            trot_gait = 3.0
            trot_duty_balance = -30.0
            feet_slip = -0.30


class RM75NewModelEarthFlatHip84Knee150RefineCfgPPO(
    RM75NewModelEarthFlatHip84Knee150CfgPPO
):
    class algorithm(RM75NewModelEarthFlatHip84Knee150CfgPPO.algorithm):
        learning_rate = 2.5e-5
        entropy_coef = 0.0001
        schedule = "fixed"

    class runner(RM75NewModelEarthFlatHip84Knee150CfgPPO.runner):
        experiment_name = "RM75_newmodel_earth_flat_hip84_knee150_refine"
        run_name = "duty_balance_refine_v1"
        max_iterations = 1200
        save_interval = 100
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        warm_start_path = HIP84_V4_BEST_CHECKPOINT
        warm_start_max_action_std = 0.05


class RM75NewModelEarthFlatHip84Knee150RefineV2Cfg(
    RM75NewModelEarthFlatHip84Knee150RefineCfg
):
    """Directly target every foot's duty cycle after v1 plateaued."""

    class rewards(RM75NewModelEarthFlatHip84Knee150RefineCfg.rewards):
        trot_duty_balance_tolerance = 0.05

        class scales(RM75NewModelEarthFlatHip84Knee150RefineCfg.rewards.scales):
            trot_gait = 3.5
            trot_duty_balance = -150.0


class RM75NewModelEarthFlatHip84Knee150RefineV2CfgPPO(
    RM75NewModelEarthFlatHip84Knee150RefineCfgPPO
):
    class algorithm(RM75NewModelEarthFlatHip84Knee150RefineCfgPPO.algorithm):
        learning_rate = 1.5e-5
        entropy_coef = 5.0e-5
        schedule = "fixed"

    class runner(RM75NewModelEarthFlatHip84Knee150RefineCfgPPO.runner):
        experiment_name = "RM75_newmodel_earth_flat_hip84_knee150_refine_v2"
        run_name = "per_foot_duty_refine_v2"
        max_iterations = 800
        save_interval = 50
        warm_start_path = HIP84_REFINE_V1_BEST_CHECKPOINT
        warm_start_max_action_std = 0.04


class RM75NewModelEarthRampCfg(RM75TorqueLimitedVisualRampTrot35DegCfg):
    """35-degree visual ramp curriculum after V1.1 flat adaptation."""

    class init_state(RM75TorqueLimitedVisualRampTrot35DegCfg.init_state):
        default_joint_angles = dict(V11_DEFAULT_JOINT_ANGLES)

    class terrain(RM75TorqueLimitedVisualRampTrot35DegCfg.terrain):
        level_curriculum = [
            {"iter": 0, "max_level": 2},
            {"iter": 1500, "max_level": 3},
            {"iter": 3500, "max_level": 4},
            {"iter": 6500, "max_level": 6},
            {"iter": 10500, "max_level": 7},
            {"iter": 16000, "max_level": 8},
            {"iter": 22000, "max_level": 9},
        ]


class RM75NewModelEarthRampCfgPPO(
    RM75TorqueLimitedVisualRampTrot35DegCfgPPO
):
    class policy(RM75TorqueLimitedVisualRampTrot35DegCfgPPO.policy):
        visual_residual_freeze_iterations = 250
        base_actor_freeze_iterations = 1500
        base_actor_lr_scale = 0.20
        init_noise_std = 0.30

    class algorithm(RM75TorqueLimitedVisualRampTrot35DegCfgPPO.algorithm):
        learning_rate = 1.25e-4
        entropy_coef = 0.002
        schedule = "fixed"

    class runner(RM75TorqueLimitedVisualRampTrot35DegCfgPPO.runner):
        max_iterations = 30000
        save_interval = 100
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        warm_start_max_action_std = 0.25
        warm_start_critic_common_dim = 76


class RM75NewModelEarthRampHip100Knee150Cfg(RM75NewModelEarthRampCfg):
    class control(RM75NewModelEarthRampCfg.control):
        joint_torque_limits = {"hip": 100.0, "thigh": 100.0, "calf": 150.0}


class RM75NewModelEarthRampHip100Knee150CfgPPO(
    RM75NewModelEarthRampCfgPPO
):
    class runner(RM75NewModelEarthRampCfgPPO.runner):
        experiment_name = "RM75_newmodel_earth_ramp35_hip100_knee150"
        run_name = "stage_b_visual_ramp"
        warm_start_path = (
            "{LEGGED_GYM_ROOT_DIR}/logs/"
            "RM75_newmodel_earth_flat_hip100_knee150/selected/model.pt"
        )


class RM75NewModelEarthRampHip84Knee150Cfg(RM75NewModelEarthRampCfg):
    class control(RM75NewModelEarthRampCfg.control):
        joint_torque_limits = {"hip": 84.0, "thigh": 84.0, "calf": 150.0}


class RM75NewModelEarthRampHip84Knee150CfgPPO(
    RM75NewModelEarthRampCfgPPO
):
    class runner(RM75NewModelEarthRampCfgPPO.runner):
        experiment_name = "RM75_newmodel_earth_ramp35_hip84_knee150"
        run_name = "stage_b_visual_ramp"
        warm_start_path = (
            "{LEGGED_GYM_ROOT_DIR}/logs/"
            "RM75_newmodel_earth_flat_hip84_knee150/selected/model.pt"
        )
