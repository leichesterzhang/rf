"""Additional hard-torque-limit variants of the RM75 35 degree ramp task."""

from legged_gym.envs.rm75.rm75_config_visual_ramp_trot_35deg_torque_limited import (
    RM75TorqueLimitedVisualRampTrot35DegCfg,
    RM75TorqueLimitedVisualRampTrot35DegCfgPPO,
)


class RM75TorqueLimitedVisualRampTrot35DegHip100Knee150Cfg(
    RM75TorqueLimitedVisualRampTrot35DegCfg
):
    """Use 100 N m at both hip motors and 150 N m at the knee motor."""

    class control(RM75TorqueLimitedVisualRampTrot35DegCfg.control):
        joint_torque_limits = {
            "hip": 100.0,
            "thigh": 100.0,
            "calf": 150.0,
        }


class RM75TorqueLimitedVisualRampTrot35DegHip100Knee150CfgPPO(
    RM75TorqueLimitedVisualRampTrot35DegCfgPPO
):
    class runner(RM75TorqueLimitedVisualRampTrot35DegCfgPPO.runner):
        experiment_name = "RM75_visual_ramp_trot_3ms_35deg_hip100_knee150"
        run_name = "warmstart_torque_hip100_knee160_model30000"


class RM75TorqueLimitedVisualRampTrot35DegHip84Knee150Cfg(
    RM75TorqueLimitedVisualRampTrot35DegCfg
):
    """Use 84 N m at both hip motors and 150 N m at the knee motor."""

    class control(RM75TorqueLimitedVisualRampTrot35DegCfg.control):
        joint_torque_limits = {
            "hip": 84.0,
            "thigh": 84.0,
            "calf": 150.0,
        }


class RM75TorqueLimitedVisualRampTrot35DegHip84Knee150CfgPPO(
    RM75TorqueLimitedVisualRampTrot35DegCfgPPO
):
    class runner(RM75TorqueLimitedVisualRampTrot35DegCfgPPO.runner):
        experiment_name = "RM75_visual_ramp_trot_3ms_35deg_hip84_knee150"
        run_name = "warmstart_torque_hip100_knee160_model30000"
