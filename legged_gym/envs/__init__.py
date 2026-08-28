from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_env import Go2Robot
from legged_gym.envs.go2.go2_parkour_env import Go2ParkourRobot
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO, GO2CfgCTS, GO2CfgMoECTS, GO2CfgMoENGCTS, GO2CfgMCPCTS, GO2CfgACMoECTS, GO2CfgDualMoECTS
from legged_gym.envs.go2.go2_config_parkour_moe import GO2ParkourCfg, GO2CfgParkourMoE
from legged_gym.envs.dog_502.dog_502_config_parkour_moe import DOG502ParkourCfg, DOG502CfgParkourMoE
from legged_gym.envs.b2.b2_config_parkour_moe import B2ParkourCfg, B2CfgParkourMoE
from legged_gym.envs.rm75.rm75_config_parkour_moe import RM75ParkourCfg, RM75CfgParkourMoE
from legged_gym.envs.rm75.rm75_config_flat_trot import RM75FlatTrotCfg, RM75FlatTrotCfgPPO
from legged_gym.envs.rm75.rm75_flat_trot_env import RM75FlatTrotRobot
from legged_gym.envs.rm75.rm75_config_torque_limited_flat_trot import (
    RM75TorqueLimitedFlatTrotCfg,
    RM75TorqueLimitedFlatTrotCfgPPO,
)
from legged_gym.envs.rm75.rm75_torque_limited_flat_trot_env import (
    RM75TorqueLimitedFlatTrotRobot,
)
from legged_gym.envs.rm75.rm75_config_visual_ramp_trot import RM75VisualRampTrotCfg, RM75VisualRampTrotCfgPPO
from legged_gym.envs.rm75.rm75_visual_ramp_trot_env import RM75VisualRampTrotRobot
from legged_gym.envs.rm75.rm75_config_visual_ramp_trot_35deg import (
    RM75VisualRampTrot35DegCfg,
    RM75VisualRampTrot35DegCfgPPO,
)
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2", Go2Robot, GO2Cfg(), GO2CfgPPO())
task_registry.register("go2_cts", Go2Robot, GO2Cfg(), GO2CfgCTS())
task_registry.register("go2_moe_cts", Go2Robot, GO2Cfg(), GO2CfgMoECTS())
task_registry.register("go2_moe_ng_cts", Go2Robot, GO2Cfg(), GO2CfgMoENGCTS())
task_registry.register("go2_mcp_cts", Go2Robot, GO2Cfg(), GO2CfgMCPCTS())
task_registry.register("go2_ac_moe_cts", Go2Robot, GO2Cfg(), GO2CfgACMoECTS())
task_registry.register("go2_dual_moe_cts", Go2Robot, GO2Cfg(), GO2CfgDualMoECTS())
task_registry.register("go2_parkour_moe", Go2ParkourRobot, GO2ParkourCfg(), GO2CfgParkourMoE())
task_registry.register("dog_502_parkour_moe", Go2ParkourRobot, DOG502ParkourCfg(), DOG502CfgParkourMoE())
task_registry.register("b2_parkour_moe", Go2ParkourRobot, B2ParkourCfg(), B2CfgParkourMoE())
task_registry.register("RM75_parkour_moe", Go2ParkourRobot, RM75ParkourCfg(), RM75CfgParkourMoE())
task_registry.register("RM75_flat_trot_3ms", RM75FlatTrotRobot, RM75FlatTrotCfg(), RM75FlatTrotCfgPPO())
task_registry.register(
    "RM75_flat_trot_3ms_hip100_knee160",
    RM75TorqueLimitedFlatTrotRobot,
    RM75TorqueLimitedFlatTrotCfg(),
    RM75TorqueLimitedFlatTrotCfgPPO(),
)
task_registry.register(
    "RM75_visual_ramp_trot_3ms_25deg",
    RM75VisualRampTrotRobot,
    RM75VisualRampTrotCfg(),
    RM75VisualRampTrotCfgPPO(),
)
task_registry.register(
    "RM75_visual_ramp_trot_3ms_35deg",
    RM75VisualRampTrotRobot,
    RM75VisualRampTrot35DegCfg(),
    RM75VisualRampTrot35DegCfgPPO(),
)
