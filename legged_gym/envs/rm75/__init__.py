"""RM75 task package."""

from .rm75_config_parkour_moe import RM75ParkourCfg, RM75CfgParkourMoE
from .rm75_config_flat_trot import RM75FlatTrotCfg, RM75FlatTrotCfgPPO
from .rm75_flat_trot_env import RM75FlatTrotRobot
from .rm75_config_visual_ramp_trot import (
    RM75VisualRampTrotCfg,
    RM75VisualRampTrotCfgPPO,
)
from .rm75_visual_ramp_trot_env import RM75VisualRampTrotRobot
from .rm75_config_visual_ramp_trot_35deg import (
    RM75VisualRampTrot35DegCfg,
    RM75VisualRampTrot35DegCfgPPO,
)
