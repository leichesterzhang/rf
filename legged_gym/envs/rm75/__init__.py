"""RM75 task package."""

from .rm75_config_parkour_moe import RM75ParkourCfg, RM75CfgParkourMoE
from .rm75_config_flat_trot import RM75FlatTrotCfg, RM75FlatTrotCfgPPO
from .rm75_flat_trot_env import RM75FlatTrotRobot
from .rm75_config_torque_limited_flat_trot import (
    RM75TorqueLimitedFlatTrotCfg,
    RM75TorqueLimitedFlatTrotCfgPPO,
)
from .rm75_torque_limited_flat_trot_env import (
    RM75TorqueLimitedFlatTrotRobot,
)
from .rm75_config_visual_ramp_trot import (
    RM75VisualRampTrotCfg,
    RM75VisualRampTrotCfgPPO,
)
from .rm75_visual_ramp_trot_env import RM75VisualRampTrotRobot
from .rm75_config_visual_ramp_trot_35deg import (
    RM75VisualRampTrot35DegCfg,
    RM75VisualRampTrot35DegCfgPPO,
)
from .rm75_config_visual_ramp_trot_35deg_torque_limited import (
    RM75TorqueLimitedVisualRampTrot35DegCfg,
    RM75TorqueLimitedVisualRampTrot35DegCfgPPO,
)
from .rm75_torque_limited_visual_ramp_trot_env import (
    RM75TorqueLimitedVisualRampTrotRobot,
)
from .rm75_config_visual_ramp_trot_35deg_torque_variants import (
    RM75TorqueLimitedVisualRampTrot35DegHip100Knee150Cfg,
    RM75TorqueLimitedVisualRampTrot35DegHip100Knee150CfgPPO,
    RM75TorqueLimitedVisualRampTrot35DegHip84Knee150Cfg,
    RM75TorqueLimitedVisualRampTrot35DegHip84Knee150CfgPPO,
)
from .rm75_config_newmodel_earth_torque_pipeline import (
    RM75NewModelEarthFlatHip100Knee150Cfg,
    RM75NewModelEarthFlatHip100Knee150CfgPPO,
    RM75NewModelEarthFlatHip84Knee150Cfg,
    RM75NewModelEarthFlatHip84Knee150CfgPPO,
    RM75NewModelEarthFlatHip84Knee150RefineCfg,
    RM75NewModelEarthFlatHip84Knee150RefineCfgPPO,
    RM75NewModelEarthFlatHip84Knee150RefineV2Cfg,
    RM75NewModelEarthFlatHip84Knee150RefineV2CfgPPO,
    RM75NewModelEarthRampHip100Knee150Cfg,
    RM75NewModelEarthRampHip100Knee150CfgPPO,
    RM75NewModelEarthRampHip84Knee150Cfg,
    RM75NewModelEarthRampHip84Knee150CfgPPO,
)
