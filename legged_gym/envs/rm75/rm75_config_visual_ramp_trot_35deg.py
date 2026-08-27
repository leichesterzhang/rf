"""RM75 visual trot training extended to continuous 35 degree ramps."""

from legged_gym.envs.rm75.rm75_config_visual_ramp_trot import (
    RM75VisualRampTrotCfg,
    RM75VisualRampTrotCfgPPO,
)


class RM75VisualRampTrot35DegCfg(RM75VisualRampTrotCfg):
    class terrain(RM75VisualRampTrotCfg.terrain):
        ramp_max_angle_deg = 35.0
        # With ten rows and mgdp_max_difficulty=0.9 these levels correspond
        # approximately to 8, 16, 23, 27, 31, and 35 degrees.
        level_curriculum = [
            {"iter": 0, "max_level": 2},
            {"iter": 1000, "max_level": 4},
            {"iter": 5000, "max_level": 6},
            {"iter": 10000, "max_level": 7},
            {"iter": 16000, "max_level": 8},
            {"iter": 22000, "max_level": 9},
        ]


class RM75VisualRampTrot35DegCfgPPO(RM75VisualRampTrotCfgPPO):
    class runner(RM75VisualRampTrotCfgPPO.runner):
        experiment_name = "RM75_visual_ramp_trot_3ms_35deg"
        run_name = "warmstart_model25000"

