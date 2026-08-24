from legged_gym.envs.go2.go2_config_parkour_moe import (
    MGDP_FLAT_TERRAIN_ID,
    MGDP_ROUGH_FLAT_TERRAIN_ID,
)
from legged_gym.envs.rm75.rm75_config_parkour_moe import (
    RM75CfgParkourMoE,
    RM75ParkourCfg,
)


RM75_TROT_TERRAIN_IDS = [8, MGDP_FLAT_TERRAIN_ID, MGDP_ROUGH_FLAT_TERRAIN_ID]


class RM75TrotSlopeCfg(RM75ParkourCfg):
    """Forward-only RM75 trot at 3 m/s on flat and 1.5 m/s on a 35-degree ramp."""

    class sim(RM75ParkourCfg.sim):
        gravity = [0.0, 0.0, -9.81]

        class physx(RM75ParkourCfg.sim.physx):
            # 4096 RM75s on the shared trimesh need roughly 19.5M aggregate
            # broad-phase pairs. Keep margin so PhysX does not silently drop
            # contacts as the robots spread across the terrain.
            max_gpu_contact_pairs = 2**25
            default_buffer_size_multiplier = 10

    class env(RM75ParkourCfg.env):
        episode_length_s = 20.0

    class init_state(RM75ParkourCfg.init_state):
        turn_over = False
        zero_reset_root_velocities = True

    class domain_rand(RM75ParkourCfg.domain_rand):
        # A 35-degree slope needs mu > tan(35 deg) ~= 0.70 even before
        # dynamic effects. Keep the training range physically feasible.
        friction_range = [0.80, 1.25]
        max_push_vel_xy = 0.20
        max_push_ang_vel = 0.25

    class terrain(RM75ParkourCfg.terrain):
        curriculum = True
        num_rows = 10
        num_cols = 30
        max_init_terrain_level = 1
        ramp_max_angle_deg = 35.0
        terrain_length = 12.0
        terrain_width = 5.0
        start_platform_length = 2.5
        start_platform_width = 2.4
        # MGDP terrain order:
        # [single_gap, step_stone, two_row_stones, one_row_stones,
        #  single_bridge, air_beams, air_stones, hurdle, ramp, corridor,
        #  stairs_up, flat, rough_flat]
        terrain_proportions = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.60,
            0.0,
            0.0,
            0.30,
            0.10,
        ]

    class commands(RM75ParkourCfg.commands):
        heading_command = True
        omni_enabled = False
        omni_ratio = 0.0
        omni_terrain_ids = []
        zero_command = False
        flat_speed_search_enabled = False
        resampling_time = 5.0

        # Train only world +X forward trot. forward_only is an explicit guard in
        # the environment; the one-hot probabilities also keep saved configs and
        # diagnostics self-describing.
        forward_only = True
        locomotion_mode_probabilities = [0.0, 1.0, 0.0, 0.0]

        # Always use the exact terrain-conditioned target for the active stage
        # instead of mixing forward and lateral commands.  Starting a random
        # policy directly at 3 m/s made every episode terminate on excessive
        # tilt, so the forward-only fine-tune ramps speed while the terrain
        # curriculum independently raises the ramp to exactly 35 degrees.
        fixed_target_speed = True
        minimum_speed_ratio = 1.0
        minimum_moving_speed = 0.0
        target_speed_probability = 1.0

        trot_speed_curriculum = [
            {
                "iter": 0,
                "flat_forward": 0.8,
                "slope_forward": 0.4,
                "flat_lateral": 0.0,
                "slope_lateral": 0.0,
            },
            {
                "iter": 2000,
                "flat_forward": 1.5,
                "slope_forward": 0.8,
                "flat_lateral": 0.0,
                "slope_lateral": 0.0,
            },
            {
                "iter": 6000,
                "flat_forward": 2.2,
                "slope_forward": 1.1,
                "flat_lateral": 0.0,
                "slope_lateral": 0.0,
            },
            {
                "iter": 12000,
                "flat_forward": 3.0,
                "slope_forward": 1.5,
                "flat_lateral": 0.0,
                "slope_lateral": 0.0,
            },
        ]

        # The custom sampler owns the discrete flat/ramp targets; keep the
        # inherited rectangular command curriculum disabled.
        command_range_curriculum = []
        terrain_max_command_ranges = [
            {
                "lin_vel_x": [0.0, 3.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [-1.2, 1.2],
                "heading": [0.0, 0.0],
            }
            for _ in range(13)
        ]

        class ranges(RM75ParkourCfg.commands.ranges):
            lin_vel_x = [0.0, 3.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.2, 1.2]
            heading = [0.0, 0.0]
            new_lin_vel_x = [0.0, 3.0]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [-1.2, 1.2]
            new_heading = [0.0, 0.0]

    class termination(RM75ParkourCfg.termination):
        trot_max_roll = 1.0
        trot_max_pitch = 1.15

    class rewards(RM75ParkourCfg.rewards):
        only_positive_rewards = False
        base_height_target = 0.53
        max_contact_force = 420.0

        tracking_forward_sigma = 0.50
        tracking_lateral_sigma = 0.35
        tracking_cross_sigma = 0.20
        tracking_stand_sigma = 0.12

        trot_command_threshold = 0.08
        trot_terrain_ids = RM75_TROT_TERRAIN_IDS
        # Explicit order used by the diagonal-pair rewards:
        # FL/RR versus FR/RL.
        trot_foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        trot_foot_height_sigma = 0.05
        trot_no_progress_timeout_s = 3.0
        trot_no_progress_min_delta = 0.08
        terrain_normal_scan_x = 0.50
        terrain_normal_scan_y = 0.40

        flat_gait_terrain_ids = RM75_TROT_TERRAIN_IDS
        flat_gait_cmd_threshold = 0.08
        stand_still_terrain_ids = RM75_TROT_TERRAIN_IDS
        base_height_excluded_terrain_ids = []
        hip_to_default_excluded_terrain_ids = []
        heading_conditioned_terrain_ids = []
        tracking_lin_vel_full_reward_terrain_ids = []

        # RM75TrotSlopeRobot uses route progress and a stricter timeout tailored
        # to this forward-only task.
        no_progress_timeout_s = 0.0

        curriculum_rewards = [
            {
                "reward_name": "correct_base_height",
                "start_iter": 0,
                "end_iter": 6000,
                "start_value": 0.5,
                "end_value": 2.0,
            },
            {
                "reward_name": "trot_diagonal_sync",
                "start_iter": 0,
                "end_iter": 4000,
                "start_value": 0.25,
                "end_value": 1.0,
            },
            {
                "reward_name": "trot_diagonal_opposition",
                "start_iter": 0,
                "end_iter": 4000,
                "start_value": 0.25,
                "end_value": 1.0,
            },
            {
                "reward_name": "feet_slip",
                "start_iter": 0,
                "end_iter": 6000,
                "start_value": 0.25,
                "end_value": 1.0,
            },
        ]

        class scales(RM75ParkourCfg.rewards.scales):
            termination = -20.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            correct_base_height = -1.0
            torques = -2.0e-6
            dof_acc = -1.0e-8
            feet_regulation = -0.02
            hip_to_default = -0.02
            action_rate = -0.006
            action_smoothness = -0.006
            collision = -2.0
            feet_contact_forces = -0.002
            stand_still = -0.5
            low_speed_when_commanded = -1.0

            # Keep the contact-duty regularizer, but use explicit diagonal
            # terms below to define trot rather than merely "stable gait".
            flat_gait = -0.25
            trot_diagonal_sync = -1.5
            trot_diagonal_opposition = -1.0
            trot_foot_height_sync = -0.20
            feet_slip = -0.25
            trot_heading_error = -0.80
            terrain_normal_alignment = -0.75

            # Smooth ramps do not need the inherited parkour skill rewards.
            hard_terrain_yaw_penalty = 0.0
            omni_target_yaw_error = 0.0
            upward_foothold_clearance = 0.0
            targeted_foothold_touchdown = 0.0
            single_bridge_narrow_stance = 0.0
            flat_forward_speed = 0.0
            stairs_omni_lead_touch_above = 0.0
            stairs_omni_trail_followup = 0.0
            stairs_omni_support_transfer_progress = 0.0


class RM75CfgTrotSlopeMoE(RM75CfgParkourMoE):
    class runner(RM75CfgParkourMoE.runner):
        experiment_name = "RM75_trot_slope"
        run_name = ""
        max_iterations = 30000
        save_interval = 100
        # This task is a forward-only fine-tune.  Reuse the mature policy and
        # estimator, but start with fresh optimizer moments/LR rather than the
        # 1e-5 PPO learning rate stored in the mixed-locomotion checkpoint.
        load_optimizer_on_resume = False
