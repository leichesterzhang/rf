# RM75 有效配置与奖励源码证据

由 tools/audit_rm75_source.py 生成；解析配置继承，未启动仿真。权重为配置原值，运行时通常还乘策略步长和奖励课程系数。

## RM75_parkour_moe

环境：`Go2ParkourRobot`；Actor：`ActorCriticParkourMoE`；算法：`PPOParkourMoE`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.008 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.008 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.05 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -1.5 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -1.5 | [legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -2.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -2e-05 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| feet_regulation | -0.025 | [legged_gym/envs/base/legged_robot.py:1700](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1700) |
| flat_forward_speed | 1.0 | [legged_gym/envs/go2/go2_parkour_env.py:1155](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1155) |
| flat_gait | -0.8 | [legged_gym/envs/go2/go2_parkour_env.py:1228](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1228) |
| hard_terrain_yaw_penalty | -0.2 | [legged_gym/envs/go2/go2_parkour_env.py:1167](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1167) |
| hip_to_default | -0.03 | [legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120) |
| lin_vel_z | -2.0 | [legged_gym/envs/base/legged_robot.py:1519](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1519) |
| low_speed_when_commanded | -0.5 | [legged_gym/envs/go2/go2_parkour_env.py:1097](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1097) |
| omni_target_yaw_error | -0.05 | [legged_gym/envs/go2/go2_parkour_env.py:1171](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1171) |
| single_bridge_narrow_stance | 0.5 | [legged_gym/envs/go2/go2_parkour_env.py:1191](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1191) |
| stairs_omni_lead_touch_above | 0.6 | [legged_gym/envs/go2/go2_parkour_env.py:1179](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1179) |
| stairs_omni_support_transfer_progress | 0.8 | [legged_gym/envs/go2/go2_parkour_env.py:1187](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1187) |
| stairs_omni_trail_followup | 1.0 | [legged_gym/envs/go2/go2_parkour_env.py:1183](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1183) |
| stand_still | -0.5 | [legged_gym/envs/go2/go2_parkour_env.py:1146](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1146) |
| targeted_foothold_touchdown | 0.5 | [legged_gym/envs/go2/go2_parkour_env.py:1565](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1565) |
| torques | -2e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 0.5 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 1.0 | [legged_gym/envs/go2/go2_parkour_env.py:1130](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1130) |
| upward_foothold_clearance | 0.5 | [legged_gym/envs/go2/go2_parkour_env.py:1330](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1330) |

配置类位置：[legged_gym/envs/rm75/rm75_config_parkour_moe.py:10](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_parkour_moe.py:10)

## RM75_flat_trot_3ms

环境：`RM75FlatTrotRobot`；Actor：`ActorCritic`；算法：`PPO`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.01 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.005 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/base/legged_robot.py:1690](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1690) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -1e-06 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_regulation | -0.02 | [legged_gym/envs/base/legged_robot.py:1700](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1700) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:129](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:129) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_env.py:55](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_env.py:55) |
| lin_vel_z | -1.5 | [legged_gym/envs/base/legged_robot.py:1519](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1519) |
| orientation | -1.0 | [legged_gym/envs/base/legged_robot.py:1527](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1527) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.01 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -2e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/base/legged_robot.py:1613](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1613) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:83](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:83) |

配置类位置：[legged_gym/envs/rm75/rm75_config_flat_trot.py:7](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_flat_trot.py:7)

## RM75_flat_trot_3ms_hip100_knee160

环境：`RM75TorqueLimitedFlatTrotRobot`；Actor：`ActorCritic`；算法：`PPO`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.01 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.005 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/base/legged_robot.py:1690](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1690) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -1e-06 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_regulation | -0.02 | [legged_gym/envs/base/legged_robot.py:1700](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1700) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:129](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:129) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_env.py:55](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_env.py:55) |
| lin_vel_z | -1.5 | [legged_gym/envs/base/legged_robot.py:1519](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1519) |
| orientation | -1.0 | [legged_gym/envs/base/legged_robot.py:1527](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1527) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.01 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -2e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/base/legged_robot.py:1613](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1613) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:83](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:83) |

配置类位置：[legged_gym/envs/rm75/rm75_config_torque_limited_flat_trot.py:9](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_torque_limited_flat_trot.py:9)

## RM75_visual_ramp_trot_3ms_25deg

环境：`RM75VisualRampTrotRobot`；Actor：`ActorCriticVisualResidual`；算法：`PPOParkourMoE`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.01 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.005 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -1e-06 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| downhill_overspeed | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120) |
| lateral_velocity | -0.4 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334) |
| lin_vel_z | -1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325) |
| orientation | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329) |
| speed_tracking_error | -0.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.01 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -2e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350) |

配置类位置：[legged_gym/envs/rm75/rm75_config_visual_ramp_trot.py:9](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_visual_ramp_trot.py:9)

## RM75_visual_ramp_trot_3ms_35deg

环境：`RM75VisualRampTrotRobot`；Actor：`ActorCriticVisualResidual`；算法：`PPOParkourMoE`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.01 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.005 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -1e-06 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| downhill_overspeed | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120) |
| lateral_velocity | -0.4 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334) |
| lin_vel_z | -1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325) |
| orientation | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329) |
| speed_tracking_error | -0.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.01 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -2e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350) |

配置类位置：[legged_gym/envs/rm75/rm75_config_visual_ramp_trot_35deg.py:9](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_visual_ramp_trot_35deg.py:9)

## RM75_visual_ramp_trot_3ms_35deg_hip100_knee160

环境：`RM75TorqueLimitedVisualRampTrotRobot`；Actor：`ActorCriticVisualResidual`；算法：`PPOParkourMoE`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.012 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.006 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -7.5e-07 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| downhill_overspeed | -1.2 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120) |
| lateral_velocity | -0.4 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334) |
| lin_vel_z | -1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325) |
| orientation | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329) |
| speed_tracking_error | -0.65 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.006 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350) |

配置类位置：[legged_gym/envs/rm75/rm75_config_visual_ramp_trot_35deg_torque_limited.py:9](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_visual_ramp_trot_35deg_torque_limited.py:9)

## RM75_visual_ramp_trot_3ms_35deg_hip100_knee150

环境：`RM75TorqueLimitedVisualRampTrotRobot`；Actor：`ActorCriticVisualResidual`；算法：`PPOParkourMoE`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.012 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.006 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -7.5e-07 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| downhill_overspeed | -1.2 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120) |
| lateral_velocity | -0.4 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334) |
| lin_vel_z | -1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325) |
| orientation | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329) |
| speed_tracking_error | -0.65 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.006 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350) |

配置类位置：[legged_gym/envs/rm75/rm75_config_visual_ramp_trot_35deg_torque_variants.py:9](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_visual_ramp_trot_35deg_torque_variants.py:9)

## RM75_visual_ramp_trot_3ms_35deg_hip84_knee150

环境：`RM75TorqueLimitedVisualRampTrotRobot`；Actor：`ActorCriticVisualResidual`；算法：`PPOParkourMoE`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.012 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.006 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -7.5e-07 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| downhill_overspeed | -1.2 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120) |
| lateral_velocity | -0.4 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334) |
| lin_vel_z | -1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325) |
| orientation | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329) |
| speed_tracking_error | -0.65 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.006 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350) |

配置类位置：[legged_gym/envs/rm75/rm75_config_visual_ramp_trot_35deg_torque_variants.py:30](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_visual_ramp_trot_35deg_torque_variants.py:30)

## RM75_newmodel_earth_flat_hip100_knee150

环境：`RM75TorqueLimitedFlatTrotRobot`；Actor：`ActorCritic`；算法：`PPO`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.01 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.005 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/base/legged_robot.py:1690](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1690) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_regulation | -0.02 | [legged_gym/envs/base/legged_robot.py:1700](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1700) |
| feet_slip | -0.25 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:129](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:129) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_env.py:55](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_env.py:55) |
| lin_vel_z | -2.0 | [legged_gym/envs/base/legged_robot.py:1519](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1519) |
| orientation | -1.0 | [legged_gym/envs/base/legged_robot.py:1527](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1527) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.02 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -3e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 2.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/base/legged_robot.py:1613](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1613) |
| trot_gait | 2.0 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:83](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:83) |

配置类位置：[legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:140](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:140)

## RM75_newmodel_earth_flat_hip84_knee150

环境：`RM75TorqueLimitedFlatTrotRobot`；Actor：`ActorCritic`；算法：`PPO`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.01 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.005 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/base/legged_robot.py:1690](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1690) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_regulation | -0.02 | [legged_gym/envs/base/legged_robot.py:1700](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1700) |
| feet_slip | -0.25 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:129](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:129) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_env.py:55](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_env.py:55) |
| lin_vel_z | -2.0 | [legged_gym/envs/base/legged_robot.py:1519](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1519) |
| orientation | -1.0 | [legged_gym/envs/base/legged_robot.py:1527](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1527) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.02 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -3e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 2.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/base/legged_robot.py:1613](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1613) |
| trot_gait | 2.0 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:83](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:83) |

配置类位置：[legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:154](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:154)

## RM75_newmodel_earth_flat_hip84_knee150_refine

环境：`RM75TorqueLimitedFlatTrotRobot`；Actor：`ActorCritic`；算法：`PPO`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.01 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.005 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/base/legged_robot.py:1690](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1690) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_regulation | -0.02 | [legged_gym/envs/base/legged_robot.py:1700](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1700) |
| feet_slip | -0.3 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:129](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:129) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_env.py:55](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_env.py:55) |
| lin_vel_z | -2.0 | [legged_gym/envs/base/legged_robot.py:1519](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1519) |
| orientation | -1.0 | [legged_gym/envs/base/legged_robot.py:1527](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1527) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.02 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -3e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 2.5 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/base/legged_robot.py:1613](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1613) |
| trot_duty_balance | -30.0 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:140](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:140) |
| trot_gait | 3.0 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:83](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:83) |

配置类位置：[legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:168](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:168)

## RM75_newmodel_earth_flat_hip84_knee150_refine_v2

环境：`RM75TorqueLimitedFlatTrotRobot`；Actor：`ActorCritic`；算法：`PPO`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.01 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.005 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/base/legged_robot.py:1690](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1690) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_regulation | -0.02 | [legged_gym/envs/base/legged_robot.py:1700](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1700) |
| feet_slip | -0.3 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:129](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:129) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_env.py:55](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_env.py:55) |
| lin_vel_z | -2.0 | [legged_gym/envs/base/legged_robot.py:1519](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1519) |
| orientation | -1.0 | [legged_gym/envs/base/legged_robot.py:1527](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1527) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.02 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -3e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 2.5 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/base/legged_robot.py:1613](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1613) |
| trot_duty_balance | -150.0 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:140](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:140) |
| trot_gait | 3.5 | [legged_gym/envs/rm75/rm75_flat_trot_env.py:83](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:83) |

配置类位置：[legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:220](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:220)

## RM75_newmodel_earth_ramp35_hip100_knee150

环境：`RM75TorqueLimitedVisualRampTrotRobot`；Actor：`ActorCriticVisualResidual`；算法：`PPOParkourMoE`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.012 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.006 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -7.5e-07 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| downhill_overspeed | -1.2 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120) |
| lateral_velocity | -0.4 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334) |
| lin_vel_z | -1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325) |
| orientation | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329) |
| speed_tracking_error | -0.65 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.006 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350) |

配置类位置：[legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:293](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:293)

## RM75_newmodel_earth_ramp35_hip84_knee150

环境：`RM75TorqueLimitedVisualRampTrotRobot`；Actor：`ActorCriticVisualResidual`；算法：`PPOParkourMoE`。

| 奖励 | 配置权重 | 实际实现 |
|---|---:|---|
| action_rate | -0.012 | [legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564) |
| action_smoothness | -0.006 | [legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664) |
| ang_vel_xy | -0.1 | [legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523) |
| collision | -2.0 | [legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568) |
| correct_base_height | -2.0 | [legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116) |
| dof_acc | -1e-08 | [legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560) |
| dof_pos_limits | -3.0 | [legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576) |
| dof_power | -7.5e-07 | [legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672) |
| downhill_overspeed | -1.2 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341) |
| feet_contact_forces | -0.002 | [legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660) |
| feet_slip | -0.1 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397) |
| hip_to_default | -0.02 | [legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120) |
| lateral_velocity | -0.4 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334) |
| lin_vel_z | -1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325) |
| orientation | -1.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329) |
| speed_tracking_error | -0.65 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337) |
| termination | -2.0 | [legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572) |
| torque_limits | -0.006 | [legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587) |
| torques | -1.5e-06 | [legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552) |
| tracking_ang_vel | 1.0 | [legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627) |
| tracking_lin_vel | 4.0 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315) |
| trot_gait | 1.5 | [legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350) |

配置类位置：[legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:310](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_config_newmodel_earth_torque_pipeline.py:310)

# 奖励函数原文（去重）

## _reward_action_rate

[legged_gym/envs/base/legged_robot.py:1564](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1564)

```python
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

## _reward_action_smoothness

[legged_gym/envs/base/legged_robot.py:1664](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1664)

```python
    def _reward_action_smoothness(self):
        # a_t - 2a_{t-1} + a_{t-2}
        if not hasattr(self, 'last_last_actions'):
            self.last_last_actions = torch.zeros_like(self.last_actions)
        rew = torch.sum((self.actions - 2 * self.last_actions + self.last_last_actions).pow(2), dim=1)
        self.last_last_actions[:] = self.last_actions[:]
        return rew
```

## _reward_ang_vel_xy

[legged_gym/envs/base/legged_robot.py:1523](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1523)

```python
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
```

## _reward_collision

[legged_gym/envs/base/legged_robot.py:1568](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1568)

```python
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
```

## _reward_correct_base_height

[legged_gym/envs/go2/go2_parkour_env.py:1116](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1116)

```python
    def _reward_correct_base_height(self):
        rew = super()._reward_correct_base_height()
        return rew * self._base_height_penalty_mask().float()
```

## _reward_dof_acc

[legged_gym/envs/base/legged_robot.py:1560](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1560)

```python
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
```

## _reward_dof_pos_limits

[legged_gym/envs/base/legged_robot.py:1576](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1576)

```python
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
```

## _reward_dof_power

[legged_gym/envs/base/legged_robot.py:1672](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1672)

```python
    def _reward_dof_power(self):
        # Penalize power consumption
        power = self.torques * self.dof_vel
        rew = torch.sum(torch.abs(power), dim=1)
        return rew
```

## _reward_feet_regulation

[legged_gym/envs/base/legged_robot.py:1700](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1700)

```python
    def _reward_feet_regulation(self):
        # CTS抬腿正则奖励, 在脚末端速度增大同时, 要求高度尽可能高
        base_height = self._get_base_height()  # 更新刚体空间位置 (开悟比赛无法修改环境, 只能在奖励中完成计算了)
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        feet_xy_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:9]
        base_pos = self.root_states[:, 0:3].unsqueeze(1)
        delta_feet = feet_pos - base_pos
        feet2base_height = (delta_feet * self.projected_gravity.unsqueeze(1)).sum(-1)  # 脚相对于身体的高度 (N, 4)
        feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)  # 脚相对于地面的高度 (N, 4)
        rew = (feet_xy_vel.pow(2).sum(-1) * torch.exp(-feet_height / (0.025 * self.cfg.rewards.base_height_target))).sum(-1)
        return rew
```

## _reward_flat_forward_speed

[legged_gym/envs/go2/go2_parkour_env.py:1155](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1155)

```python
    def _reward_flat_forward_speed(self):
        speed_search_mask = self._flat_speed_search_mask()
        if not torch.any(speed_search_mask):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        speed = self._forward_reward_velocity_x().clamp(min=0.0)
        max_reward_speed = float(getattr(self.cfg.rewards, "flat_speed_search_max_speed_reward", 2.5))
        if max_reward_speed > 0.0:
            speed = torch.clamp(speed, max=max_reward_speed)
        if self.cfg.init_state.turn_over:
            speed_search_mask &= ~self._fall_recovery_mask()
        return speed * speed_search_mask.float()
```

## _reward_flat_gait

[legged_gym/envs/go2/go2_parkour_env.py:1228](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1228)

```python
    def _reward_flat_gait(self):
        active_envs = self._flat_gait_mask()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        prev_contact = self.flat_gait_last_contacts
        prev_contact_filt = self.flat_gait_last_contact_filt
        contact_filt = torch.logical_or(contact, prev_contact)
        active_feet = active_envs.unsqueeze(-1)

        min_stance_time = float(getattr(self.cfg.rewards, "flat_gait_min_stance_time", 0.10))
        liftoff = prev_contact_filt & ~contact_filt
        early_liftoff = liftoff & (self.flat_gait_stance_time < min_stance_time) & active_feet

        next_stance_time = torch.where(
            contact_filt,
            self.flat_gait_stance_time + self.dt,
            torch.zeros_like(self.flat_gait_stance_time),
        )
        self.flat_gait_stance_time = torch.where(
            active_feet,
            next_stance_time,
            torch.zeros_like(self.flat_gait_stance_time),
        )
        self.flat_gait_last_contacts = torch.where(
            active_feet,
            contact,
            torch.zeros_like(contact),
        )
        self.flat_gait_last_contact_filt = torch.where(
            active_feet,
            contact_filt,
            torch.zeros_like(contact_filt),
        )

        ema_alpha = float(getattr(self.cfg.rewards, "flat_gait_contact_ema_alpha", 0.98))
        ema_alpha = min(max(ema_alpha, 0.0), 1.0)
        neutral_duty = float(getattr(self.cfg.rewards, "flat_gait_min_contact_duty", 0.2))
        neutral_ema = torch.full_like(self.flat_gait_contact_ema, neutral_duty)
        contact_ema = ema_alpha * self.flat_gait_contact_ema + (1.0 - ema_alpha) * contact_filt.float()
        self.flat_gait_contact_ema = torch.where(active_feet, contact_ema, neutral_ema)

        min_contact_duty = float(getattr(self.cfg.rewards, "flat_gait_min_contact_duty", 0.20))
        low_contact_duty = (min_contact_duty - self.flat_gait_contact_ema).clamp(min=0.0).sum(dim=1)

        num_feet = self.feet_indices.shape[0]
        if num_feet >= 4:
            rear_contact_duty = self.flat_gait_contact_ema[:, 2:4]
        else:
            rear_contact_duty = self.flat_gait_contact_ema[:, max(0, num_feet // 2):]

        min_rear_contact_duty = float(getattr(self.cfg.rewards, "flat_gait_min_rear_contact_duty", 0.30))
        low_rear_duty = (min_rear_contact_duty - rear_contact_duty).clamp(min=0.0).sum(dim=1)
        if rear_contact_duty.shape[1] >= 2:
            rear_balance_tolerance = float(getattr(self.cfg.rewards, "flat_gait_rear_balance_tolerance", 0.20))
            rear_balance = (
                torch.abs(rear_contact_duty[:, 0] - rear_contact_duty[:, 1]) - rear_balance_tolerance
            ).clamp(min=0.0)
        else:
            rear_balance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        reward = (
            float(getattr(self.cfg.rewards, "flat_gait_early_liftoff_weight", 1.0))
            * early_liftoff.float().sum(dim=1)
            + float(getattr(self.cfg.rewards, "flat_gait_contact_duty_weight", 0.5))
            * low_contact_duty
            + float(getattr(self.cfg.rewards, "flat_gait_rear_duty_weight", 1.0))
            * low_rear_duty
            + float(getattr(self.cfg.rewards, "flat_gait_rear_balance_weight", 0.5))
            * rear_balance
        )
        return reward * active_envs.float()
```

## _reward_hard_terrain_yaw_penalty

[legged_gym/envs/go2/go2_parkour_env.py:1167](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1167)

```python
    def _reward_hard_terrain_yaw_penalty(self):
        conditioned_mask = self._heading_conditioned_mask()
        return torch.square(self.base_ang_vel[:, 2]) * conditioned_mask.float()
```

## _reward_hip_to_default

[legged_gym/envs/go2/go2_parkour_env.py:1120](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1120)

```python
    def _reward_hip_to_default(self):
        rew = super()._reward_hip_to_default()
        valid_mask = ~self._terrain_id_mask(self.hip_to_default_excluded_terrain_ids)
        return rew * valid_mask.float()
```

## _reward_lin_vel_z

[legged_gym/envs/base/legged_robot.py:1519](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1519)

```python
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
```

## _reward_low_speed_when_commanded

[legged_gym/envs/go2/go2_parkour_env.py:1097](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1097)

```python
    def _reward_low_speed_when_commanded(self):
        cmd_xy = self.commands[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        active_cmd = cmd_speed > self.cfg.rewards.low_speed_cmd_threshold

        actual_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        min_speed = torch.minimum(
            cmd_speed,
            torch.full_like(cmd_speed, self.cfg.rewards.low_speed_min_xy_speed),
        )
        return (min_speed - actual_speed).clamp(min=0.0) * active_cmd
```

## _reward_omni_target_yaw_error

[legged_gym/envs/go2/go2_parkour_env.py:1171](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1171)

```python
    def _reward_omni_target_yaw_error(self):
        omni_mask = self._omni_env_mask()
        if not torch.any(omni_mask):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        yaw_error = torch.abs(wrap_to_pi(self.omni_target_yaw - self._root_yaw()))
        deadzone = float(getattr(self.cfg.rewards, "omni_target_yaw_deadzone", 0.0))
        return torch.square((yaw_error - deadzone).clamp(min=0.0)) * omni_mask.float()
```

## _reward_single_bridge_narrow_stance

[legged_gym/envs/go2/go2_parkour_env.py:1191](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1191)

```python
    def _reward_single_bridge_narrow_stance(self):
        if not hasattr(self, "terrain_ids"):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        bridge_id = int(getattr(self.cfg.rewards, "single_bridge_terrain_id", 4))
        active_cmd = self._forward_reward_command_x() > float(
            getattr(self.cfg.rewards, "single_bridge_narrow_stance_cmd_threshold", 0.2)
        )
        active_envs = (self.terrain_ids == bridge_id) & active_cmd
        if not torch.any(active_envs):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        num_feet = self.feet_indices.shape[0]
        feet_pos_world = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[
            :, self.feet_indices, 0:3
        ]
        feet_pos_relative = feet_pos_world - self.root_states[:, 0:3].unsqueeze(1)
        local_feet_pos = quat_rotate_inverse(
            self.base_quat.repeat_interleave(num_feet, dim=0),
            feet_pos_relative.reshape(-1, 3),
        ).reshape(self.num_envs, num_feet, 3)

        lateral_target = float(getattr(self.cfg.rewards, "single_bridge_narrow_stance_lateral_target", 0.06))
        lateral_tolerance = float(
            getattr(self.cfg.rewards, "single_bridge_narrow_stance_lateral_tolerance", 0.01)
        )
        sigma = float(getattr(self.cfg.rewards, "single_bridge_narrow_stance_sigma", 0.0025))
        foot_signs = torch.tensor(
            [1.0, -1.0, 1.0, -1.0],
            device=self.device,
            dtype=local_feet_pos.dtype,
        ).unsqueeze(0)[:, :num_feet]
        target_y = foot_signs * lateral_target
        y_error = (torch.abs(local_feet_pos[:, :, 1] - target_y) - lateral_tolerance).clamp(min=0.0)
        reward = torch.exp(-torch.square(y_error) / max(sigma, 1.0e-6)).mean(dim=1)
        return reward * active_envs.float()
```

## _reward_stairs_omni_lead_touch_above

[legged_gym/envs/go2/go2_parkour_env.py:1179](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1179)

```python
    def _reward_stairs_omni_lead_touch_above(self):
        self._update_stairs_omni_reward_cache()
        return self._stairs_omni_reward_cache["stairs_omni_lead_touch_above"]
```

## _reward_stairs_omni_support_transfer_progress

[legged_gym/envs/go2/go2_parkour_env.py:1187](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1187)

```python
    def _reward_stairs_omni_support_transfer_progress(self):
        self._update_stairs_omni_reward_cache()
        return self._stairs_omni_reward_cache["stairs_omni_support_transfer_progress"]
```

## _reward_stairs_omni_trail_followup

[legged_gym/envs/go2/go2_parkour_env.py:1183](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1183)

```python
    def _reward_stairs_omni_trail_followup(self):
        self._update_stairs_omni_reward_cache()
        return self._stairs_omni_reward_cache["stairs_omni_trail_followup"]
```

## _reward_stand_still

[legged_gym/envs/go2/go2_parkour_env.py:1146](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1146)

```python
    def _reward_stand_still(self):
        reward = super()._reward_stand_still()
        if self.stand_still_terrain_ids.numel() > 0:
            reward = reward * self._terrain_id_mask(self.stand_still_terrain_ids).float()
        speed_search_mask = self._flat_speed_search_mask()
        if torch.any(speed_search_mask):
            reward[speed_search_mask] = 0.0
        return reward
```

## _reward_targeted_foothold_touchdown

[legged_gym/envs/go2/go2_parkour_env.py:1565](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1565)

```python
    def _reward_targeted_foothold_touchdown(self):
        feet_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        feet_pos = feet_states[:, :, 0:3]
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        prev_contact = self.targeted_foothold_last_contacts
        contact_filt = torch.logical_or(contact, prev_contact)
        swing_start = prev_contact & ~contact

        candidate_targets_world, candidate_valid = self._compute_targeted_foothold_candidates()
        need_new_target = swing_start | ((~contact_filt) & (~self.targeted_foothold_target_valid))
        update_mask = need_new_target & candidate_valid
        self.targeted_foothold_targets_world = torch.where(
            update_mask.unsqueeze(-1),
            candidate_targets_world,
            self.targeted_foothold_targets_world,
        )
        self.targeted_foothold_target_valid = torch.where(
            need_new_target,
            candidate_valid,
            self.targeted_foothold_target_valid,
        )

        first_contact = (
            self.targeted_foothold_air_time > self.cfg.rewards.targeted_foothold_min_air_time
        ) & contact_filt
        self.targeted_foothold_air_time += self.dt
        self.targeted_foothold_air_time *= ~contact_filt
        self.targeted_foothold_last_contacts = contact

        target_delta_xy = feet_pos[:, :, :2] - self.targeted_foothold_targets_world[:, :, :2]
        target_dist_sq = torch.sum(torch.square(target_delta_xy), dim=-1)
        foot_reward = torch.exp(
            -target_dist_sq / self.cfg.rewards.targeted_foothold_distance_sigma
        )
        foot_reward = foot_reward * first_contact.float() * self.targeted_foothold_target_valid.float()
        self.targeted_foothold_target_valid &= ~contact_filt
        self.targeted_foothold_targets_world = torch.where(
            self.targeted_foothold_target_valid.unsqueeze(-1),
            self.targeted_foothold_targets_world,
            torch.zeros_like(self.targeted_foothold_targets_world),
        )
        return foot_reward.sum(dim=1)
```

## _reward_torques

[legged_gym/envs/base/legged_robot.py:1552](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1552)

```python
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
```

## _reward_tracking_ang_vel

[legged_gym/envs/base/legged_robot.py:1627](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1627)

```python
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        if self.cfg.rewards.dynamic_sigma is None:
            sigma = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_ang_vel"]
            vmax = self.dynamic_sigma_cfg["max_ang_vel"]
            sigma = self._get_dynamic_sigma(torch.abs(self.commands[:, 2]), vmin, vmax)
        ang_vel_error_sq = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error_sq/sigma)
```

## _reward_tracking_lin_vel

[legged_gym/envs/go2/go2_parkour_env.py:1130](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1130)

```python
    def _reward_tracking_lin_vel(self):
        reward = super()._reward_tracking_lin_vel()
        speed_search_mask = self._flat_speed_search_mask()
        if torch.any(speed_search_mask):
            reward[speed_search_mask] = 0.0
        conditioned_mask = self._heading_conditioned_mask()
        if torch.any(conditioned_mask):
            heading_align = self._forward_heading_alignment()
            reward[conditioned_mask] = reward[conditioned_mask] * heading_align[conditioned_mask]

        full_reward_mask = self._terrain_id_mask(self.tracking_lin_vel_full_reward_terrain_ids)
        active_cmd = self.commands[:, 0] > 1.0e-6
        full_reward_mask &= active_cmd & (self.base_lin_vel[:, 0] >= self.commands[:, 0])
        reward[full_reward_mask] = 1.0
        return reward
```

## _reward_upward_foothold_clearance

[legged_gym/envs/go2/go2_parkour_env.py:1330](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_parkour_env.py:1330)

```python
    def _reward_upward_foothold_clearance(self):
        feet_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        feet_pos = feet_states[:, :, 0:3]
        feet_xy_speed = torch.norm(feet_states[:, :, 7:9], dim=-1)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.clearance_last_contacts)
        self.clearance_last_contacts = contact
        in_air = ~contact_filt
        self.clearance_air_time = torch.where(
            in_air,
            self.clearance_air_time + self.dt,
            torch.zeros_like(self.clearance_air_time),
        )

        current_ground = self._sample_heightfield(feet_pos[:, :, :2])
        foot_clearance = (feet_pos[:, :, 2] - current_ground).clamp(min=0.0)

        base_forward = quat_apply(
            self.base_quat,
            torch.tensor([[1.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1),
        )
        forward_xy = base_forward[:, :2]
        forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        omni_mask = self._omni_env_mask()
        if torch.any(omni_mask):
            world_x_forward = torch.zeros_like(forward_xy)
            world_x_forward[:, 0] = 1.0
            forward_xy = torch.where(omni_mask.unsqueeze(-1), world_x_forward, forward_xy)

        lookahead_distance = torch.full(
            (self.num_envs,),
            float(self.cfg.rewards.clearance_lookahead_distance),
            device=self.device,
            dtype=feet_pos.dtype,
        )
        omni_stairs_lookahead = float(
            getattr(
                self.cfg.rewards,
                "omni_stairs_clearance_lookahead_distance",
                self.cfg.rewards.clearance_lookahead_distance,
            )
        )
        if hasattr(self, "terrain_ids") and omni_stairs_lookahead != float(self.cfg.rewards.clearance_lookahead_distance):
            omni_stairs_mask = self._omni_env_mask() & (self.terrain_ids == 10)
            lookahead_distance = torch.where(
                omni_stairs_mask,
                torch.full_like(lookahead_distance, omni_stairs_lookahead),
                lookahead_distance,
            )

        lookahead_xy = feet_pos[:, :, :2] + forward_xy.unsqueeze(1) * lookahead_distance.view(-1, 1, 1)
        ahead_ground = self._sample_heightfield(lookahead_xy)
        terrain_rise = (ahead_ground - current_ground).clamp(min=0.0)
        #print(terrain_rise)
        cmd_x = self._forward_reward_command_x()
        active_cmd = (cmd_x > self.cfg.rewards.clearance_min_command_x).unsqueeze(-1)
        min_base_vel = torch.clamp(
            cmd_x * self.cfg.rewards.clearance_min_base_vel_ratio,
            min=self.cfg.rewards.clearance_min_base_vel,
        ).unsqueeze(-1)
        progressing = self._forward_reward_velocity_x().unsqueeze(-1) > min_base_vel
        min_foot_speed = torch.full_like(
            feet_xy_speed,
            float(self.cfg.rewards.clearance_min_foot_speed),
        )
        omni_stairs_mask = self._omni_env_mask() & (self.terrain_ids == 10) if hasattr(self, "terrain_ids") else self._omni_env_mask()
        if torch.any(omni_stairs_mask):
            min_foot_speed = torch.where(
                omni_stairs_mask.unsqueeze(-1),
                torch.full_like(
                    min_foot_speed,
                    float(getattr(self.cfg.rewards, "omni_stairs_clearance_min_foot_speed", self.cfg.rewards.clearance_min_foot_speed)),
                ),
                min_foot_speed,
            )
            if not bool(getattr(self.cfg.rewards, "omni_stairs_clearance_requires_progress", True)):
                progressing = torch.where(
                    omni_stairs_mask.unsqueeze(-1),
                    torch.ones_like(progressing),
                    progressing,
                )
        moving_feet = feet_xy_speed > min_foot_speed
        clearance_enabled = ~self._terrain_id_mask(self.clearance_excluded_terrain_ids).unsqueeze(-1)
        active_feet = active_cmd & progressing & moving_feet & in_air & clearance_enabled & (
            terrain_rise > self.cfg.rewards.clearance_active_step_height
        )
        desired_clearance = (
            self.cfg.rewards.clearance_base_height
            + self.cfg.rewards.clearance_rise_gain * terrain_rise
        ).clamp(max=self.cfg.rewards.clearance_max_height)
        clearance_error = foot_clearance - desired_clearance
        foot_reward = torch.exp(
            -torch.square(clearance_error) / self.cfg.rewards.clearance_tracking_sigma
        )
        overtime_ratio = torch.clamp(
            (self.clearance_air_time - self.cfg.rewards.clearance_max_air_time)
            / self.cfg.rewards.clearance_max_air_time,
            min=0.0,
            max=1.0,
        )
        foot_reward = foot_reward - self.cfg.rewards.clearance_overtime_penalty * overtime_ratio
        foot_reward = foot_reward * active_feet.float()
        #print(foot_reward[0])
        return foot_reward.sum(dim=1)
```

## _reward_correct_base_height

[legged_gym/envs/base/legged_robot.py:1690](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1690)

```python
    def _reward_correct_base_height(self):
        base_height = self._get_base_height()
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew
```

## _reward_feet_contact_forces

[legged_gym/envs/base/legged_robot.py:1660](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1660)

```python
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
```

## _reward_feet_slip

[legged_gym/envs/rm75/rm75_flat_trot_env.py:129](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:129)

```python
    def _reward_feet_slip(self):
        """Penalize horizontal foot motion while a foot is supporting the body."""
        contact_threshold = float(self.cfg.rewards.trot_contact_force_threshold)
        contacts = self.contact_forces[:, self.trot_feet_indices, 2] > contact_threshold
        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        feet_xy_velocity = body_states[:, self.trot_feet_indices, 7:9]
        return torch.sum(
            torch.sum(torch.square(feet_xy_velocity), dim=2) * contacts.float(),
            dim=1,
        )
```

## _reward_hip_to_default

[legged_gym/envs/go2/go2_env.py:55](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/go2/go2_env.py:55)

```python
    def _reward_hip_to_default(self):
        hip_dof_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint']
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        default_hip_pos = self.default_dof_pos[:, hip_dof_indices]
        return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)
```

## _reward_orientation

[legged_gym/envs/base/legged_robot.py:1527](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1527)

```python
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
```

## _reward_termination

[legged_gym/envs/base/legged_robot.py:1572](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1572)

```python
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
```

## _reward_torque_limits

[legged_gym/envs/base/legged_robot.py:1587](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1587)

```python
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
```

## _reward_tracking_lin_vel

[legged_gym/envs/base/legged_robot.py:1613](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/base/legged_robot.py:1613)

```python
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        if self.cfg.rewards.dynamic_sigma is None:
            sigma_x = sigma_y = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_lin_vel"]
            vmax = self.dynamic_sigma_cfg["max_lin_vel"]
            sigma_x = self._get_dynamic_sigma(torch.abs(self.commands[:, 0]), vmin, vmax)
            sigma_y = self._get_dynamic_sigma(torch.abs(self.commands[:, 1]), vmin, vmax)
        lin_vel_error_sq = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        scaled_error = lin_vel_error_sq[:, 0] / sigma_x + lin_vel_error_sq[:, 1] / sigma_y
        # print(f"{self.base_lin_vel[:, :2]=}, {lin_vel_error_sq=}")
        return torch.exp(-scaled_error)
```

## _reward_trot_gait

[legged_gym/envs/rm75/rm75_flat_trot_env.py:83](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:83)

```python
    def _reward_trot_gait(self):
        """Reward alternating diagonal support with balanced per-foot duty cycles.

        Foot order is FL, FR, RL, RR. A trot support phase therefore has either
        FL+RR or FR+RL in contact. The EMA term prevents the policy from keeping
        only one diagonal pair in stance indefinitely.
        """
        contact_threshold = float(self.cfg.rewards.trot_contact_force_threshold)
        contacts = self.contact_forces[:, self.trot_feet_indices, 2] > contact_threshold
        contact_filt = torch.logical_or(contacts, self.trot_last_contacts)
        self.trot_last_contacts[:] = contacts

        alpha = float(self.cfg.rewards.trot_contact_ema_alpha)
        self.trot_contact_ema.mul_(alpha).add_(contact_filt.float(), alpha=1.0 - alpha)

        contact = contact_filt.float()
        diagonal_a = 0.5 * (contact[:, 0] + contact[:, 3])  # FL + RR
        diagonal_b = 0.5 * (contact[:, 1] + contact[:, 2])  # FR + RL

        diagonal_sync_error = torch.abs(contact[:, 0] - contact[:, 3])
        diagonal_sync_error += torch.abs(contact[:, 1] - contact[:, 2])
        diagonal_sync_score = (1.0 - 0.5 * diagonal_sync_error).clamp(0.0, 1.0)
        diagonal_opposition_score = torch.abs(diagonal_a - diagonal_b)

        mean_duty = self.trot_contact_ema.mean(dim=1, keepdim=True)
        duty_spread = torch.abs(self.trot_contact_ema - mean_duty).mean(dim=1)
        balance_tolerance = float(self.cfg.rewards.trot_duty_balance_tolerance)
        duty_balance_score = (1.0 - duty_spread / balance_tolerance).clamp(0.0, 1.0)

        target_duty = float(self.cfg.rewards.trot_target_contact_duty)
        duty_sigma = float(self.cfg.rewards.trot_contact_duty_sigma)
        target_duty_score = torch.exp(
            -torch.square(mean_duty.squeeze(1) - target_duty) / duty_sigma
        )

        active_command = self.commands[:, 0] > float(
            self.cfg.rewards.trot_min_command_speed
        )
        return (
            diagonal_sync_score
            * diagonal_opposition_score
            * duty_balance_score
            * target_duty_score
            * active_command.float()
        )
```

## _reward_downhill_overspeed

[legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:341)

```python
    def _reward_downhill_overspeed(self):
        self._update_terrain_profile()
        downhill = self.local_terrain_slope < -math.tan(math.radians(2.0))
        overspeed = torch.clamp(
            self.terrain_tangent_speed - self.terrain_target_speed,
            min=0.0,
        )
        return torch.square(overspeed) * downhill.float()
```

## _reward_feet_slip

[legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:397)

```python
    def _reward_feet_slip(self):
        contact_threshold = float(self.cfg.rewards.trot_contact_force_threshold)
        contacts = self.contact_forces[:, self.trot_feet_indices, 2] > contact_threshold
        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        feet_world_velocity = body_states[:, self.trot_feet_indices, 7:10]
        return torch.sum(
            torch.sum(torch.square(feet_world_velocity), dim=2) * contacts.float(),
            dim=1,
        )
```

## _reward_lateral_velocity

[legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:334)

```python
    def _reward_lateral_velocity(self):
        return torch.square(self.root_states[:, 8])
```

## _reward_lin_vel_z

[legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:325)

```python
    def _reward_lin_vel_z(self):
        self._update_terrain_profile()
        return torch.square(self.terrain_normal_velocity)
```

## _reward_orientation

[legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:329)

```python
    def _reward_orientation(self):
        self._update_terrain_profile()
        normal_body = quat_rotate_inverse(self.base_quat, self.terrain_normal_world)
        return torch.sum(torch.square(normal_body[:, :2]), dim=1)
```

## _reward_speed_tracking_error

[legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:337)

```python
    def _reward_speed_tracking_error(self):
        self._update_terrain_profile()
        return torch.square(self.terrain_target_speed - self.terrain_tangent_speed)
```

## _reward_tracking_lin_vel

[legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:315)

```python
    def _reward_tracking_lin_vel(self):
        self._update_terrain_profile()
        tangent_error = torch.square(
            self.terrain_target_speed - self.terrain_tangent_speed
        )
        lateral_error = torch.square(self.root_states[:, 8])
        return torch.exp(
            -(tangent_error + lateral_error) / float(self.cfg.rewards.tracking_sigma)
        )
```

## _reward_trot_gait

[legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_visual_ramp_trot_env.py:350)

```python
    def _reward_trot_gait(self):
        self._update_terrain_profile()
        contact_threshold = float(self.cfg.rewards.trot_contact_force_threshold)
        contacts = self.contact_forces[:, self.trot_feet_indices, 2] > contact_threshold
        contact_filt = torch.logical_or(contacts, self.trot_last_contacts)
        self.trot_last_contacts[:] = contacts

        alpha = float(self.cfg.rewards.trot_contact_ema_alpha)
        self.trot_contact_ema.mul_(alpha).add_(contact_filt.float(), alpha=1.0 - alpha)

        contact = contact_filt.float()
        diagonal_a = 0.5 * (contact[:, 0] + contact[:, 3])
        diagonal_b = 0.5 * (contact[:, 1] + contact[:, 2])
        diagonal_sync_error = torch.abs(contact[:, 0] - contact[:, 3])
        diagonal_sync_error += torch.abs(contact[:, 1] - contact[:, 2])
        diagonal_sync_score = (1.0 - 0.5 * diagonal_sync_error).clamp(0.0, 1.0)
        diagonal_opposition_score = torch.abs(diagonal_a - diagonal_b)

        mean_duty = self.trot_contact_ema.mean(dim=1, keepdim=True)
        duty_spread = torch.abs(self.trot_contact_ema - mean_duty).mean(dim=1)
        duty_balance_score = (
            1.0
            - duty_spread / float(self.cfg.rewards.trot_duty_balance_tolerance)
        ).clamp(0.0, 1.0)

        flat_duty = float(self.cfg.rewards.trot_flat_contact_duty)
        uphill_duty = float(self.cfg.rewards.trot_uphill_contact_duty)
        downhill_duty = float(self.cfg.rewards.trot_downhill_contact_duty)
        slope_duty = torch.where(
            self.local_terrain_slope >= 0.0,
            torch.full_like(self.local_terrain_slope, uphill_duty),
            torch.full_like(self.local_terrain_slope, downhill_duty),
        )
        target_duty = flat_duty + self.visual_slope_factor * (slope_duty - flat_duty)
        target_duty_score = torch.exp(
            -torch.square(mean_duty.squeeze(1) - target_duty)
            / float(self.cfg.rewards.trot_contact_duty_sigma)
        )
        active = self.terrain_target_speed > float(self.cfg.rewards.trot_min_command_speed)
        return (
            diagonal_sync_score
            * diagonal_opposition_score
            * duty_balance_score
            * target_duty_score
            * active.float()
        )
```

## _reward_trot_duty_balance

[legged_gym/envs/rm75/rm75_flat_trot_env.py:140](D:/codex_workplace/月面四足强化学习/rm75-trot-slope-rl/legged_gym/envs/rm75/rm75_flat_trot_env.py:140)

```python
    def _reward_trot_duty_balance(self):
        """Penalize each foot's duty error relative to the configured target.

        This is intentionally separate from the multiplicative trot score.  A
        policy can otherwise keep a clean diagonal rhythm while consistently
        shortening one diagonal pair, which produces the lateral wobble seen in
        the hip84 policy.
        """
        target_duty = float(self.cfg.rewards.trot_target_contact_duty)
        return torch.mean(
            torch.square(self.trot_contact_ema - target_duty), dim=1
        )
```
