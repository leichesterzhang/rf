# Parkour MoE 当前框架设计更新版

> 基于 `doc/parkour_moe_task_design_zh.md` 更新，不覆盖原文件。  
> 更新时间：2026-05-13。  
> 依据当前工作区代码整理，重点描述当前真正生效的训练、网络、部署链路。

## 1. 文档范围

本文档整理当前 `go2_parkour_moe` 框架的完整设计，包括：

1. 任务目标与环境设计
2. 地形课程、指令、reset 与终止条件
3. 奖励函数与随机化
4. Estimator-Actor-Critic 网络结构
5. PPO 与 estimator 联合训练流程
6. MuJoCo / play 推理与可视化部署链路
7. 当前实现中需要注意的有效项与预留项

主要对应文件：

- `legged_gym/envs/go2/go2_config_parkour_moe.py`
- `legged_gym/envs/go2/go2_parkour_env.py`
- `legged_gym/envs/go2/go2_config_vanilla.py`
- `legged_gym/envs/base/legged_robot_config.py`
- `legged_gym/utils/terrain_mgdp.py`
- `rsl_rl/rsl_rl/modules/actor_critic_parkour_moe.py`
- `rsl_rl/rsl_rl/algorithms/ppo_parkour_moe.py`
- `rsl_rl/rsl_rl/runners/onpolicyrunner_parkour_moe.py`
- `deploy/deploy_mujoco/deploy_go2.py`
- `deploy/deploy_mujoco/configs/go2_parkour_moe.yaml`
- `legged_gym/scripts/play.py`
- `legged_gym/scripts/evaluate_parkour_moe.py`

## 2. 一句话总览

当前框架是一个面向 Go2 极限通行任务的视觉-本体融合强化学习系统：

```text
proxy depth + proprio history + vision flag
    -> token encoder + Transformer
    -> terrain selector + fall recovery selector
    -> 3 个 Read/Write MoE experts + shared_state
    -> mcp_code
    -> actor
    -> 12 维关节位置残差动作

privileged_obs + vision flag
    -> critic
    -> value
```

核心设计意图：

- 用代理深度图和本体历史构建可部署的中间控制码 `mcp_code`。
- 用特权观测只训练 critic 和 estimator 监督目标，不直接喂给 actor。
- 用 `terrain_id`、`fall_recovery` 和 MoE gate 让网络显式区分地形模式、恢复模式和视觉可用性。
- 用 `shared_state + Read/WriteExpert` 替代完全独立的专家记忆，降低专家切换时的表示跳变。
- 用 token confidence、token dropout、vision mask 训练视觉鲁棒性。

## 3. 当前关键配置快照

| 项目 | 当前值 |
| --- | --- |
| 机器人 | Go2 |
| 并行环境数 | `4096` |
| 动作维度 | `12` |
| actor 当前观测 | `45` |
| critic 特权观测 | `269` |
| episode 时长 | `20 s` |
| 仿真步长 | `sim.dt = 0.005 s` |
| 控制 decimation | `4` |
| 策略周期 | `0.02 s` |
| 地形风格 | `mgdp_parkour` |
| 地形网格 | `10 x 45` |
| 单块尺寸 | `10.0 m x 4.0 m` |
| 高度扫描 | `17 x 11 = 187` 点 |
| 深度来源 | proxy raycast depth |
| 深度缓存 | `1` 帧 |
| 深度输入 | `1 x 58 x 87` |
| Estimator 专家数 | `3` |
| Transformer | `1` 层，`4` heads，token dim `64` |
| 视觉 token | `8 x 11 = 88` |
| 本体历史 token | `10` |
| `mcp_code` | `68` 维 |
| actor 实际 MCP 输入 | `68 + vision_flag(1) = 69` 维 |
| actor 总输入 | `69 + obs_now(45) = 114` 维 |
| critic 总输入 | `269 + vision_flag(1) = 270` 维 |

## 4. 环境与任务设计

### 4.1 任务能力目标

当前任务覆盖三类能力：

1. 障碍通行能力：gap、石块、窄桥、悬空梁、悬空平台、hurdle、坡面、楼梯、通道。
2. 简单地形运动能力：在 ramp、stairs、flat 等地形上支持一部分全向/随机 yaw 运动。
3. 恢复能力：flat 地形上启用翻身初始化，训练 fall recovery 分支。

### 4.2 地形类型与采样比例

当前 `terrain_proportions` 为：

```python
[0.15, 0.0, 0.05, 0.1, 0.1, 0.1, 0.05, 0.15, 0.05, 0.05, 0.1, 0.1]
```

对应 12 类 MGDP parkour 地形：

| id | 名称 | 比例 | 当前采样 | 主要训练意图 |
| ---: | --- | ---: | --- | --- |
| 0 | `single_gap` | `0.15` | 是 | 跨沟、速度保持 |
| 1 | `step_stone` | `0.00` | 否 | 离散踏点，当前不参与采样 |
| 2 | `two_row_stones` | `0.05` | 是 | 双排落脚点选择 |
| 3 | `one_row_stones` | `0.10` | 是 | 单排落脚点选择 |
| 4 | `single_bridge` | `0.10` | 是 | 窄桥、收窄步态 |
| 5 | `air_beams` | `0.10` | 是 | 悬空梁通行 |
| 6 | `air_stones` | `0.05` | 是 | 低净空/悬空平台 |
| 7 | `hurdle` | `0.15` | 是 | 跨越障碍 |
| 8 | `ramp` | `0.05` | 是 | 坡面与全向运动 |
| 9 | `corridor` | `0.05` | 是 | 窄通道 |
| 10 | `stairs_up` | `0.10` | 是 | 上台阶与全向运动 |
| 11 | `flat` | `0.10` | 是 | 平地、全向运动、翻身恢复 |

### 4.3 课程难度

地形课程使用 `num_rows = 10`，原始 difficulty 为：

```text
0.0, 0.1, 0.2, ..., 0.9
```

部分几何生成函数会使用：

```text
normalized_difficulty = difficulty / 0.9
```

课程更新规则：

- 若机器人相对 reset origin 的平面距离大于 `terrain_length / 2 = 5 m`，升级。
- 若移动距离小于期望指令距离的一半，降级。
- 等级裁剪在 `[0, 9]`。
- 初始最大地形等级为 `2`。

### 4.4 地形几何摘要

| 地形 | 难度变化重点 |
| --- | --- |
| `single_gap` | gap 宽度从约 `0.1 m` 增至 `1.0 m`，深度 `0.6 m` |
| `step_stone` | 石块尺寸减小、间距和高度扰动增大，但当前比例为 `0` |
| `two_row_stones` | 双排石块尺寸随难度减小，间距和高度增大 |
| `one_row_stones` | 单排石块 x 向 gap 由 `0.1 m` 到 `0.35 m` |
| `single_bridge` | 桥面从宽桥逐渐变窄 |
| `air_beams` | 高度场提供基底，额外 trimesh 生成悬空梁 |
| `air_stones` | 高度场提供基底，额外 trimesh 生成悬空平台 |
| `hurdle` | hurdle 高度和踏点难度一起随 difficulty 增大 |
| `ramp` | 最大坡角 `30 deg` |
| `corridor` | 通道半宽从约 `0.4 m` 缩到 `0.2 m` |
| `stairs_up` | 台阶高度从 `0.05 m` 增至 `0.25 m` |
| `flat` | 无额外几何难度 |

### 4.5 指令设计

普通 parkour 地形的指令范围：

| 指令 | 范围 |
| --- | --- |
| `lin_vel_x` | `[0.0, 1.5]` |
| `lin_vel_y` | `[0.0, 0.0]` |
| `ang_vel_yaw` | `[-1.0, 1.0]` |
| `heading` | `[0.0, 0.0]` |

因此普通障碍任务本质是沿世界 `+x` 前进，并通过 heading control 抑制偏航。

当前额外启用了 omni 机制：

- `omni_enabled = True`
- `omni_terrain_ids = [8, 10, 11]`，即 ramp、stairs_up、flat
- `omni_ratio = 0.5`
- `omni_blind = True`
- `omni_random_yaw = True`
- `omni_yaw_range = [-pi, pi]`

这意味着在简单地形中，一部分环境会随机 yaw 初始化，并用 body-frame 速度方向运动；这些 omni 环境在 runner 中会被强制 `mask_vision=False`，用于训练纯本体运动分支。

### 4.6 Reset 与翻身恢复

当前 `turn_over = True`，但只作用在：

```python
turn_over_terrain_ids = [11]  # flat
```

翻身初始化比例：

| 类型 | 比例 | 高度范围 | roll 范围 |
| --- | ---: | --- | --- |
| backflip | `0.4` | `[0.20, 0.21]` | `[pi/2, pi]` |
| sideflip | `0.4` | `[0.34, 0.35]` | `[0, pi/2]` |
| noflip | `0.2` | 常规站立 | 水平 |

普通 reset：

- 放回 terrain 的 `reset_origin`
- 起始平台长宽均为 `2.0 m`
- x 抖动 `0.15 m`
- y 抖动 `0.0 m`
- 初始姿态水平朝前
- base 线速度和角速度随机到 `[-0.5, 0.5]`

omni reset 会额外随机 yaw，并设置 body command direction。

### 4.7 Proxy 深度相机

当前相机不使用 Isaac Gym 相机传感器，而是强制使用 proxy raycast：

| 项目 | 当前值 |
| --- | --- |
| source | `proxy` |
| backend | `warp` 优先，否则 torch |
| 更新间隔 | `5` env steps |
| buffer_len | `1` |
| raw shape | `60 x 106` |
| crop 后输出 | `58 x 87` |
| clipping range | `2.0 m` |
| 相机位置 | `[0.33, 0.0, 0.08]` |
| 相机姿态 | ROS convention，向下约 `65 deg` |
| 位姿误差 | xyz `[-0.01, 0.01]`，yaw `[-5, 5] deg` |
| 深度高斯噪声 | `0.05` |
| 像素 dropout | `0.0` |

深度归一化为：

```text
depth_norm = depth / clipping_range - 0.5
clip to [-0.5, 0.5]
```

## 5. 观测与监督目标

### 5.1 Actor 观测 `obs_now`

actor 当前观测为 `45` 维：

| 组成 | 维度 |
| --- | ---: |
| `base_ang_vel` | 3 |
| `projected_gravity` | 3 |
| `commands[:3]` | 3 |
| `dof_pos - default_dof_pos` | 12 |
| `dof_vel` | 12 |
| `previous actions` | 12 |
| 合计 | 45 |

### 5.2 Critic 特权观测 `privileged_obs`

当前 critic 特权观测为 `269` 维：

| 组成 | 维度 | 说明 |
| --- | ---: | --- |
| base privileged obs | 263 | 继承 Go2：速度、重力、指令、关节、接触力、力矩、加速度、高度扫描 |
| `feet_heights` | 4 | 四足相对地面高度 |
| `terrain_id` | 1 | 当前地形 id |
| `fall_recovery` | 1 | 当前是否处于翻身恢复状态 |
| 合计 | 269 |  |

Estimator 监督目标从 `critic_obs` 中切片：

- `gt_v_t`: base 线速度，3 维
- `gt_m_t`: 高度扫描图，187 维
- `gt_h_tf`: 四足高度，4 维
- `gt_terrain_id`: 地形 id，1 维
- `gt_fall_recovery`: 翻身恢复标签，1 维
- `gt_next_obs`: 下一步 actor 观测，45 维

## 6. 奖励设计

### 6.1 总体形式

每步 reward 可概括为：

```text
r_t = sum_i scale_i * dt * curriculum_i(iter) * raw_reward_i
```

当前 `only_positive_rewards = False`，所以总奖励不会被截断为非负。

### 6.2 基础运动奖励

| reward | scale | 作用 |
| --- | ---: | --- |
| `tracking_lin_vel` | `1.0` | 跟踪平面速度，普通 parkour 上乘 heading alignment |
| `tracking_ang_vel` | `0.5` | 跟踪 yaw 角速度 |
| `lin_vel_z` | `-2.0` | 抑制竖直速度，0 到 1500 iter 衰减到 0 |
| `ang_vel_xy` | `-0.05` | 抑制 roll/pitch 角速度 |
| `dof_acc` | `-2.5e-7` | 抑制关节加速度 |
| `dof_power` | `-2e-5` | 抑制功耗 |
| `torques` | `-1e-4` | 抑制大力矩 |
| `correct_base_height` | `-1.0` | 约束 base 高度，0 到 5000 iter 放大到 10 倍 |
| `action_rate` | `-0.01` | 抑制动作一阶变化 |
| `action_smoothness` | `-0.01` | 抑制动作二阶变化 |
| `collision` | `-1.0` | 惩罚 thigh/calf/base/Head_lower 接触 |
| `dof_pos_limits` | `-2.0` | 惩罚关节软限位 |
| `feet_regulation` | `-0.05` | 抑制贴地高速扫腿 |
| `hip_to_default` | `-0.05` | 约束 hip 不长期偏离默认姿态 |
| `stand_still` | `-0.5` | 零指令时约束站立 |

### 6.3 Parkour 特化奖励

| reward | scale | 生效范围 | 作用 |
| --- | ---: | --- | --- |
| `low_speed_when_commanded` | `-0.5` | 有前向指令时 | 前进太慢惩罚 |
| `hard_terrain_yaw_penalty` | `-0.2` | 非 omni 且 heading conditioned | 抑制障碍地形偏航 |
| `omni_target_yaw_error` | `-0.05` | omni env | 约束随机 yaw 目标 |
| `upward_foothold_clearance` | `0.5` | 排除 gap、air_stones、corridor、flat | 前方抬升时鼓励摆动腿清障 |
| `targeted_foothold_touchdown` | `0.5` | step_stone、two/one row、air_beams、hurdle、ramp、stairs_up | 首次落地接近候选落脚点 |
| `single_bridge_narrow_stance` | `0.5` | single_bridge | 鼓励窄桥上收窄左右脚横向距离 |
| `flat_gait` | `-0.5` | flat | 惩罚平地过早离地、接触占空比不足、后腿失衡 |
| `upright` | `1.0` | 翻身奖励 | 鼓励恢复 upright |
| `turn_over_base_height` | `-1.0` | 翻身奖励 | 翻身期间约束 base 高度 |

### 6.4 重要 reward mask

- base height 奖励排除：`single_gap`, `two_row_stones`, `one_row_stones`, `air_beams`, `air_stones`, `hurdle`
- `hip_to_default` 排除：`single_bridge`
- `target_posture` 只在 `flat` 上生效
- `upward_foothold_clearance` 排除：`single_gap`, `air_stones`, `corridor`, `flat`
- `targeted_foothold_touchdown` 生效：`step_stone`, `two_row_stones`, `one_row_stones`, `air_beams`, `hurdle`, `ramp`, `stairs_up`

## 7. 终止条件

| 终止项 | 条件 |
| --- | --- |
| `time_out` | episode 超过 `20 s` |
| `base_contact_upside_down` | base 接触且机身倒置 |
| `fall_base_height_cutoff` | base 高度低于 `-0.1 m` |
| `fall_foot_height_cutoff` | 足端最低高度低于阈值，`single_gap` 放宽到 `-0.3 m` |
| `no_progress` | 有前向指令但长时间无前进，普通 `2 s`，omni `4 s` |
| `out_of_block` | 跑出当前 terrain block，margin `0.8 m` |

当前 `no_progress_excluded_terrain_ids = [11]`，flat 不做无进度终止。

## 8. 随机化设计

### 8.1 地形随机化

- 12 类地形按列采样，10 个难度等级。
- 石块、hurdle、高度、间距等几何参数随 difficulty 和随机数变化。
- `mgdp_add_roughness = False`，当前不额外叠加 roughness。
- `air_beams` 和 `air_stones` 使用额外 trimesh。

### 8.2 动力学与控制随机化

| 项目 | 范围 |
| --- | --- |
| friction | `[0.0, 2.0]` |
| restitution | `[0.0, 0.5]` |
| base mass | `[-1.0, 1.0] kg` |
| link mass multiplier | `[0.9, 1.1]` |
| base COM offset | `[-0.03, 0.03] m` |
| PD stiffness multiplier | `[0.9, 1.1]` |
| PD damping multiplier | `[0.9, 1.1]` |
| motor zero offset | `[-0.035, 0.035] rad` |
| motor strength | `[0.8, 1.2]` |
| push linear velocity | `[-0.4, 0.4] m/s` |
| push angular velocity | `[-0.6, 0.6] rad/s` |
| action delay | 0 到 20 ms 级随机延迟 |

### 8.3 观测与视觉随机化

- actor 观测噪声开启。
- proxy depth 有相机位姿误差、深度高斯噪声。
- estimator 训练时对视觉 token 做随机 dropout：

```text
token_dropout_min = 0.00
token_dropout_max = 0.30
token_dropout_mode = spatial_block
```

- 当前默认不是对 `88` 个 token 做等概率独立 Bernoulli 丢弃，而是先在 token grid 上按空间先验采样若干连通块，再把这些块里的 token 置为 padding。
- 空间先验默认更偏向：
  - 图像下部：模拟近场地面反光、贴地噪声、溅水/污渍
  - 图像上部：模拟远处过曝、天空边缘、深度失真
- 因此每次更常见的是“一片区域一起坏掉”，而不是离散坏点。

- runner 对 easy terrain 进行视觉开关训练：

```text
easy_terrain_names = ["ramp", "stairs_up", "air_stones", "flat"]
vision_toggle_interval = 20
```

- omni env 因 `omni_blind=True`，始终强制 `mask_vision=False`。

## 9. 网络设计

### 9.1 完全展开的数据流总图

当前一次 estimator 前向可以完整写成：

```text
obs_now [B,45]
proprio_hist_flat [B,450]
depth_camera [B,1,58,87]
mask_vision [B]
token_padding_mask(optional)
    |
    | reshape history
    v
proprio_seq [B,10,45]
    |
    +---------------- visual branch ----------------+
    |                                               |
    | depth -> image_encoder                        |
    |   [B,1,58,87]                                 |
    |   -> Conv(1,16,3,2,1) + ELU                  |
    |   -> Conv(16,32,3,2,1) + ELU                 |
    |   -> Conv(32,64,3,2,1) + ELU                 |
    |   -> image_features [B,64,8,11]              |
    |   -> flatten -> image_tokens [B,1,88,64]     |
    |   -> LayerNorm                               |
    |                                               |
    | local patch reconstruction                    |
    |   -> local_patch_targets [B,1,88,225]        |
    |   -> local_patch_recon [B,1,88,225]          |
    |   -> context_patch_recon [B,1,88,225]        |
    |   -> token_recon_loss [B,1,88]               |
    |   -> token_confidence [B,1,88]               |
    |                                               |
    +---------------- proprio branch --------------+
    |                                               |
    | proprio_seq [B,10,45]                         |
    |   -> Linear 45->256 + ELU                    |
    |   -> Linear 256->128 + ELU                   |
    |   -> Linear 128->64                          |
    |   -> proprio_tokens [B,10,64]                |
    |   -> LayerNorm                               |
    |                                               |
    +---------------- mask / encoding -------------+
    |                                               |
    | training_token_dropout_mask [B,1,88]         |
    | external_token_padding_mask [B,1,88]         |
    | image_padding_mask [B,1,88]                  |
    | vision_token [B,1,64]                        |
    | add modality encoding                        |
    | add image 2D pos encoding                    |
    | add frame 1D pos encoding                    |
    | add proprio 1D pos encoding                  |
    |                                               |
    +---------------- transformer input -----------+
    |                                               |
    | image_tokens reshape -> [B,88,64]            |
    | proprio_tokens -> [B,10,64]                  |
    | vision_token -> [B,1,64]                     |
    | concat -> transformer_input [B,99,64]        |
    | concat padding -> transformer_mask [B,99]    |
    |                                               |
    +---------------- transformer -----------------+
    |                                               |
    | TransformerEncoder(1 layer, 4 heads)         |
    | -> encoded_tokens [B,99,64]                  |
    | -> slice proprio token range                 |
    | -> fused_tokens [B,10,64]                    |
    |                                               |
    +---------------- selectors -------------------+
    |                                               |
    | shared_pool_query attention pool             |
    | -> selector_feature [B,64]                   |
    | -> terrain_logits [B,12]                     |
    | -> terrain_probs [B,12]                      |
    | -> terrain_pred [B]                          |
    | -> fall_recovery_logits [B,1]                |
    | -> fall_recovery_prob [B,1]                  |
    |                                               |
    +---------------- expert features -------------+
    |                                               |
    | expert_pool_queries attention pool           |
    | -> expert_pooled_features [B,3,64]           |
    | terrain_token_encoder(12->64->64)            |
    | fall_recovery_encoder(1->64->64)             |
    | -> 加到每个 expert feature 上                |
    |                                               |
    +---------------- router ----------------------+
    |                                               |
    | context_feature = shared_head(selector)      |
    | terrain_router_feature [B,32]                |
    | fall_router_feature [B,16]                   |
    | concat(obs_now, vision_flag, context, terrain, fall)
    | -> [B,158]
    | -> router 158->128->3->softmax               |
    | -> gating_weights [B,3]                      |
    |                                               |
    +---------------- motion branch ---------------+
    |                                               |
    | motion_gru(fused_tokens [B,10,64])           |
    | -> motion_hidden [B,64]                      |
    | -> motion_prediction_head [B,64]             |
    | -> v_t [B,3]                                 |
    | -> h_tf [B,4]                                |
    | -> motion_z_mu [B,16]                        |
    | -> motion_z_logvar [B,16]                    |
    | -> motion_z_t [B,16]                         |
    | -> decoder_obs([v_t,h_tf,motion_z_t])        |
    | -> o_hat [B,45]                              |
    |                                               |
    +---------------- terrain / MoE branch --------+
    |                                               |
    | for each expert i in {0,1,2}:                |
    |   read(shared_state_prev) -> hidden_init_i   |
    |   GRUCell(expert_feature_i, hidden_init_i)   |
    |   -> hidden_i [B,64]                         |
    |   write(hidden_i) -> shared_delta_i [B,64]   |
    |                                               |
    | stack hidden -> [B,3,64]                     |
    | stack delta -> [B,3,64]                      |
    | weighted sum by gate                         |
    | -> mixed_hidden [B,64]                       |
    | -> mixed_delta [B,64]                        |
    |                                               |
    | shared_state update:                         |
    | concat(shared_prev, mixed_delta) [B,128]     |
    | -> candidate [B,64]                          |
    | -> gate [B,64]                               |
    | -> shared_state [B,64]                       |
    | -> LayerNorm                                 |
    |                                               |
    | terrain_prediction_head([mixed_hidden,shared_state])
    | -> [B,64]
    | -> z_mu [B,16]                               |
    | -> z_logvar [B,16]                           |
    | -> terrain_z_t [B,16]                        |
    | -> z_tm [B,32]                               |
    | -> terrain_decoder_obs(z_t) -> terrain_o_hat [B,45]
    | -> decoder_map(z_tm) -> m_hat [B,187]
    |                                               |
    +---------------- final outputs ---------------+
    |                                               |
    | terrain_pred_onehot [B,12]                   |
    | fall_recovery_pred [B,1]                     |
    | mcp_code = concat(                           |
    |   v_t(3), h_tf(4), z_mu(16), z_tm(32),       |
    |   terrain_onehot(12), fall_pred(1)           |
    | ) -> [B,68]                                  |
    | runner append vision_flag(1)                 |
    | -> actor_mcp_code [B,69]                     |
```

这条链路里最关键的几点是：

- 视觉 token 最终不是直接 pooled 成一个 image embedding，而是先通过 self-attention 注入 `10` 个 proprio token，再从这 10 个融合 token 上做 selector、motion branch 和 MoE。
- `terrain_pred` 与 `fall_recovery_pred` 不是旁路日志，它们显式进入 expert feature、router 和最终 `mcp_code`。
- 专家不是维护三套完全独立的长期 memory，而是通过 `shared_state` 做跨专家状态连续化。

### 9.2 Estimator 输入

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `obs_now` | `[B, 45]` | 当前 actor 观测 |
| `proprio_hist` | `[B, 450]` -> `[B, 10, 45]` | 10 帧本体历史 |
| `depth_camera` | `[B, 1, 58, 87]` | 1 帧 proxy depth |
| `mask_vision` | `[B]` | 是否允许使用视觉 token |
| `token_padding_mask` | 可选 | 部署/推理时外部传入坏 token mask |

### 9.3 Token 编码

视觉分支：

```text
depth [B,1,58,87]
  -> Conv2d 1->16 stride2
  -> Conv2d 16->32 stride2
  -> Conv2d 32->64 stride2
  -> feature [B,64,8,11]
  -> 88 visual tokens, dim=64
```

本体分支：

```text
10 x obs_now(45)
  -> MLP 45->256->128->64
  -> 10 proprio tokens, dim=64
```

辅助 token：

- `vision_flag` token：1 个
- modality encoding：区分视觉、本体、flag、terrain 类型
- 2D sin-cos encoding：视觉 token 位置
- 1D sin-cos encoding：历史时间位置

Transformer 输入 token 数：

```text
88 visual + 10 proprio + 1 vision flag = 99 tokens
```

当前后续 pooling 使用 Transformer 编码后的 `10` 个 proprio token slice。视觉信息通过 self-attention 注入这些 proprio token。

### 9.4 Token confidence 与局部重建

Estimator core 会对视觉 token 做局部 patch 重建：

- `local_patch_size = 15`
- `local_patch_stride = 8`
- `context_recon_loss_weight = 0.5`

每个视觉 token 得到：

```text
token_recon_loss = self_patch_loss + 0.5 * context_patch_loss
token_confidence = exp(-token_recon_loss / temperature)
```

训练时：

- token reconstruction loss 进入 estimator loss
- training token dropout 随机屏蔽视觉 token

部署时：

- MuJoCo adapter 可用 token confidence 构建 token padding mask
- depth preview 可叠加显示 bad token / random dropout / grid

### 9.5 Transformer 后的中间表示

Transformer 输出：

```text
encoded_tokens [B,99,64]
```

其中 token 排列顺序是：

```text
[88 visual tokens] + [10 proprio tokens] + [1 vision flag token]
```

但当前 estimator 在下游直接切出：

```text
fused_tokens = encoded_tokens[:, proprio_token_start:proprio_token_end, :]
            = [B,10,64]
```

也就是：

- 下游 selector / motion / expert pooling 都建立在这 `10` 个已经吸收了视觉上下文的 proprio tokens 上。
- 视觉 token 本身不再单独直接进入 selector head 或 expert head，而是通过 Transformer 影响这 10 个 proprio tokens。

这一步是当前实现和“把视觉 token 单独池化成一个 image embedding”非常不同的地方。

### 9.6 Terrain selector 与 fall recovery selector

Transformer 后的 proprio token 经 attention pool 得到 selector feature：

```text
selector_feature [B,64]
  -> terrain_logits [B,12]
  -> terrain_probs / terrain_pred
  -> fall_recovery_logits [B,1]
  -> fall_recovery_prob / pred
```

这两个 selector 同时服务于：

- estimator 监督损失
- expert feature 注入
- router 输入
- `mcp_code` 的离散状态标识

### 9.7 3 专家 Read/Write MoE

当前专家数为 `3`。

每个 expert 都是：

```text
shared_state_prev
  -> read_i()
  -> GRUCell initial hidden

expert_feature_i
  -> GRUCell
  -> hidden_i
  -> write_i()
  -> shared_delta_i
```

router 输入：

```text
obs_now(45)
+ vision_flag(1)
+ context_feature(64)
+ terrain_feature(32)
+ fall_recovery_feature(16)
= 158
```

router 输出：

```text
gating_weights [B,3]
```

共享状态更新：

```text
mixed_delta = sum_i gate_i * shared_delta_i
shared_state = LayerNorm(
    gate * candidate(shared_state_prev, mixed_delta)
    + (1-gate) * shared_state_prev
)
```

相比完全独立的专家 hidden，这个设计让专家通过公共记忆空间交换状态，有利于视觉开关、地形切换和 fall recovery 切换时保持连续性。

### 9.8 Motion branch 与 Terrain branch

Estimator 内部实际分成两条预测分支：

1. Motion branch：
   - `motion_gru` 读取 10 个 proprio token
   - 输出 `v_t`, `h_tf`, `motion_z_mu/logvar`
   - 用于重建下一步 obs 的 motion 部分

2. Terrain branch：
   - 由 MoE expert hidden 与 `shared_state` 组成
   - 输出 `z_mu/logvar`, `z_tm`
   - 用于地形 latent、地图重建和 actor 控制上下文

### 9.9 `mcp_code` 组成

当前 estimator 输出给 actor 的 `mcp_code` 为 `68` 维：

| 片段 | 维度 |
| --- | ---: |
| `v_t` | 3 |
| `h_tf` | 4 |
| `z_mu` | 16 |
| `z_tm` | 32 |
| `terrain_pred_onehot` | 12 |
| `fall_recovery_pred` | 1 |
| 合计 | 68 |

runner 会再拼接：

```text
vision_flag(1)
```

所以 actor 实际收到的 MCP 部分是 `69` 维。

### 9.10 Actor-Critic

Actor 输入的完整展开：

```text
estimator mcp_code [B,68]
+ vision_flag [B,1]
= actor_mcp_code [B,69]

actor_input = concat(actor_mcp_code, obs_now)
            = [B,69] + [B,45]
            = [B,114]
```

Actor MLP：

```text
Linear 114 -> 512
ELU
Linear 512 -> 256
ELU
Linear 256 -> 128
ELU
Linear 128 -> 12
```

输出：

```text
action_mean [B,12]
```

训练时：

```text
distribution = Normal(mean, std)
action ~ distribution.sample()
```

推理时：

```text
action = action_mean
```

Critic 输入的完整展开：

```text
critic_obs [B,269]
+ vision_flag [B,1]
= critic_input [B,270]
```

Critic MLP：

```text
Linear 270 -> 512
ELU
Linear 512 -> 256
ELU
Linear 256 -> 128
ELU
Linear 128 -> 1
```

输出：

```text
value [B,1]
```

### 9.11 推理时的数据流差异

训练时和推理时的网络主结构相同，但有 4 个关键差异：

1. 训练时 `token_dropout_mask` 来自 `build_training_token_dropout_mask()`，推理时默认不启用训练 dropout。
2. 推理时可以由 runner 或 MuJoCo adapter 额外传入 `external token_padding_mask`。
3. 推理时 actor 不采样高斯动作，而是直接取均值。
4. play / MuJoCo 推理路径都会缓存中间调试信息：

```text
gating_weights
terrain_probs
terrain_pred
token_confidence
token_dropout_mask
image_height / image_width
shared outputs for overlay
```

actor：

```text
mcp_code(68) + vision_flag(1) + obs_now(45)
  -> MLP 114 -> 512 -> 256 -> 128 -> 12
  -> action mean
```

critic：

```text
privileged_obs(269) + vision_flag(1)
  -> MLP 270 -> 512 -> 256 -> 128 -> 1
  -> value
```

策略分布：

- 训练：高斯策略采样动作
- 推理：直接使用均值动作
- `std` 是全局可学习参数，维度 `12`

动作物理意义：

```text
q_target = q_default + action_scale(0.25) * action
```

再进入 PD 控制器。

## 10. 训练流程

### 10.1 PPO 配置

| 项目 | 当前值 |
| --- | --- |
| `num_envs` | `4096` |
| `num_steps_per_env` | `24` |
| 单轮样本数 | `98304` |
| `max_iterations` | `30000` |
| `save_interval` | `100` |
| actor-critic LR | `2e-4` |
| estimator LR | `1e-3` |
| `schedule` | `adaptive` |
| `desired_kl` | `0.01` |
| `clip_param` | `0.2` |
| `gamma` | `0.99` |
| `lam` | `0.95` |
| `entropy_coef` | `0.01` |
| `num_learning_epochs` | `5` |
| `num_mini_batches` | `4` |
| `max_grad_norm` | `1.0` |

### 10.2 Rollout 中的数据流

每个 rollout step：

1. runner 维护 10 帧 `history`。
2. 每 `camera.update_interval=5` env steps 刷新 estimator。
3. estimator 读取 `proprio_hist + depth_camera + mask_vision`。
4. runner 把 `mcp_code + vision_flag` 交给 actor。
5. env step 后得到下一步 obs、reward、done、privileged obs。
6. 若 estimator 刷新过，则立即用当前 step 的监督目标更新 estimator。
7. PPO rollout storage 保存 `obs / critic_obs / action / reward / done / value / log_prob / mcp_code`。
8. rollout 结束后做 PPO update。

### 10.3 Estimator loss

当前 estimator 总损失实际包含：

| loss | 形式 | 作用 |
| --- | --- | --- |
| `vt_loss` | `MSE(v_t, gt_v_t)` | 速度估计 |
| `ht_loss` | `MSE(h_tf, gt_h_tf)` | 足端高度估计 |
| `mt_loss` | `MSE(m_hat, gt_m_t)` | 高度扫描重建 |
| `reconstruction_loss` | motion obs recon 与 terrain obs recon 平均 | 下一步本体预测 |
| `terrain_id_loss` | CE | 地形分类 |
| `fall_recovery_loss` | BCE | 翻身恢复分类 |
| `local_patch_recon_loss` | token patch reconstruction | 视觉 token 置信度学习 |
| `z_kl_loss` | KL warmup 到 `1e-2` | latent 约束 |
| `terrain_swav_loss` | warmup 到 `0.02` | gate-view 与 map-view 的跨视角一致性 |

当前 `load_balance_loss` 会记录日志，但没有加进 `total_loss`。  
`gate_smooth_coef`、`cycle_consistency_coef`、`vision_consistency_coef` 是保留参数，当前没有实际进入损失。

### 10.4 DepthAutoEncoder 当前状态

`actor_critic_parkour_moe.py` 中仍保留 `DepthAutoEncoder` 类，但当前 `PPOParkourMoE` 和 `OnPolicyRunnerParkourMoE` 没有实例化或训练它。

因此当前主线中：

- 有效的是 estimator core 内部的 token patch 重建与 token confidence。
- 独立 `DepthAutoEncoder` 是预留/历史接口，不是当前训练必经链路。

## 11. 推理、部署与可视化

### 11.1 Isaac play 推理

`legged_gym/scripts/play.py` 当前支持：

- 导出 `parkour_moe_inference_bundle.pt`
- 显示高度扫描、重建扫描、相机姿态、相机 hit points
- 显示 MoE gating weights
- 推理时可启用 random token dropout
- depth viewer 上可叠加 token dropout 网格

默认新增项：

```python
PLAY_RANDOM_TOKEN_DROPOUT = True
SHOW_TOKEN_DROPOUT_OVERLAY = True
```

### 11.2 MuJoCo 部署

MuJoCo 部署加载 bundle：

```text
actor_critic_state_dict
estimator_state_dict
policy_cfg
estimator_cfg
```

部署 adapter 做：

1. 维护本体历史。
2. 从 MuJoCo camera 得到 depth buffer。
3. estimator 编码 visual tokens。
4. 根据 token confidence / manual vision / random dropout 构建 deploy mask。
5. 生成 `mcp_code + vision_flag`。
6. actor 输出 12 维动作。

当前 `go2_parkour_moe.yaml` 中的部署可视化：

- depth preview: `480 x 320`
- token overlay: 开启
- recon scan: 开启
- reflective noise: 开启，blocks `4~5`
- confidence mask: 开启，threshold `0.98`
- random token dropout 配置存在但默认 `enabled: false`

### 11.3 部署端 token mask 信息

部署端会输出并缓存：

- `deploy_token_padding_mask`
- `deploy_random_token_dropout_mask`
- `deploy_bad_token_count`
- `deploy_random_token_dropout_count`
- `deploy_token_confidence`
- `image_height / image_width`
- `terrain_probs / terrain_pred`

OpenCV depth preview 会把这些信息画到深度图上：

- 红色：bad / padding token
- 橙色：random dropout token
- 网格：token grid
- 文本：masked token 数、random drop 数、地形预测与置信度

## 12. 当前注意事项

1. 旧文档中的若干数字已经过期。当前是 `4096 env`、`269 critic obs`、`3 experts`、`1 depth frame`、`68-dim mcp_code`、actor MCP 输入 `69` 维。
2. `step_stone` 当前采样比例为 `0`，所以 terrain classifier 仍输出 12 类，但该类没有训练样本。
3. `corridor` 当前比例为 `0.05`，已经参与采样。
4. 当前主线没有使用独立 `DepthAutoEncoder`，视觉置信度来自 estimator core 的 token patch reconstruction。
5. 训练时视觉会被 easy terrain toggle 和 omni blind 关闭；普通推理路径中视觉默认开启，除非显式传入 mask/dropout。
6. `load_balance_loss` 当前只记录，不参与 total loss。
7. `fall_recovery_pred` 已进入 `mcp_code`，这是当前框架相对早期版本的重要变化。
8. MuJoCo 部署比训练多了一层面向调试和鲁棒性实验的 token confidence / random dropout / overlay 机制。
