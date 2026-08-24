# Parkour MoE 任务设计文档

## 1. 文档范围

本文档基于当前仓库里的 `go2_parkour_moe` 实现，整理以下 6 个方面：

1. 环境设计
2. 奖励函数设计
3. 终止条件设计
4. 网络设计
5. 启用的随机化处理
6. 训练配置

本文档对应的主要代码文件如下：

- `legged_gym/envs/go2/go2_config_parkour_moe.py`
- `legged_gym/envs/go2/go2_parkour_env.py`
- `legged_gym/envs/go2/go2_env.py`
- `legged_gym/envs/base/legged_robot.py`
- `legged_gym/envs/base/legged_robot_config.py`
- `legged_gym/utils/terrain.py`
- `legged_gym/utils/terrain_mgdp.py`
- `rsl_rl/rsl_rl/modules/actor_critic_parkour_moe.py`
- `rsl_rl/rsl_rl/algorithms/ppo_parkour_moe.py`
- `rsl_rl/rsl_rl/runners/onpolicyrunner_parkour_moe.py`

除非特别说明，下面描述的都是“当前代码真正生效的路径”，而不是历史版本或预留接口。

## 2. 环境设计

### 2.1 基本任务配置

当前 `parkour_moe` 任务的核心环境配置如下：

- 机器人：Go2
- 仿真环境数：`8192`
- 动作维度：`12`
- actor 当前观测维度：`45`
- critic 特权观测维度：`268`
- episode 长度：`20 s`
- 控制周期：`dt = sim.dt * decimation = 0.005 * 4 = 0.02 s`
- 地形网格：`10` 行 `45` 列
- 单块地形尺寸：长 `10.0 m`，宽 `4.0 m`
- 地形类型风格：`mgdp_parkour`
- 高度扫描：开启，扫描点为 `17 x 11 = 187` 个
- 深度输入：代理深度相机 `depth_camera`，缓存 `2` 帧，输入分辨率为 `58 x 87`

额外说明：

- 当前任务关闭了 `turn_over` 初始化，即不从翻身状态开始训练。
- 相机实际采用 `proxy` 模式，不走 Isaac Gym 真实相机传感器，而是基于高度场/三角网格做 raycast 合成深度图。
- reset 时会优先把机器人放回地形起始平台，`x` 方向带 `0.15 m` 抖动，`y` 方向抖动为 `0.0 m`。

### 2.2 地形类型与课程难度

#### 2.2.1 课程难度的定义

当前地形课程使用 `num_rows = 10` 行，因此实际生成时的原始难度取值为：

$$
d \in \left\{0.0, 0.1, 0.2, \dots, 0.9\right\}
$$

也就是说，代码里并不会直接采样 `d = 1.0`。  
但部分地形生成函数会再做一次归一化：

$$
\hat d = \frac{d}{0.9} \in [0, 1]
$$

因此：

- 课程行号的原始难度范围是 `0.0 ~ 0.9`
- 对使用 `_normalized_difficulty()` 的地形，最高一行会被映射为归一化难度 `1.0`

#### 2.2.2 当前启用/未启用的地形

当前 `terrain_proportions` 为：

```python
[0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1]
```

对应 12 种地形：

| terrain_id | 地形名 | 当前比例 | 是否采样 |
| --- | --- | ---: | --- |
| 0 | `single_gap` | 0.1 | 是 |
| 1 | `step_stone` | 0.0 | 否 |
| 2 | `two_row_stones` | 0.1 | 是 |
| 3 | `one_row_stones` | 0.1 | 是 |
| 4 | `single_bridge` | 0.1 | 是 |
| 5 | `air_beams` | 0.1 | 是 |
| 6 | `air_stones` | 0.1 | 是 |
| 7 | `hurdle` | 0.1 | 是 |
| 8 | `ramp` | 0.1 | 是 |
| 9 | `corridor` | 0.0 | 否 |
| 10 | `stairs_up` | 0.1 | 是 |
| 11 | `flat` | 0.1 | 是 |

因此，当前虽然地形分类头仍然输出 `12` 类，但 `step_stone` 和 `corridor` 在当前配置下没有训练样本。

#### 2.2.3 各地形在 difficulty 下的关键几何参数

下面给出每类地形当前代码中的“关键难度参数”。  
说明：

- 表中的范围是按当前课程行 `d = 0.0 ~ 0.9` 推出来的。
- 对用了归一化难度的地形，会显式写成 $\hat d = d / 0.9$。
- 高度场离散化使用 `horizontal_scale = 0.05 m`、`vertical_scale = 0.005 m`，因此最终几何会有轻微量化误差。
- `mgdp_add_roughness = False`，所以当前所有 parkour 地形都不会再额外叠加 roughness。

| terrain_id | 地形名 | 当前关键参数 | difficulty 最小时 | difficulty 最大时 |
| --- | --- | --- | --- | --- |
| 0 | `single_gap` | gap 宽度：`0.1 + 0.8 * \hat d`，深度固定 `0.6 m` | `0.10 m` | `0.90 m` |
| 1 | `step_stone` | 石块尺寸：`0.7` 或 `0.7 - 0.5 d^2`；石块间距：`0.05` 或 `0.4 * floor(10d) / 10`；石块高度上限：`0.05 + 0.18 d` | 尺寸约 `0.70 m`，间距 `0.05 m`，高度上限 `0.05 m` | 尺寸约 `0.295 m`，间距 `0.36 m`，高度上限 `0.212 m` |
| 2 | `two_row_stones` | 双排石块尺寸：`0.8` 或 `0.8 - 0.5 d`；排间推进间距：`0.1` 或 `0.4 * floor(10d) / 10`；高度上限：`0.05 + 0.18 d` | 尺寸约 `0.80 m`，间距约 `0.08 ~ 0.10 m`，高度上限 `0.05 m` | 尺寸约 `0.35 m`，间距 `0.36 m`，高度上限 `0.212 m` |
| 3 | `one_row_stones` | 单排石块尺寸与高度公式同上；实际 `x` 向间距在生成函数中随 `d` 增大并被裁剪 | 尺寸约 `0.80 m`，实际 `x` 间距约 `0.20 m` | 尺寸约 `0.35 m`，实际 `x` 间距约 `0.80 m` |
| 4 | `single_bridge` | 桥面宽度：`0.6` 或 `0.8 - 0.6 d` | 低难度约 `0.60 m`，在 `d = 0.2` 会出现 `0.68 m` 的分段拐点 | 约 `0.26 m` |
| 5 | `air_beams` | 悬空梁长度：`0.35` 或 `0.35 - 0.1 d`；梁间距：`0.1` 或 `0.4 * floor(10d) / 10`；梁高度尺度：`0.04` 或 `0.05 + 0.18 d` | 梁长 `0.35 m`，间距约 `0.08 ~ 0.10 m`，高度尺度 `0.04 m` | 梁长约 `0.26 m`，间距 `0.36 m`，高度尺度约 `0.212 m` |
| 6 | `air_stones` | 悬空平台下沿净空：`0.5 - 0.25 * \hat d`；平台尺寸固定 `4.5 m x (1.8~2.4) m x 0.5 m` | 净空 `0.50 m` | 净空 `0.25 m` |
| 7 | `hurdle` | 障碍高度范围：若 `d < 0.1` 为 `[0.05 + 0.18 d, 0.15 + 0.18 d]`，否则为 `[0.1 + 0.4 d, 0.2 + 0.5 d]` | 高度范围约 `[0.05, 0.15] m` | 高度范围约 `[0.46, 0.65] m` |
| 8 | `ramp` | 坡面角度：`30 * \hat d` 度 | `0 deg` | `30 deg` |
| 9 | `corridor` | 中间通道宽度随 `d` 缩小，当前实现对应全通道宽度大约 `0.8 m -> 0.2 m` | 约 `0.80 m` | 约 `0.20 m` |
| 10 | `stairs_up` | 台阶高度：`0.05 + 0.20 * \hat d`；台阶宽度固定 `0.31 m` | 台阶高 `0.05 m` | 台阶高 `0.25 m` |
| 11 | `flat` | 无额外几何难度 | 平地 | 平地 |

### 2.3 地形课程机制

当前地形课程开启，且初始最大等级为：

```python
max_init_terrain_level = 2
```

因此训练开始时，环境只会在难度等级 `0, 1, 2` 之间分配。之后在每次 reset 时，根据机器人完成情况更新地形等级。

当前课程更新规则如下：

1. 计算机器人相对 `reset_origin` 的平面移动距离

$$
\text{distance} = \left\|p_{xy} - p^{reset}_{xy}\right\|_2
$$

2. 如果：

$$
\text{distance} > \frac{L_{terrain}}{2}
$$

则升级一档。当前 `terrain_length = 10 m`，所以升级阈值是 `5 m`。

3. 否则，如果机器人实际移动距离小于“指令期望距离的一半”，则降级一档：

$$
\text{distance} < \left\|c_{xy}\right\|_2 \cdot T_{episode} \cdot 0.5
$$

其中当前 `T_{episode} = 20 s`。

4. 等级会被裁剪到合法范围 `[0, 9]`。如果已经超过最后一级，则会被随机重新放回某个合法等级。

### 2.4 指令设计

当前 parkour 任务的指令采样非常偏“前向通关”：

- `lin_vel_x`：`[0.0, 1.5]`
- `lin_vel_y`：`[0.0, 0.0]`
- `ang_vel_yaw`：`[-1.0, 1.0]`
- `heading`：`[0.0, 0.0]`
- `heading_command = True`

这意味着：

1. 真正采样的是前向速度和目标航向。
2. 但目标航向被固定为 `0`，所以本质上任务希望机器人始终沿世界系 `+x` 方向前进。
3. `commands[:, 2]` 并不是直接采样出来的，而是在每个物理步前根据 heading error 重新计算：

$$
\omega_z^{cmd} = \text{clip}\left(0.5 \cdot \text{wrapToPi}(\psi^{cmd} - \psi), \omega_z^{min}, \omega_z^{max}\right)
$$

在当前配置下，$\psi^{cmd} = 0$。

另外还有两个约束：

- `zero_command = True`
- `zero_command_threshold = 0.2`

也就是当采样到的平面速度范数小于 `0.2` 时，会直接把 `xy` 指令清零。

### 2.5 reset 设计

当前 parkour reset 的关键点如下：

- 重置位置回到 terrain 的 `reset_origin`
- 起始平台长度 `2.0 m`
- 起始平台宽度 `2.0 m`
- `x` 抖动 `0.15 m`
- `y` 抖动 `0.0 m`
- 非翻身初始化时，机身姿态固定为水平朝前
- reset 时 base 线速度和角速度会随机到 `[-0.5, 0.5]`

这意味着任务更偏向“从指定朝向的起跑平台稳定出发”，而不是从随机航向出发。

## 3. 奖励函数设计

### 3.1 奖励汇总规则

当前 reward 的总形式为：

$$
r_t = \sum_i w_i^{cfg} \cdot \Delta t \cdot \alpha_i(k) \cdot \hat r_i
$$

其中：

- $w_i^{cfg}$：配置文件里的原始权重
- $\Delta t = 0.02$
- $\alpha_i(k)$：reward curriculum 系数，若无课程则恒为 `1`
- $\hat r_i$：对应 reward 函数的原始输出

当前 reward 还有两个重要设定：

- `only_positive_rewards = False`，所以总奖励不会被截断为非负
- `termination` reward 的 scale 为 `0`，因此当前没有额外的终止奖励/惩罚项

### 3.2 当前启用的 reward curriculum

当前只有两个 reward 带课程系数：

1. `lin_vel_z`

$$
\alpha_{lin\_vel\_z}(k) = \text{lerp}(1.0, 0.0)
$$

- 作用区间：训练迭代 `0 -> 1500`
- 结果：`lin_vel_z` 的惩罚会从原始权重 `-2.0` 线性衰减到 `0.0`

2. `correct_base_height`

$$
\alpha_{correct\_base\_height}(k) = \text{lerp}(1.0, 10.0)
$$

- 作用区间：训练迭代 `0 -> 5000`
- 结果：`correct_base_height` 的原始权重会从 `-1.0` 线性放大到 `-10.0`

### 3.3 符号说明

为了让公式更紧凑，下面统一使用：

- $v_b = (v_x, v_y, v_z)$：机身线速度
- $\omega_b = (\omega_x, \omega_y, \omega_z)$：机身角速度
- $q, \dot q$：关节位置与速度
- $a_t$：当前动作
- $a_{t-1}, a_{t-2}$：过去两步动作
- $\tau$：关节力矩
- $h_b$：base 相对地面的估计高度
- $h_b^\*$：目标 base 高度，当前为 `0.38 m`
- $\psi$：当前 yaw
- $c^{cmd}$：速度指令

### 3.4 当前启用的 reward 一览

下表列出当前配置下所有非零 reward。

| reward 名 | 原始公式 | 配置权重 | 每步实际系数 | 作用 |
| --- | --- | ---: | ---: | --- |
| `tracking_lin_vel` | $\exp(-((v_x^{cmd}-v_x)^2 + (v_y^{cmd}-v_y)^2)/0.25) \cdot m_{heading}$ | `1.0` | `0.02` | 奖励平面速度跟踪；在当前 parkour 中还会乘以朝前对齐系数 $m_{heading}=\max(\cos\psi, 0)$ |
| `tracking_ang_vel` | $\exp(-(\omega_z^{cmd}-\omega_z)^2/0.25)$ | `0.5` | `0.01` | 奖励 yaw 角速度对 heading 修正命令的跟踪 |
| `lin_vel_z` | $v_z^2$ | `-2.0`，并线性衰减到 `0.0` | `-0.04 -> 0.0` | 前期强抑制机身上下跳动，后期逐渐放开 |
| `ang_vel_xy` | $\omega_x^2 + \omega_y^2$ | `-0.05` | `-0.001` | 抑制 pitch/roll 抖动 |
| `dof_acc` | $\sum_j ((\dot q_{j,t-1}-\dot q_{j,t})/\Delta t)^2$ | `-2.5e-7` | `-5e-9` | 抑制关节加速度尖峰 |
| `dof_power` | $\sum_j |\tau_j \dot q_j|$ | `-2e-5` | `-4e-7` | 抑制功耗 |
| `torques` | $\sum_j \tau_j^2$ | `-1e-4` | `-2e-6` | 抑制过大力矩 |
| `correct_base_height` | $(h_b - h_b^\*)^2 \cdot m_{base}$ | `-1.0`，并线性放大到 `-10.0` | `-0.02 -> -0.2` | 约束机身高度；但在若干 parkour 地形上被显式关闭 |
| `action_rate` | $\sum_j (a_{t,j}-a_{t-1,j})^2$ | `-0.01` | `-2e-4` | 抑制相邻两步动作突变 |
| `action_smoothness` | $\sum_j (a_{t,j} - 2a_{t-1,j} + a_{t-2,j})^2$ | `-0.01` | `-2e-4` | 抑制二阶动作抖动 |
| `collision` | 被惩罚 body 上的接触次数 | `-1.0` | `-0.02` | 惩罚 `thigh/calf/base/Head_lower` 接触 |
| `dof_pos_limits` | $\sum_j [\max(q_j^{low}-q_j, 0) + \max(q_j-q_j^{high}, 0)]$ | `-2.0` | `-0.04` | 惩罚关节接近软限位 |
| `feet_regulation` | $\sum_f \|v_{f,xy}\|^2 \exp(-h_f / (0.025 h_b^\*))$ | `-0.05` | `-0.001` | 惩罚“贴地高速扫腿”，鼓励迈步时抬脚 |
| `hip_to_default` | $\sum_{hip} |q_{hip} - q_{hip}^{default}|$ | `-0.05` | `-0.001` | 约束髋关节不要长期偏离默认姿态 |
| `low_speed_when_commanded` | $\max(\min(\|c_{xy}\|, 0.25) - \|v_{b,xy}\|, 0) \cdot \mathbb{I}[\|c_{xy}\| > 0.2]$ | `-0.5` | `-0.01` | 有前向指令但机器人速度太低时额外惩罚 |
| `hard_terrain_yaw_penalty` | $\omega_z^2 \cdot \mathbb{I}[c_x > 0]$ | `-0.2` | `-0.004` | 在当前 parkour 中基本等价于“有前向任务时惩罚多余偏航” |
| `upward_foothold_clearance` | $\sum_f [\exp(-(c_f-c_f^\*)^2/0.04) - 1.25 \rho_f] \cdot \mathbb{I}_f$ | `0.5` | `0.01` | 奖励摆动腿对即将到来的上台阶/上凸起的清障高度匹配 |
| `targeted_foothold_touchdown` | $\sum_f \exp(-\|p_{f,xy}^{td} - p_{f,xy}^\*\|^2/0.03) \cdot \mathbb{I}_f^{td}$ | `0.5` | `0.01` | 奖励脚在落地瞬间落到“目标可落脚点”附近 |

其中：

- `m_heading`：只在 `heading_conditioned_terrain_ids = range(12)` 且 `cmd_x > 0` 时生效，因此当前几乎所有 parkour 地形都会受朝前对齐影响。
- `m_base`：base height 奖励掩码，对 `terrain_id in [0, 2, 3, 5, 6]` 关闭，即：
  - `single_gap`
  - `two_row_stones`
  - `one_row_stones`
  - `air_beams`
  - `air_stones`

### 3.5 两个关键地形感知 reward 的详细展开

#### 3.5.1 `upward_foothold_clearance`

这项奖励是当前 parkour 里非常关键的“前瞻式抬脚奖励”。

1. 先计算当前脚底离地高度：

$$
c_f = \max(z_f - z_{ground}(x_f, y_f), 0)
$$

2. 再沿着机身前向方向看一小段距离 `0.06 m`，得到前方地形抬升量：

$$
\Delta h_f = \max(z_{ground}^{ahead} - z_{ground}^{now}, 0)
$$

3. 目标抬脚高度为：

$$
c_f^\* = \min(0.03 + 1.0 \cdot \Delta h_f, 0.7)
$$

4. 单脚奖励为：

$$
r_f^{clear} = \exp(-(c_f-c_f^\*)^2 / 0.04) - 1.25 \rho_f
$$

其中 $\rho_f$ 是摆动超时惩罚，若单次腾空超过 `1.0 s` 就会逐渐变大。

5. 只有满足下面条件时，这只脚才会真正拿到奖励：

- 前向命令 `cmd_x > 0.2`
- base 前向速度足够大
- 足端平面速度 `> 0.1`
- 当前处于摆动相
- 前方地形确实在上升，且上升量 `> 0.02 m`
- 地形不属于 `air_stones`

这项奖励的作用是：

- 在面对 hurdle、stairs、ramp 等“前方有抬升”的地形时，鼓励摆动腿主动增加清障高度
- 同时又避免脚抬得过高、摆太久

#### 3.5.2 `targeted_foothold_touchdown`

这项奖励会从当前高度扫描点里，替每只脚动态挑一个“最值得踩”的落脚点。

目标点筛选逻辑如下：

1. 先为每只脚定义一个 anchor 区域，前脚更靠前，后脚稍靠后。
2. 在该 anchor 周围的一个搜索框里，从高度扫描点中挑候选点。
3. 候选点打分公式为：

$$
\text{score} =
1.0 \cdot \text{normalized\_height}
- 6.0 \cdot \text{anchor\_distance}^2
- 1.0 \cdot \text{edge\_mask}
$$

4. 得分最高的点被选为当前脚的落脚目标。

落地奖励在“脚从摆动转为首次接触地面”的瞬间计算：

$$
r_f^{td} = \exp(-\|p_{f,xy}^{td} - p_{f,xy}^\*\|^2 / 0.03)
$$

并且只有满足以下条件时才计分：

- 该脚本次摆动时间大于 `0.12 s`
- 当前地形属于 `targeted_foothold_terrain_ids`
- 当前 `cmd_x > 0.2`
- 目标点有效

当前这项奖励覆盖的地形是：

- `single_gap`
- `step_stone`
- `two_row_stones`
- `one_row_stones`
- `single_bridge`
- `air_beams`
- `hurdle`
- `ramp`
- `corridor`
- `stairs_up`
- `flat`

即仅排除了 `air_stones`。

这项奖励的作用是：

- 在可选落脚点较稀疏的 terrain 上，显式鼓励脚落在“高、稳、远离边缘”的位置
- 帮助策略学到更明确的 stepping 策略，而不是只靠速度跟踪和碰撞惩罚被动学

## 4. 终止条件设计

当前终止条件由基础 `LeggedRobot.check_termination()` 和 parkour 版 `Go2ParkourRobot.check_termination()` 共同决定。

### 4.1 当前启用的终止条件

| 终止原因 | 代码条件 | 当前阈值/范围 | 说明 |
| --- | --- | --- | --- |
| `base_contact_upside_down` | `base` 接触且 `projected_gravity[:,2] > 0` | 接触力范数 `> 1.0` | 当前 asset 只把 `base` 放在 `terminate_after_contacts_on` 里，而且必须是“底朝上碰 base”才会终止 |
| `time_out` | `episode_length_buf > max_episode_length` | `20 s / 0.02 = 1000` 步 | 超时终止；PPO 中会按 timeout 方式处理 bootstrap |
| `fall_base_height_cutoff` | `root_states[:,2] < fall_base_height` | `-0.1 m` | 对所有 `terrain_id in range(12)` 生效 |
| `fall_foot_height_cutoff` | `min(feet_z) < fall_foot_height` | 默认 `-0.1 m`，`single_gap` 特判 `-0.3 m` | gap 地形允许脚下探得更低，否则会误杀 |
| `no_progress` | 前向最大进度长时间没有提升 | `cmd_x > 0.2`，`2.0 s` 内未增加 `0.1 m` | 当前 `no_progress_min_terrain_level = 0`，所以从最低等级就开始监控 |
| `out_of_block` | 机器人跑出当前 terrain block 范围 | 相对块中心 `x > 5.8 m` 或 `y > 2.8 m` | 由 `terrain_length/width` 和 `reset_outside_block_margin = 0.8` 决定 |

### 4.2 终止条件的具体解释

#### 4.2.1 基础碰撞终止

当前并没有“任何身体碰撞都直接终止”的设计。

- `terminate_after_contacts_on = ["base"]`
- 非 base 的 `thigh/calf/Head_lower` 只会进 `collision` reward，不会直接 reset

而且即使是 `base`，也不是一碰就 reset，而是要求：

$$
\text{base\_contact} \land \text{upside\_down}
$$

所以当前终止逻辑相对宽松，更强调继续尝试通过地形，而不是轻微碰撞立即结束。

#### 4.2.2 高度跌落终止

Parkour 里增加了“高度过低终止”，其目的非常明确：

- 避免机器人掉坑之后长时间无效挣扎
- 避免 critic/value 被大量无意义的低位姿状态污染

特别是 `single_gap` 的脚部高度阈值被放宽到 `-0.3 m`，这说明当前实现考虑到了脚在 gap 上方跨越时的合法下探。

#### 4.2.3 无进度终止

当前无进度逻辑基于“当前 episode 的最大前向进度”：

$$
\text{forward\_progress} = x_{base} - x_{progress\_origin}
$$

若在 `2.0 s` 内始终没有超过“历史最大进度 + 0.1 m”，且此时前向命令大于 `0.2 m/s`，则判定为卡住并重置。

这项终止在 parkour 中非常重要，因为它能尽早结束“原地蹭动、反复试错但毫无前进”的无效 rollout。

## 5. 网络设计

## 5.1 输入部分

### 5.1.1 训练阶段总览

训练阶段的网络相关输入如下：

| 输入名 | 维度 | 是否 actor 可见 | 物理含义 |
| --- | --- | --- | --- |
| `obs_now` | `[B, 45]` | 是 | 当前时刻本体观测 |
| `proprio_hist` | `[B, 450]`，内部 reshape 为 `[B, 10, 45]` | 是 | 最近 `10` 帧本体观测历史 |
| `depth_camera` | `[B, 2, 58, 87]` | 间接可见 | 最近 `2` 帧代理深度图 |
| `mask_vision` | `[B]` | 间接可见 | 当前样本是否允许 estimator 使用视觉 token |
| `critic_obs` | `[B, 268]` | 否 | critic 使用的特权观测 |
| `gt_v_t` | `[B, 3]` | 否 | estimator 监督目标：base 线速度 |
| `gt_h_tf` | `[B, 4]` | 否 | estimator 监督目标：4 足相对地面高度 |
| `gt_m_t` | `[B, 187]` | 否 | estimator 监督目标：局部高度图 |
| `gt_terrain_id` | `[B, 1]` | 否 | estimator 监督目标：当前 terrain id |
| `gt_next_obs` | `[B, 45]` | 否 | estimator 监督目标：下一时刻 actor 观测 |

### 5.1.2 推理阶段总览

推理阶段不再使用特权观测，也不再训练 critic，因此真正参与部署链路的输入只有：

| 输入名 | 维度 | 物理含义 |
| --- | --- | --- |
| `obs_now` | `[B, 45]` | 当前本体观测 |
| `proprio_hist` | `[B, 450]` | 10 帧本体历史 |
| `depth_camera` | `[B, 2, 58, 87]` | 2 帧代理深度 |
| `mask_vision` | `[B]` | 当前代码里固定为全 `1`，即推理时始终开启视觉 |

注意：

- 代码里确实保留了用 depth autoencoder 重建误差决定 `mask_vision` 的接口。
- 但推理路径中真正根据 `selector_threshold` 去关闭视觉的那行代码目前被注释掉了。
- 因此“当前仓库的真实推理行为”是：**视觉始终开启**。

### 5.1.3 `obs_now` 的 45 维组成

`obs_now` 来自 `Go2Robot.compute_observations()`：

| 组成 | 维度 | 物理含义 | 当前缩放 |
| --- | ---: | --- | --- |
| `base_ang_vel` | 3 | 机身角速度（body frame） | `0.25` |
| `projected_gravity` | 3 | 重力方向投影到机身坐标系 | 不缩放 |
| `commands[:3]` | 3 | `lin_vel_x, lin_vel_y, ang_vel_yaw` 指令 | `[2.0, 2.0, 0.25]` |
| `dof_pos - default_dof_pos` | 12 | 关节偏离默认姿态量 | `1.0` |
| `dof_vel` | 12 | 关节速度 | `0.05` |
| `actions` | 12 | 上一时刻动作 | 不缩放 |
| 合计 | 45 |  |  |

### 5.1.4 `critic_obs` 的 268 维组成

当前 `critic_obs` 的来源是：

1. `Go2Robot` 生成的 `263` 维 privileged obs
2. `Go2ParkourRobot` 追加 `4` 维足端高度
3. 再追加 `1` 维 `terrain_id`

总维度：

$$
263 + 4 + 1 = 268
$$

具体组成如下：

| 组成 | 维度 | 物理含义 |
| --- | ---: | --- |
| `base_lin_vel` | 3 | 机身线速度（body frame） |
| `base_ang_vel` | 3 | 机身角速度 |
| `projected_gravity` | 3 | 重力投影 |
| `commands[:3]` | 3 | 当前指令 |
| `dof_pos error` | 12 | 关节位置误差 |
| `dof_vel` | 12 | 关节速度 |
| `actions` | 12 | 上一时刻动作 |
| `foot contact forces` | 4 | 四足接触力模长 |
| `torques / torque_limits` | 12 | 归一化电机力矩 |
| `(last_dof_vel - dof_vel) / dt * 1e-4` | 12 | 缩放后的关节加速度 |
| `height_measurements` | 187 | 前向偏置的地形高度扫描 |
| `feet_heights` | 4 | 四足相对地面的高度 |
| `terrain_id` | 1 | 当前地形类别 id |
| 合计 | 268 |  |

其中：

- `height_measurements` 对应扫描窗口 `x = -0.4 ~ 1.2 m`、`y = -0.5 ~ 0.5 m`
- estimator 的监督目标就是从这个 `critic_obs` 里直接切片得到的：
  - `v_t`：前 `3` 维
  - `m_t`：第 `76:263` 维，共 `187` 维
  - `h_tf`：第 `263:267` 维，共 `4` 维
  - `terrain_id`：第 `267:268` 维

## 5.2 网络中层部分

当前网络可以分成 4 大块：

1. `ParkourEstimator`
2. `ActorCriticParkourMoE`
3. `DepthAutoEncoder`
4. runner 侧的 `mcp_code + vision_flag` 拼接逻辑

### 5.2.1 Estimator 主链路

| 模块 | 输入 | 网络形式 | 输出 | 输出含义 |
| --- | --- | --- | --- | --- |
| `image_encoder` | `[B, 2, 58, 87]` | `Conv2d(2,16,3,2,1) -> Conv2d(16,32,3,2,1) -> Conv2d(32,64,3,2,1)` | `[B, 64, 8, 11]` | 视觉特征图 |
| `image_tokens` | `[B, 64, 8, 11]` | flatten + transpose | `[B, 88, 64]` | 88 个视觉 token |
| `proprio_encoder` | `[B, 10, 45]` | `45 -> 256 -> 128 -> 64` MLP | `[B, 10, 64]` | 10 个本体历史 token |
| `vision_token` | `[B]` | reshape + broadcast | `[B, 1, 1]`，后续编码成 `[B, 1, 64]` | 视觉是否可用的开关 token |
| `selector transformer` | `88 + 10 + 1 = 99` 个 token | `TransformerEncoderLayer(d_model=64, nhead=4, ff=256)`，`1` 层 | `[B, 99, 64]` | 第一阶段多模态编码 |
| `shared attention pool` | `[B, 99, 64]` | 单 query attention pooling | `[B, 64]` | 全局 selector 特征 |
| `selector_head` | `[B, 64]` | `64 -> 64 -> 12` | `[B, 12]` | terrain logits / terrain probs |
| `terrain_token_encoder` | `[B, 12]` | `12 -> 64 -> 64` | `[B, 64]` | 地形类别 token |
| `main transformer` | `88 + 10 + 1 + 1 = 100` 个 token | 同上，再过 1 次 Transformer | `[B, 100, 64]` | 第二阶段融合后的 token 序列 |
| `shared_pool` | `[B, 100, 64]` | 单 query attention pooling | `[B, 64]` | 全局共享上下文 |
| `shared_head` | `[B, 64]` | `64 -> 64` | `[B, 64]` | router 用的上下文特征 |
| `expert_pool` | `[B, 100, 64]` | 4 个 query 的 attention pooling | `[B, 4, 64]` | 4 个 expert 的输入特征 |
| `router_terrain_encoder` | `[B, 12]` | `12 -> 32 -> 32` | `[B, 32]` | router 用的 terrain 特征 |
| `router` | `obs_now(45) + vision_flag(1) + context(64) + terrain_feat(32)` | `142 -> 128 -> 4 -> Softmax` | `[B, 4]` | 4 个 expert 的 gating weights |
| `4 x ReadWriteExpert` | 每个 expert 吃 `[B, 64]` 特征和 `[B, 64]` shared state | `read MLP -> GRUCell -> write MLP` | `hidden_state_i: [B,64]`，`shared_delta_i: [B,64]` | 每个 expert 对共享记忆的读写结果 |
| `mixture + shared_state_updater` | 4 个 `shared_delta_i` + gating weights + 上一步 `shared_state_prev` | 加权求和后过 `GRUCell(64,64)` + `LayerNorm` | `[B, 64]` | 当前公共记忆 `shared_state` |
| `prediction_head` | `[B, 64]` | `64 -> 64` | `[B, 64]` | 供各个预测头共享的状态 |

### 5.2.2 Estimator 输出头

| 模块 | 输入 | 网络形式 | 输出 | 含义 |
| --- | --- | --- | --- | --- |
| `fc_v_t` | `[B, 64]` | `64 -> 3` | `[B, 3]` | 预测 base 线速度 |
| `fc_h_tf` | `[B, 64]` | `64 -> 4` | `[B, 4]` | 预测四足高度 |
| `fc_z_mu` | `[B, 64]` | `64 -> 16` | `[B, 16]` | 潜变量均值 |
| `fc_z_logvar` | `[B, 64]` | `64 -> 16` | `[B, 16]` | 潜变量方差参数 |
| `fc_z_tm` | `[B, 64]` | `64 -> 32` | `[B, 32]` | 地形图 latent |
| `decoder_obs` | `v_t(3) + h_tf(4) + z_t(16)` | `23 -> 64 -> 45` | `[B, 45]` | 重建下一时刻 actor 观测 |
| `decoder_map` | `z_tm(32)` | `32 -> 64 -> 187` | `[B, 187]` | 重建高度图 |

其中一个很重要的实现细节是：

- actor 不使用采样后的 `z_t`
- actor 使用的是确定性的 `z_mu`

也就是说：

- stochastic latent 主要服务于 estimator 的自监督/辅助学习
- 真正喂给 actor 的中间控制码是更稳定的确定性表示

### 5.2.3 `mcp_code` 的组成

当前 estimator 返回给 actor 的 `mcp_code` 为：

$$
\text{mcp\_code} =
[v_t(3), h_{tf}(4), z_\mu(16), z_{tm}(32), \text{terrain\_onehot}(12)]
$$

因此维度是：

$$
3 + 4 + 16 + 32 + 12 = 67
$$

随后 runner 会再拼接一维 `vision_flag`，得到真正给 actor/critic 用的版本：

$$
67 + 1 = 68
$$

### 5.2.4 Actor-Critic 结构

| 模块 | 输入 | 网络形式 | 输出 | 含义 |
| --- | --- | --- | --- | --- |
| actor 输入拼接 | `mcp_code_with_flag(68) + obs_now(45)` | concat | `[B, 113]` | actor 的完整输入 |
| actor MLP | `[B, 113]` | `113 -> 512 -> 256 -> 128 -> 12`，激活 `ELU` | `[B, 12]` | 动作均值 |
| critic 输入拼接 | `critic_obs(268) + vision_flag(1)` | concat | `[B, 269]` | critic 的完整输入 |
| critic MLP | `[B, 269]` | `269 -> 512 -> 256 -> 128 -> 1`，激活 `ELU` | `[B, 1]` | 状态价值 |

actor 采用高斯策略：

- 均值来自 actor MLP 输出
- 方差是一个全局可学习参数 `std \in R^{12}`

训练时采样：

$$
a \sim \mathcal{N}(\mu, \sigma)
$$

推理时直接输出均值：

$$
a = \mu
$$

### 5.2.5 DepthAutoEncoder

这部分不是 actor 主链路的一部分，但它是当前训练中的有效模块：

| 模块 | 输入 | 网络形式 | 输出 | 含义 |
| --- | --- | --- | --- | --- |
| encoder | `[B, 2, 58, 87]` | 三层 `Conv2d` | `[B, 64, 8, 11]` | 深度压缩特征 |
| latent | flatten 后 | `64*8*11 -> 128` | `[B, 128]` | 深度重建 latent |
| decoder | `128 -> 64*8*11` 后三层 `ConvTranspose2d` | 上采样回 `58 x 87` | `[B, 2, 58, 87]` | 深度重建图 |

当前它的有效作用是：

1. 提供单独的深度重建损失
2. 推理路径中会计算重建误差，但当前不会真正用于关闭视觉

## 5.3 输出部分：动作空间

当前动作空间是 `12` 维连续动作，对应 Go2 的 12 个关节。

### 5.3.1 动作的物理意义

动作不是直接的力矩，而是 **关节位置残差命令**。

当前控制配置：

- `control_type = "P"`
- `action_scale = 0.25`

因此 actor 输出的动作 $a \in \mathbb{R}^{12}$ 会先变成目标关节位置增量：

$$
q^{target} = q^{default} + 0.25 \cdot a
$$

然后进入 PD 控制器：

$$
\tau =
K_p \odot (q^{target} - q + q^{offset})
- K_d \odot \dot q
$$

其中：

- $q^{offset}$：motor zero offset 域随机化
- $K_p, K_d$：还会乘以各自的随机倍率

若开启电机强度随机化，最终还会再乘一个 `motor_strength` 系数，然后裁剪到力矩上限。

### 5.3.2 动作输出的使用方式

- 训练阶段：从高斯分布里采样动作
- 推理阶段：直接使用 actor 均值
- 环境侧还会做动作延迟随机化，因此“真正下发到仿真器的动作”可能是当前动作和上一动作的混合切换版本

## 6. 启用的随机化处理

当前随机化可以分成 4 类：

1. 地形生成随机化
2. 机器人动力学随机化
3. 控制链随机化
4. 观测/感知随机化

### 6.1 地形生成随机化

这类随机化发生在 terrain 生成阶段，而不是每次 reset 都重新生成整张地形图。

| 项目 | 当前状态 | 说明 |
| --- | --- | --- |
| 课程难度 | 开启 | 10 个 difficulty level，按 reset 成绩升降级 |
| 地形类型比例 | 开启 | 12 类地形中当前采样 10 类 |
| terrain 内部随机几何 | 开启 | 例如石块高度、beam 宽度、beam 高度、hurdle 高度都带随机采样 |
| roughness 叠加 | 关闭 | `mgdp_add_roughness = False` |
| air beam extra trimesh | 开启 | `add_air_beam = True` |
| air stone extra trimesh | 开启 | `add_air_stone = True` |
| reset 平台位置抖动 | 开启 | `x` 抖动 `0.15 m`，`y` 抖动 `0.0 m` |

### 6.2 机器人动力学随机化

| 项目 | 发生时机 | 当前范围/规则 |
| --- | --- | --- |
| 摩擦系数 | 创建 env 时 | 每个 env 从 `64` 个 bucket 中抽样，范围 `[0.0, 2.0]` |
| restitution | 创建 env 时 | `[0.0, 0.5]` |
| base 质量 | 创建 env 时 | 在默认值上加 `U[-1.0, 1.0] kg` |
| link 质量 | 创建 env 时 | 各非 base link 乘 `U[0.9, 1.1]` |
| base COM | 创建 env 时 | `x/y/z` 各自加 `U[-0.03, 0.03] m` |
| reset 初速度 | 每次 reset | `root_states[:, 7:13]` 置为 `U[-0.5, 0.5]` |
| 外力推搡 | 每 `4 s` | base 线速度 `x/y` 置为 `U[-0.4, 0.4]`，角速度 `x/y/z` 置为 `U[-0.6, 0.6]` |

### 6.3 控制链随机化

| 项目 | 发生时机 | 当前范围/规则 |
| --- | --- | --- |
| PD 刚度倍率 | 每次 reset | `U[0.9, 1.1]` |
| PD 阻尼倍率 | 每次 reset | `U[0.9, 1.1]` |
| motor zero offset | 每次 reset | `U[-0.035, 0.035]` |
| motor strength | 每次 reset | `U[0.8, 1.2]` |
| action delay | 每个 env step | 在 4 个 decimation 子步里随机切换到新动作，相当于延迟 `{0, 5, 10, 15, 20} ms` |

### 6.4 观测与感知随机化

#### 6.4.1 actor 观测噪声

当前 actor 观测噪声开启：

```python
noise.add_noise = True
noise_level = 1.0
```

对 `Go2ParkourRobot` 而言，实际加噪的是 actor 的 45 维观测，且使用 `Go2Robot._get_noise_scale_vec()`，因此对应的有效噪声幅值为：

| 观测项 | 噪声 scale 配置 | 最终幅值 |
| --- | ---: | ---: |
| `base_ang_vel` | `0.2` | `0.2 * 0.25 = 0.05` |
| `projected_gravity` | `0.05` | `0.05` |
| `commands` | `0.0` | `0.0` |
| `dof_pos` | `0.01` | `0.01` |
| `dof_vel` | `1.5` | `1.5 * 0.05 = 0.075` |
| `previous actions` | `0.0` | `0.0` |

噪声形式为逐维独立的均匀噪声：

$$
o = o + (2u-1)\cdot s,\quad u \sim \mathcal{U}(0,1)
$$

#### 6.4.2 视觉开关随机化

这项随机化是当前训练里很有代表性的设计。

runner 会对 easy terrain 上的环境周期性翻转 `mask_vision`：

- easy terrain：`ramp`、`stairs_up`、`air_stones`、`flat`
- 翻转周期：每 `20` 个训练 iteration

这等价于在训练中显式制造“视觉可用/不可用”两种模式，逼 estimator 学会双模态和纯本体两条路径。

但要强调：

- 这是 **训练时启用**
- 当前 **推理时没有真正关掉视觉**

## 7. 训练配置

### 7.1 PPO 主训练配置

| 项目 | 当前值 |
| --- | --- |
| `seed` | `1` |
| `num_envs` | `8192` |
| `num_steps_per_env` | `24` |
| 单次 rollout 样本数 | `8192 * 24 = 196608` |
| `max_iterations` | `30000` |
| `save_interval` | `100` |
| `learning_rate` | `2e-4` |
| `schedule` | `adaptive` |
| `desired_kl` | `0.01` |
| `clip_param` | `0.2` |
| `gamma` | `0.99` |
| `lam` | `0.95` |
| `entropy_coef` | `0.01` |
| `value_loss_coef` | `1.0` |
| `use_clipped_value_loss` | `True` |
| `num_learning_epochs` | `5` |
| `num_mini_batches` | `4` |
| 每个 mini-batch 大小 | `196608 / 4 = 49152` |
| `max_grad_norm` | `1.0` |

### 7.2 Estimator 结构配置

| 项目 | 当前值 |
| --- | --- |
| `history_length` | `10` |
| `proprio_dim` | `45` |
| `expert_num` | `4` |
| `vision_token_dim` | `64` |
| `shared_state_dim` | `64` |
| `expert_hidden_dim` | `64` |
| `latent_dim` | `16` |
| `terrain_latent_dim` | `32` |
| `terrain_map_dim` | `187` |
| `feet_height_dim` | `4` |
| `num_terrain_types` | `12` |
| `transformer_num_layers` | `1` |
| `transformer_nhead` | `4` |
| `transformer_ff_dim` | `256` |
| `terrain_window_length` | `10` |
| `shared_state_alpha` | `0.5` |

### 7.3 Estimator 的训练目标与损失

当前 estimator 的总损失由以下几部分组成：

$$
\mathcal{L}_{est}
=
\mathcal{L}_{ht}
+\mathcal{L}_{mt}
+\mathcal{L}_{vt}
+\mathcal{L}_{obs\_recon}
+\lambda_{tid}\mathcal{L}_{terrain\_id}
+\lambda_{zkl}\mathcal{L}_{zkl}
+\lambda_{lb}\mathcal{L}_{load\_balance}
+\lambda_{swav}\mathcal{L}_{terrain\_swav}
$$

具体如下：

| 损失项 | 形式 | 当前权重/调度 |
| --- | --- | --- |
| `ht_loss` | `MSE(h_tf, gt_h_tf)` | `1.0` |
| `mt_loss` | `MSE(m_hat, gt_m_t)` | `1.0` |
| `vt_loss` | `MSE(v_t, gt_v_t)` | `1.0` |
| `reconstruction_loss` | `MSE(o_hat, gt_next_obs)` | `1.0` |
| `terrain_id_loss` | `CrossEntropy(terrain_logits, terrain_id)` | `terrain_id_loss_coef = 1.0` |
| `z_kl_loss` | 高斯 KL | 从 `0` 线性 warmup 到 `1e-2`，区间 `0 -> 5000` 次 estimator update |
| `load_balance_loss` | gate 使用均衡损失 | `load_balance_coef = 0.01` |
| `terrain_swav_loss` | gate-view 与 map-view 的跨视角 SwAV | 从 `0` warmup 到 `0.02`，区间 `2000 -> 10000` 次 estimator update |

另外还有一个独立优化器训练 `DepthAutoEncoder`：

| 项目 | 当前值 |
| --- | --- |
| `depth_learning_rate` | `1e-3` |
| `image_recon_batch_size` | `1024` |
| 损失 | 深度图重建 MSE |

### 7.4 训练流程上的关键 runner 配置

| 项目 | 当前值 | 作用 |
| --- | --- | --- |
| `camera.update_interval` | `5` env steps | estimator 与 depth buffer 的刷新频率 |
| `vision_toggle_interval` | `20` iterations | easy terrain 上 vision mask 的翻转周期 |
| `easy_terrain_names` | `["ramp", "stairs_up", "air_stones", "flat"]` | 只在这些地形上做视觉开关训练 |
| `enable_timing` | `True` | 开启性能计时 |
| `timing_sync_cuda` | `True` | 计时时同步 CUDA |

### 7.5 当前实现中“存在但未实际生效”的训练项

当前 `PPOParkourMoE.__init__()` 里还有几个参数：

- `gate_smooth_coef`
- `cycle_consistency_coef`
- `vision_consistency_coef`

但在当前 `update_estimator()` 总损失里并没有真正使用它们。  
所以如果只看当前仓库的实际训练逻辑，这 3 项应视为“预留接口”，而不是当前生效的损失项。

## 8. 当前实现的几个注意事项

### 8.1 当前 `terrain_id` 监督是 12 类，但其中 2 类没有样本

当前分类头输出 `12` 维 one-hot，并把 argmax 后的 one-hot 拼进 `mcp_code`。  
但由于 `step_stone` 和 `corridor` 当前比例为 `0.0`，所以这两类在当前训练配置下实际上没有数据覆盖。

### 8.2 当前推理阶段不会自动关闭视觉

虽然训练阶段会对 easy terrain 翻转 `mask_vision`，而且推理阶段也会计算 depth autoencoder 的重建误差，但：

- 推理时真正根据 `selector_threshold` 去设置 `mask[self.easy_idx_tensor]` 的代码是注释掉的
- 所以当前部署行为是 **始终使用视觉**

如果后面需要“视觉失效 fallback”，需要把推理路径里的 selector 逻辑重新打开并验证。

### 8.3 `mcp_code` 用的是 `z_mu` 而不是采样后的 `z_t`

这意味着 actor 输入的中间控制码是确定性的。  
当前 stochastic latent 更像是 estimator 训练时的辅助约束，而不是直接给策略引入随机潜变量。

这对部署是有利的，因为：

- actor 输入更稳定
- 导出的策略行为更可重复
