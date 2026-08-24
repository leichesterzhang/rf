# RM75 正向 Trot 平地/斜坡训练：奖励函数与新手上手指南

> 适用任务：`rm75_trot_slope`  
> 整理日期：2026-08-24  
> 本文以当前工作区代码为准，目标是帮助第一次接触本项目和强化学习的读者理解“机器人为什么会学成现在这样”。

## 1. 先建立一个最重要的直觉

强化学习里没有人直接告诉策略“哪条腿现在应该抬起来”。环境每隔一个控制周期观察机器人状态，给策略一个总分。策略经过大量尝试，逐渐找到能让长期总分更高的动作方式。

本任务每一步的总奖励可以简化为：

```text
机器人状态、指令、接触信息
        ↓
各个原始 reward/cost 函数
        ↓
原始值 × 配置 scale × 控制周期 dt × 可选课程系数
        ↓
所有项相加
        ↓
若 episode 非正常结束，再加 termination 惩罚
        ↓
PPO 用一段时间内的累计回报更新策略
```

这里有三个必须记住的规则：

1. `scale > 0` 通常表示奖励，`scale < 0` 通常表示惩罚。
2. 大多数惩罚函数返回的是大于等于 0 的“代价”，再乘负系数。
3. 不能只比较 scale 大小。比如关节加速度原始值可能很大，所以即使 scale 只有 `-1e-8` 也可能产生可见影响。

## 2. 当前任务到底在训练什么

当前任务不是全向运动任务，而是一个只向世界坐标 `+X` 前进的对角 trot 任务。

| 项目 | 当前设置 |
| --- | --- |
| 任务名 | `rm75_trot_slope` |
| 运动模式 | 只允许 `forward` |
| 侧向速度 | 始终为 `0` |
| 目标航向 | 世界坐标 `0 rad`，即 `+X` |
| 地形 | 60% ramp、30% flat、10% rough flat |
| 最大坡度 | `35°` |
| episode 上限 | `20 s` |
| 物理仿真周期 | `0.005 s` |
| 控制 decimation | `4` |
| 策略控制周期 | `dt = 0.02 s`，即 `50 Hz` |
| 每个 PPO iteration 的采样长度 | 每环境 `24` 个控制步，即 `0.48 s` |
| 动作 | 12 个腿部关节的位置残差，`action_scale = 0.2` |

速度不是从一开始就直接设为最终值。当前采用只包含正向运动的速度课程：

| 学习迭代 | 平地目标 | ramp 目标 | 侧向目标 |
| ---: | ---: | ---: | ---: |
| `0–1999` | `0.8 m/s` | `0.4 m/s` | `0` |
| `2000–5999` | `1.5 m/s` | `0.8 m/s` | `0` |
| `6000–11999` | `2.2 m/s` | `1.1 m/s` | `0` |
| `12000+` | `3.0 m/s` | `1.5 m/s` | `0` |

这样做的原因是：从随机策略直接要求 `3 m/s`，策略很容易进入“倾倒—重置—再次倾倒”的坏循环。现在使用已有 checkpoint 提供站立和基础步态能力，再只针对正向任务进行微调。

## 3. 奖励代码应该从哪里看

奖励配置和实现分散在继承链中。阅读顺序建议如下：

1. `legged_gym/envs/rm75/rm75_config_trot_slope.py`
   - 当前任务的速度、地形、终止阈值和奖励 scale。
2. `legged_gym/envs/rm75/rm75_trot_slope_env.py`
   - 正向速度跟踪、trot 对角腿、打滑、航向和坡面姿态奖励。
3. `legged_gym/envs/go2/go2_parkour_env.py`
   - 低速、髋关节、基础步态等继承项。
4. `legged_gym/envs/go2/go2_env.py`
   - `hip_to_default` 的底层实现。
5. `legged_gym/envs/base/legged_robot.py`
   - 通用稳定性、能耗、动作平滑、接触力和奖励汇总逻辑。

函数名和配置名通过约定自动关联：

```text
scales.tracking_lin_vel
        ↓
_reward_tracking_lin_vel()
```

只要 scale 非零，`_prepare_reward_function()` 就会寻找对应的 `_reward_<name>()` 并在每个控制步调用。scale 为 0 的项目会被移除，不参与计算。

## 4. 奖励系数为什么还要乘 dt

初始化环境时，代码会把每个非零 scale 乘以控制周期：

```python
effective_scale = configured_scale * dt
```

当前 `dt = 0.02 s`。例如：

```text
tracking_lin_vel 配置值 = 3.0
单步实际系数          = 3.0 × 0.02 = 0.06
```

当速度完全匹配时，`tracking_lin_vel` 的原始值为 1，因此每控制步贡献约 `+0.06`。若连续保持 1 秒，即 50 个控制步，仅这一项累计约 `+3.0`。

使用 dt 的好处是：改变控制频率时，每秒累计奖励不会按步数成倍变化。

## 5. 当前完整奖励项

下面列出的都是当前配置中 scale 非零的项目。“单步基础系数”已经乘过 `dt=0.02`，但还没有乘原始 reward 值和奖励课程系数。

### 5.1 任务目标与方向控制

| 名称 | 配置 scale | 单步基础系数 | 含义 |
| --- | ---: | ---: | --- |
| `tracking_lin_vel` | `+3.0` | `+0.06` | 奖励实际 XY 速度跟随指令，是最主要的任务奖励 |
| `tracking_ang_vel` | `+0.5` | `+0.01` | 奖励实际 yaw 角速度跟随航向控制器给出的 yaw 速度指令 |
| `low_speed_when_commanded` | `-1.0` | `-0.02` | 有移动指令时，实际速度低于最低要求就惩罚 |
| `trot_heading_error` | `-0.8` | `-0.016` | 直接惩罚机器人 yaw 偏离世界 `+X` |
| `terrain_normal_alignment` | `-0.75` | `-0.015` | 惩罚身体上方向与局部地形法向不一致，坡上应随坡面倾斜 |

#### `tracking_lin_vel`

当前只有 forward 模式，公式可近似写成：

```text
ex = vx_cmd - vx
ey = vy_cmd - vy

raw = exp(-(ex² / 0.50 + ey² / 0.20))
```

其原始值范围为 `(0, 1]`：

- 速度完全匹配：`raw = 1`。
- 前向误差 `0.5 m/s` 且无横向误差：`raw ≈ exp(-0.5) ≈ 0.607`。
- 前向误差 `1.0 m/s`：`raw ≈ exp(-2) ≈ 0.135`。

这里的 `0.50` 和 `0.20` 是平方误差的分母，不要简单理解为普通统计学里的标准差。横向分母更小，说明任务对横向漂移更敏感。

#### `tracking_ang_vel` 与 `trot_heading_error`

任务目标航向为 `0 rad`。环境先根据航向误差生成 yaw 角速度指令：

```text
yaw_rate_cmd = clip(0.5 × wrap(target_heading - current_heading))
```

然后：

- `tracking_ang_vel` 奖励实际 yaw 速度跟随 `yaw_rate_cmd`。
- `trot_heading_error` 直接惩罚 `wrap(target_heading - current_heading)²`。

二者一个管“当前应该怎样转”，一个管“最终朝向哪里”。

#### `low_speed_when_commanded`

当前阈值为：

```text
有命令：|cmd_xy| > 0.2 m/s
最低速度要求：min(|cmd_xy|, 0.25 m/s)
raw = max(最低速度要求 - |actual_xy|, 0)
```

它主要防止策略发现“原地不动也比较安全”这一投机解。它只要求先动起来，精确追踪仍由 `tracking_lin_vel` 负责。

#### `terrain_normal_alignment`

环境用机身附近的高度扫描拟合局部坡面法向量 `n_terrain`，再计算：

```text
raw = 1 - dot(body_up, n_terrain)
```

完全对齐时为 0；夹角越大惩罚越大。这个项在 35°坡面上非常重要，因为“世界竖直”并不是坡上最稳定的身体姿态。

### 5.2 Trot 步态与足端行为

脚的顺序固定为：

```text
0 = FL（左前）
1 = FR（右前）
2 = RL（左后）
3 = RR（右后）
```

对角腿对为 `FL + RR` 和 `FR + RL`。

| 名称 | 配置 scale | 单步基础系数 | 含义 |
| --- | ---: | ---: | --- |
| `trot_diagonal_sync` | `-1.5` | `-0.03 × 课程系数` | 同一对角腿的接触状态应一致 |
| `trot_diagonal_opposition` | `-1.0` | `-0.02 × 课程系数` | 两组对角腿不要同时支撑，鼓励交替 |
| `trot_foot_height_sync` | `-0.2` | `-0.004` | 同一对角腿抬脚高度应接近 |
| `flat_gait` | `-0.25` | `-0.005` | 抑制过早离地、接触占空比过低和后腿接触不平衡 |
| `feet_slip` | `-0.25` | `-0.005 × 课程系数` | 脚接触地面时，惩罚水平滑动速度平方 |
| `feet_regulation` | `-0.02` | `-0.0004` | 摆动脚靠近地面时不应高速扫动，鼓励合理抬脚 |

这些项只在有移动命令并且地形 ID 为 ramp、flat 或 rough flat 时启用；当前任务全部地形都属于这个集合。

#### `trot_diagonal_sync`

令每只脚的接触状态 `c` 为 0 或 1：

```text
raw = 0.5 × (|c_FL - c_RR| + |c_FR - c_RL|)
```

同一对角腿同时落地或同时腾空时不受罚。一只落地、另一只腾空时受罚。

#### `trot_diagonal_opposition`

```text
pair_A = 0.5 × (c_FL + c_RR)
pair_B = 0.5 × (c_FR + c_RL)
raw = pair_A × pair_B
```

两组对角腿同时支撑时惩罚最大；全脚短暂腾空时两组都为 0，不受罚，因此允许高速 trot 中出现短暂飞行相。

#### `trot_foot_height_sync`

```text
raw = ((h_FL - h_RR)² + (h_FR - h_RL)²) / 0.05²
```

这个项让同一对角腿的摆动高度协调。由于分母 `0.05²` 很小，5 cm 的高度差已经会产生约 1 的原始代价。

#### `feet_slip`

```text
raw = Σ contact_i × (vx_foot_i² + vy_foot_i²)
```

只惩罚已经接触地面的脚。腾空脚快速摆动不会被这个项惩罚。

#### `flat_gait`

虽然名字叫 flat gait，但当前把 ramp、flat、rough flat 都列入启用范围。它组合了四类代价：

1. 支撑时间不足 `0.10 s` 就离地。
2. 四脚接触占空比低于 `0.25`。
3. 后脚接触占空比低于 `0.45`。
4. 左右后脚接触占空比差异超过 `0.12`。

接触占空比使用 `EMA alpha = 0.98` 平滑。该项更偏向“不要失去基本支撑”，对角 trot 的明确节奏主要由上面三个 `trot_*` 项决定。

### 5.3 身体稳定与姿态

| 名称 | 配置 scale | 单步基础系数 | 含义 |
| --- | ---: | ---: | --- |
| `lin_vel_z` | `-1.0` | `-0.02` | 惩罚机身竖直速度平方，减少上下弹跳 |
| `ang_vel_xy` | `-0.05` | `-0.001` | 惩罚 roll/pitch 角速度平方，减少翻滚和俯仰晃动 |
| `correct_base_height` | `-1.0` | `-0.02 × 课程系数` | 惩罚估计机身离地高度偏离 `0.53 m` |
| `hip_to_default` | `-0.02` | `-0.0004` | 惩罚四个髋关节偏离默认位置 |
| `stand_still` | `-0.5` | `-0.01` | 零速度指令下惩罚关节偏离默认姿态 |

其中：

```text
lin_vel_z raw = vz²
ang_vel_xy raw = wx² + wy²
correct_base_height raw = (estimated_height - 0.53)²
hip_to_default raw = Σ四个髋关节 |q_hip - q_default|
```

`correct_base_height` 使用机器人周围的高度扫描估计地面，而不是简单使用世界坐标 `root_z`，所以在坡面上也能得到有意义的离地高度。

`stand_still` 虽然 scale 非零，但它只在 `|cmd_xy| < 0.1 m/s` 时生效。当前任务始终给出至少 `0.4 m/s` 的前进命令，因此它实际上长期为 0，是一个配置上启用、当前任务中休眠的项目。

当前没有启用单独的通用 `orientation` 奖励。坡面姿态主要由 `terrain_normal_alignment`、高度、角速度和终止条件共同约束。

### 5.4 能耗与动作平滑

| 名称 | 配置 scale | 单步基础系数 | 原始量 |
| --- | ---: | ---: | --- |
| `torques` | `-2e-6` | `-4e-8` | `Σ torque²` |
| `dof_power` | `-2e-5` | `-4e-7` | `Σ |torque × joint_velocity|` |
| `dof_acc` | `-1e-8` | `-2e-10` | `Σ ((last_dq - dq) / dt)²` |
| `action_rate` | `-0.006` | `-0.00012` | `Σ (a_t - a_{t-1})²` |
| `action_smoothness` | `-0.006` | `-0.00012` | `Σ (a_t - 2a_{t-1} + a_{t-2})²` |

这些项分别限制：

- 电机输出过大。
- 机械功率消耗过大。
- 关节速度突变。
- 动作一阶变化过快。
- 动作二阶变化过快，即抖动或突然改变趋势。

`action_rate` 和 `action_smoothness` 看起来相似，但作用不同：前者偏向“小变化”，后者偏向“变化趋势连续”。二者共同减少高频抖动。

### 5.5 接触、安全和 episode 结束

| 名称 | 配置 scale | 单步基础系数 | 含义 |
| --- | ---: | ---: | --- |
| `collision` | `-2.0` | `-0.04` | thigh、calf、base 接触力模长超过阈值时按接触部件计数 |
| `feet_contact_forces` | `-0.002` | `-0.00004` | 四只脚接触力超过 `420 N` 的超出部分 |
| `dof_pos_limits` | `-2.0` | `-0.04` | 关节位置越过上下限的距离之和 |
| `termination` | `-20.0` | `-0.4` | 非正常 episode 结束时的一次性惩罚 |

`termination=-20` 并不表示结束时直接减 20，因为它也在初始化时乘了 `dt=0.02`，实际一次约为 `-0.4`。

正常达到 20 秒时间上限时，`_reward_termination()` 会排除 `time_out`，不会施加这项终止惩罚。跌倒、倾斜、无进展或跑出地形块等非正常 reset 会受到惩罚。

## 6. 奖励课程：惩罚不是从第一天就全部拉满

四个奖励项还有一个随迭代线性变化的额外乘数：

| 项目 | 起始乘数 | 结束乘数 | 变化区间 |
| --- | ---: | ---: | ---: |
| `correct_base_height` | `0.5` | `2.0` | `0–6000` |
| `trot_diagonal_sync` | `0.25` | `1.0` | `0–4000` |
| `trot_diagonal_opposition` | `0.25` | `1.0` | `0–4000` |
| `feet_slip` | `0.25` | `1.0` | `0–6000` |

课程乘数是乘在已有 scale 上，不是替换 scale。例如 `correct_base_height`：

```text
iteration 0:
-1.0 × 0.02 × 0.5 = -0.01

iteration 6000+:
-1.0 × 0.02 × 2.0 = -0.04
```

设计意图是：早期先让机器人保住基本运动能力，再逐步提高对姿态、trot 节奏和防滑的要求。

## 7. 终止条件不是奖励项，但同样决定策略行为

episode 提前结束意味着后续潜在正奖励全部丢失，同时通常还会得到 `termination` 惩罚。因此终止规则经常比某个小 scale 更有影响。

当前主要终止条件如下：

| TensorBoard 名称 | 条件 | 新手解释 |
| --- | --- | --- |
| `time_out_frac` | episode 超过 `20 s` | 正常结束，通常是好信号，不加 termination 惩罚 |
| `trot_base_contact_frac` | base 接触力模长超过 `1 N` | 身体碰地，立即 reset |
| `trot_excessive_tilt_frac` | `|roll| > 1.0 rad` 或 `|pitch| > 1.15 rad` | 机身倾斜约超过 `57°/66°` |
| `trot_no_progress_frac` | 有运动指令，但连续 `3 s` 未把最大前进距离提高至少 `0.08 m` | 卡住或原地踏步 |
| `out_of_block_frac` | 超出当前 `12 m × 5 m` 地形块边界，并计入 `1 m` margin | 通常表示走出了本地训练块；既可能是前进成功，也会触发 reset |
| `fall_base_height_cutoff_frac` | 世界坐标 base 高度低于阈值 | 掉入深坑；本任务平地/坡面通常少见 |
| `fall_foot_height_cutoff_frac` | 任一足端世界高度低于阈值 | 足端掉出有效地形区域 |
| `non_base_contact_frac` | 被配置为终止接触的非 base 部件碰撞 | 当前 asset 主要把 base 作为终止接触部件 |

同一步可能同时满足多个条件，所以各个 fraction 之和不保证严格等于 1。`multi_reason_frac` 用来观察多原因重叠。

## 8. 地形课程如何工作

地形共有 10 个等级，初始只在 0–1 级。对 ramp 而言，等级逐渐提高到最大 `35°`。

每次 reset 时，环境根据本 episode 的最大移动距离调整等级：

- 移动超过地形长度一半，即超过 `6 m`：升级一级。
- 移动距离明显低于指令要求：降级一级。
- 通过最高等级后：随机回到某个等级继续训练，避免只在最高难度过拟合。

因此 `Episode/terrain_level_*` 上升通常是能力提高的信号。但如果等级升得太快，同时 `trot_excessive_tilt_frac` 开始上升，就可能说明课程难度超过了当前策略承受能力。

## 9. TensorBoard 中每条曲线到底表示什么

启动方式示例：

```bash
/home/user/miniconda3/envs/unitree-rl/bin/tensorboard \
  --logdir=/home/user/5-28/logs/RM75_trot_slope \
  --host=127.0.0.1 \
  --port=6006
```

浏览器打开 `http://127.0.0.1:6006`。

### 9.1 `Train/mean_reward`

这是最近 100 个已完成 episode 的实际累计总奖励平均值。它包含所有 scale、dt、课程系数和 termination 惩罚。

观察重点：

- 长期趋势应上升。
- 短时间波动正常。
- 奖励上升但 episode 长度下降，可能是在利用某个奖励漏洞，不一定是真的变好。

### 9.2 `Train/mean_episode_length`

最近 100 个已完成 episode 的平均控制步数。当前上限约为：

```text
20 s / 0.02 s = 1000 steps
```

从几十步提高到数百步通常代表站立和稳定性显著改善。接近 1000 时应结合 `time_out_frac` 判断是否真的稳定跑满 20 秒。

训练启动时启用了随机初始 episode 长度，因此最初几轮会出现人为制造的短 episode 和 timeout。不要只根据 iteration 0–2 下结论。

### 9.3 `Episode/rew_<name>`

这些不是原始 reward，也不是单步 reward。环境记录的是：

```text
本 episode 中该项的已缩放累计值 / max_episode_length_s
```

当前除数固定为 `20 s`，即使 episode 只活了 2 秒也仍然除以 20。因此：

- 正奖励项通常为正。
- 惩罚项通常为负。
- episode 变长时，某些累计惩罚绝对值自然可能变大。
- 比较不同 run 时，要同时看 episode 长度和终止原因。

### 9.4 `Termination/*_frac`

这些比例按当次 rollout 中发生 reset 的环境数加权统计，不是全部并行环境的比例。

在只有 64 个环境的可视化训练中，如果一轮只有少数 episode 结束，`0%`、`50%`、`100%` 之间的大幅跳动可能只是小样本波动。正式 4096 环境的曲线通常更稳定。

建议重点关注：

1. `trot_excessive_tilt_frac`
2. `trot_base_contact_frac`
3. `trot_no_progress_frac`
4. `time_out_frac`
5. `out_of_block_frac`

### 9.5 `Episode/terrain_level_*`

- `terrain_level_all`：全部地形平均等级。
- `terrain_level_ramp`：坡面平均等级。
- `terrain_level_flat`：平地平均等级。
- `terrain_level_rough_flat`：粗糙平地平均等级。

最终目标不是所有曲线必须单调上升，而是在更高等级下仍能保持低跌倒率和较好的速度跟踪。

## 10. 如何判断一次训练是否健康

建议按下面顺序检查，不要只盯总奖励：

1. **先看是否活着**：`mean_episode_length` 是否从很短逐渐增加。
2. **再看怎样结束**：tilt/base contact 是否下降，timeout 是否增加。
3. **再看是否完成任务**：`rew_tracking_lin_vel` 是否提高，no progress 是否下降。
4. **再看课程**：terrain level 是否提高后仍保持稳定。
5. **最后看动作质量**：slip、contact force、power、action rate 是否没有持续恶化。

一个可操作的早期判断标准是：

- `trot_excessive_tilt_frac` 不应长期接近 1。
- 平均 episode 长度应逐渐超过 700 步。
- 速度跟踪奖励应总体上升。
- terrain level 上升时，奖励和 episode 长度不应持续崩塌。

这些是诊断门槛，不是最终性能指标。最终仍要用固定地形、固定速度的回放和独立评估环境测成功率。

## 11. 当前 178 次可视化训练案例怎么读

运行目录：

```text
logs/RM75_trot_slope/Aug24_14-10-32_forward_trot_finetune_visual_200
```

这次计划运行 200 次，但实际在 iteration 178 停止，因此只保存了 `model_0.pt` 和 `model_100.pt`。

前 171 次的主要趋势：

| 指标 | 开始 | iteration 171 |
| --- | ---: | ---: |
| `Train/mean_reward` | `0.00` | `16.37` |
| `Train/mean_episode_length` | `13.5` | `879.4` |
| `Episode/terrain_level_all` | `0.49` | `3.36` |
| `trot_excessive_tilt_frac` | 大部分为 `0` | `0` |

这说明热启动确实解决了“一开始站不起来”的主要问题。

但 iteration 172–178 出现：

- 倾斜终止率短时明显升高。
- 平均奖励从 `16.37` 降至 `11.57`。
- 平均 episode 长度从 `879` 降至 `754`。
- 速度跟踪奖励短时下降。

因此这次记录的正确结论不是“方案失败”，也不是“已经完全稳定”，而是：

> 站立和基础运动已恢复，但小规模 64 环境训练后段出现波动，需要用更大批量的短期试跑验证稳定性。

当前最可靠的已保存模型是 `model_100.pt`，因为 iteration 178 的内存状态没有形成 checkpoint。

## 12. 新手修改奖励时的安全流程

### 12.1 一次只改一个目标

不要同时改十个 scale。推荐流程：

1. 保存当前配置和 checkpoint。
2. 明确一个问题，例如“脚打滑”。
3. 找到对应 raw reward 和日志项。
4. 小幅修改，例如 20%–50%，不要直接放大 10 倍。
5. 用相同 seed、相同 checkpoint 做短训练对照。
6. 同时比较任务成功率和副作用。

### 12.2 先看 raw 量级，再决定 scale

建议临时记录某个 raw reward 的均值、最大值和分位数。目标是比较：

```text
raw_value × scale × dt
```

而不是只比较 scale。例如：

- `torques` 原始值可能是数千。
- `tracking_lin_vel` 原始值最多只有 1。
- 二者 scale 差很多是正常的。

### 12.3 警惕奖励冲突

常见冲突包括：

- 速度越高通常越耗能，`tracking_lin_vel` 与 `power/torques` 存在竞争。
- 高速 trot 需要短暂飞行相，过强的接触占空比惩罚可能压制速度。
- 过强的姿态约束可能让机器人在坡面上不敢调整身体。
- 过强的动作平滑可能导致响应太慢，爬坡时发不上力。

强化学习调参不是“所有好东西都加大”，而是在任务完成、稳定性、能耗和动作质量之间找平衡。

### 12.4 保留可复现信息

每次正式训练至少记录：

- `run_name`
- 起始 checkpoint
- 当前代码版本或修改内容
- seed
- `num_envs`
- 速度课程
- reward scales
- terrain 配置

本项目会在每个 run 目录保存 `config.yaml`。诊断历史训练时，优先读取该文件，不要假设它与当前源码完全一致。

## 13. 常见误区

### “奖励越大越好吗？”

只在同一套配置和同一归一化方式下有可比性。修改 scale 后，总奖励数值本身可能改变，不能直接与旧 run 横向比较。

### “负奖励项的曲线越接近 0 越好吗？”

通常是，但必须保证任务仍完成。机器人原地不动可以让能耗、碰撞、打滑都接近 0，却不是合格策略。

### “训练窗口里看起来能走，就说明训练成功了吗？”

不一定。训练环境带有随机地形、随机化和探索噪声；单个可视化环境也可能恰好处于容易地形。必须结合统计曲线和固定条件评估。

### “200 步训练就是仿真 200 步吗？”

这里命令行的 `--max_iterations=200` 表示 200 次 PPO 学习迭代。每次迭代每个环境采样 24 个控制步：

```text
64 环境：每 iteration 采样 64 × 24 = 1,536 transitions
4096 环境：每 iteration 采样 4096 × 24 = 98,304 transitions
```

### “为什么 iteration 0 的 episode 很短？”

runner 会随机化初始 `episode_length_buf`，让不同环境错开 reset，避免所有环境同时结束。因此最初几轮的 episode 长度和 timeout 不代表真实稳定性。

## 14. 推荐的学习顺序

如果你刚开始学习这套代码，建议依次掌握：

1. 看懂 `commands`：策略被要求做什么。
2. 看懂 observation：策略能看到什么。
3. 看懂 action 到 PD target 的映射：策略能控制什么。
4. 看懂本文列出的 reward：怎样算好坏。
5. 看懂 termination/reset：一次尝试何时结束。
6. 看懂 terrain/speed curriculum：难度怎样变化。
7. 最后再进入 PPO、GAE、actor-critic 和 MoE/estimator。

先把“环境闭环”看懂，再学习 PPO 数学，会比一开始就钻网络结构更容易建立整体认识。

## 15. 术语速查

| 术语 | 含义 |
| --- | --- |
| observation | 策略当前能看到的状态输入 |
| action | 策略输出，本任务中为 12 个关节位置残差 |
| command | 环境希望机器人达到的速度和航向 |
| reward | 希望策略增加的得分 |
| cost/penalty | 希望策略减少的代价，通常乘负 scale |
| episode | 从 reset 到下一次 reset 的一段完整尝试 |
| rollout | PPO 更新前收集的一批连续交互数据 |
| iteration | 一次 rollout 收集加一次 PPO 更新 |
| curriculum | 随训练进度逐步提高速度、惩罚或地形难度 |
| checkpoint | 策略、估计器以及可选优化器状态的保存文件 |
| policy/actor | 根据 observation 生成 action 的网络 |
| critic | 估计状态长期价值、辅助训练 actor 的网络 |
| PPO | 当前用于更新策略的强化学习算法 |

## 16. 当前最关键的源码入口

- 任务配置：`legged_gym/envs/rm75/rm75_config_trot_slope.py`
- 任务奖励：`legged_gym/envs/rm75/rm75_trot_slope_env.py`
- 通用 parkour 奖励：`legged_gym/envs/go2/go2_parkour_env.py`
- 基础奖励汇总：`legged_gym/envs/base/legged_robot.py`
- 训练入口：`legged_gym/scripts/train.py`
- PPO runner 与日志：`rsl_rl/rsl_rl/runners/onpolicyrunner_parkour_moe.py`
- PPO 更新：`rsl_rl/rsl_rl/algorithms/ppo_parkour_moe.py`
- 可视化回放：`legged_gym/scripts/play.py`

阅读时始终区分三层数值：原始函数返回值、配置 scale、最终写入 reward buffer 的已缩放值。理解这三层之后，大部分奖励调试问题都会清楚很多。
