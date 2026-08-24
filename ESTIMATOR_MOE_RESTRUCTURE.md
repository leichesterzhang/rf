# OnPolicyRunner + Estimator MoE 设计文档

## 1. 文档范围

本文档基于当前源码的实际执行路径整理，重点覆盖以下 4 个问题：

1. `OnPolicyRunnerWithExtractor` 当前真正经过的所有网络结构，以及它们之间的张量流向。
2. 当前所有观测输入的组成与维度。
3. `OnPolicyRunnerWithExtractor.learn_rl()` 的完整训练逻辑。
4. `Estimator` 的损失计算逻辑。

本文档关注的是“当前实际生效的代码”，不是历史版本，也不是配置文件里看起来应该生效但实际上已经被覆盖/忽略的逻辑。

## 2. 当前实际生效的训练路径

### 2.1 生效配置

当前 teacher 训练配置来自：

- `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_teacher_ppo_cfg.py`
- `logs/rsl_rl/unitree_go2_parkour/2026-03-14_16-24-38/params/agent.yaml`

其中实际生效的关键项是：

- `policy.class_name = ActorCriticWithEstimator`
- `algorithm.class_name = PPOWithExtractor`
- `depth_encoder = null`
- `num_steps_per_env = 24`
- `num_learning_epochs = 5`
- `num_mini_batches = 4`
- `learning_rate = 2e-4`，这是 PPO policy optimizer 的学习率
- `empirical_normalization = false`

### 2.2 当前 runner 真实经过的网络

当前 `OnPolicyRunnerWithExtractor` 在 `depth_encoder_cfg is None` 时走的是 RL 分支，因此真正会经过的网络是：

1. `ActorCriticWithEstimator`
2. `Estimator`，当前是 MoE 版本
3. `DepthAutoEncoder`，在推理接口里被当作 `selector`
4. `Memory(GRU)`，作为 `Estimator` 内部每个 expert 的时序记忆
5. `ParkourEstimatorCore`，作为 `Estimator` 的视觉+本体融合前端

另外还有一个“被初始化但不参与当前主链路”的网络：

6. `DefaultEstimator`

原因是：

- `OnPolicyRunnerWithExtractor.__init__()` 会先按配置实例化 `DefaultEstimator`
- 但 `PPOWithExtractor.__init__()` 内部直接重新创建了一个新的 `Estimator(...)`
- 因而当前训练真正更新和保存的是 `self.alg.estimator`，也就是 MoE `Estimator`
- 传入的 `estimator_ori(DefaultEstimator)` 在当前 RL 路径里没有进入前向或损失计算

## 3. 观测设计

### 3.1 observation group 总览

当前 teacher 环境只定义了两个 observation group：

1. `policy`
2. `depth_camera`

没有单独定义 `critic` group，也没有 `teacher` group。

这带来一个很重要的结果：

- `OnPolicyRunnerWithExtractor` 会把 `privileged_obs_type` 设为 `None`
- 所以 `privileged_obs = obs_all`
- critic 实际上吃的是 `policy` 观测本身，而不是单独的 privileged observation

### 3.2 `policy` 观测总维度

`policy` 观测由 `ExtremeParkourObservations` 生成，总维度为：

- `53 + 4 + 187 + 9 + 29 + 530 = 812`

具体组成如下。

| 区间 | 维度 | 含义 | 备注 |
| --- | ---: | --- | --- |
| `0:3` | 3 | `root_ang_vel_b * 0.25` | 机身角速度 |
| `3:5` | 2 | `imu_obs = [roll, pitch]` | 包装到 `[-pi, pi]` |
| `5:6` | 1 | `0 * delta_yaw` | 预留占位，恒为 0 |
| `6:7` | 1 | `delta_yaw` | 当前目标 yaw 偏差 |
| `7:8` | 1 | `delta_next_yaw` | 下一目标 yaw 偏差 |
| `8:10` | 2 | `0 * commands[:, 0:2]` | 占位，恒为 0 |
| `10:11` | 1 | `commands[:, 0:1]` | 线速度命令 x |
| `11:12` | 1 | `0 * env_idx_tensor` | 占位，恒为 0 |
| `12:13` | 1 | `0 * invert_env_idx_tensor` | 占位，恒为 0 |
| `13:25` | 12 | `joint_pos - default_joint_pos` | 关节位置误差 |
| `25:37` | 12 | `joint_vel * 0.05` | 关节速度 |
| `37:49` | 12 | `action_history_buf[:, -1]` | 上一步动作 |
| `49:53` | 4 | `_get_contact_fill()` | 4 足接触滤波结果 |
| `53:57` | 4 | `get_feet_heights()` | 4 足相对地面高度 |
| `57:244` | 187 | `measured_heights` | 地形高度图 |
| `244:253` | 9 | `priv_explicit` | 显式特权信息 |
| `253:282` | 29 | `priv_latent` | 隐式特权信息 |
| `282:812` | 530 | `_obs_history_buffer.view(...)` | 10 帧历史，每帧 53 维 |

### 3.3 `policy` 观测各部分更细的来源

#### 3.3.1 当前帧本体观测 `obs_buf`，53 维

`obs_buf` 的结构是：

- `3` 维机身角速度
- `2` 维 IMU
- `3` 维 yaw 相关，其中第 1 维恒为 0
- `4` 维 command/terrain 占位，其中只有 `command_x` 那 1 维非 0
- `12` 维关节位置误差
- `12` 维关节速度
- `12` 维上一步动作
- `4` 维接触状态

即：

- `3 + 2 + 3 + 4 + 12 + 12 + 12 + 4 = 53`

#### 3.3.2 `priv_explicit`，9 维

`priv_explicit = [base_lin_vel * 2, 0, 0]`

所以 9 维细分为：

- `0:3` 真正有值的 `base_lin_vel_b * 2.0`
- `3:6` 全 0
- `6:9` 全 0

这意味着 `gt_vt_step = privileged_obs[..., 244:247]` 实际监督的是“放大两倍后的机身线速度”。

#### 3.3.3 `priv_latent`，29 维

`priv_latent` 的组成是：

- `body_mass`: 1 维
- `body_com`: 3 维
- `friction_coeffs`: 1 维
- `(joint_stiffness / default_joint_stiffness) - 1`: 12 维
- `(joint_damping / default_joint_damping) - 1`: 12 维

合计：

- `1 + 3 + 1 + 12 + 12 = 29`

#### 3.3.4 历史观测 `history`，530 维

历史长度是 `10`，每帧历史维度是 `53`，所以：

- `10 * 53 = 530`

但这里有一个关键细节：

- 在把当前 `obs_buf` 推入历史缓冲区之前，代码会先将 `obs_buf[:, 6:8] = 0`，也就是清掉 `delta_yaw` 和 `delta_next_yaw`
- 同时将 `obs_buf[:, 10] = 0`，也就是清掉 `command_x`

因此：

- `policy` 观测前 53 维中的当前帧是带真实 yaw/command 的
- 历史 530 维中的每一帧，是“把当前帧的第 6、7、10 维强行置零”后的版本

这意味着 `Estimator` 看到的 10 帧历史不包含真实的 yaw target 和 command_x 信息。

### 3.4 `depth_camera` 观测

`depth_camera` 由 `image_features` 生成，输出形状是：

- `[B, 2, 58, 87]`

其中：

- `2` 是 `buffer_len = 2`，代表最近两帧深度图
- 每帧会先做裁剪：`[:, :-2, 4:-4]`
- 再 resize 到 `58 x 87`
- 再做归一化：`depth / clipping_range - 0.5`

初始化/reset 时，会把同一张处理后的深度图复制两次填满 buffer。

### 3.5 观测更新频率

`policy` 中的部分观测不是每个 env step 都刷新：

- `measured_heights` 只有在 `common_step_counter % 5 == 0` 时更新
- `depth_camera` 也只有在 `common_step_counter % 5 == 0` 时更新 buffer

所以当前设计实际是：

- 每 5 个 env step 刷新一次地形高度图和视觉输入
- runner 也恰好每 5 个 env step 执行一次 `Estimator` 前向与更新

这两者是对齐的。

## 4. 网络结构设计

## 4.1 总体张量流向

当前主链路可以概括为：

```text
policy obs [B, 812]
  ├─ 前 53 维 -> obs_now [B, 53]
  ├─ 后 530 维 -> proprio_hist [B, 530]
  └─ 前 282 维 -> critic 输入 [B, 282]

depth_camera [B, 2, 58, 87]
  ├─ -> Estimator(MoE) -> est_out
  └─ -> DepthAutoEncoder(selector) -> depth_recon

est_out:
  v_t [B, 3]
  h_tf [B, 4]
  z_mu [B, 16]
  z_tm [B, 32]

mcp_code = cat(v_t, h_tf, z_mu, z_tm) -> [B, 55]

actor_input = cat(mcp_code, obs_now) -> [B, 108]
critic_input = policy_obs[:, :282] -> [B, 282]

Actor -> actions [B, 12]
Critic -> value [B, 1]
```

## 4.2 `DefaultEstimator`，被实例化但当前主链路未使用

当前 config 中 `estimator.class_name = DefaultEstimator`，且 hidden dims 为 `[128, 64]`。

如果按 config 展开，这个网络结构是：

```text
input [B, 53]
  -> Linear(53, 128)
  -> ELU
  -> Linear(128, 64)
  -> ELU
  -> Linear(64, 9)
output [B, 9]
```

但在当前 RL 路径中，它没有真正进入训练主链路，原因见第 2.2 节。

## 4.3 `ActorCriticWithEstimator`

### 4.3.1 actor

actor 使用的输入不是完整 812 维观测，而是：

- 当前帧 `obs_now`: `[B, 53]`
- `Estimator` 产生的 `mcp_code`: `[B, 55]`

拼接后：

- `actor_input = [B, 108]`

其中 `55` 的组成是：

- `v_t`: 3 维
- `h_tf`: 4 维
- `z_mu`: 16 维
- `z_tm`: 32 维

actor MLP 的实际结构是：

```text
input [B, 108]
  -> Linear(108, 512)
  -> ELU
  -> Linear(512, 256)
  -> ELU
  -> Linear(256, 128)
  -> ELU
  -> Linear(128, 12)
output [B, 12]
```

动作分布为高斯分布：

- mean: actor 输出 `[B, 12]`
- std: 一个可学习的全局参数向量 `[12]`
- `noise_std_type = scalar`
- `init_noise_std = 1.0`

### 4.3.2 critic

critic 不吃 `mcp_code`，也不吃完整 812 维，而是硬编码只取前 `282` 维：

```text
critic_obs = critic_observations[:, :282]
```

这 282 维正好是：

- 当前帧 53 维本体状态
- 4 维 feet heights
- 187 维 measured heights
- 9 维 priv_explicit
- 29 维 priv_latent

结构为：

```text
input [B, 282]
  -> Linear(282, 512)
  -> ELU
  -> Linear(512, 256)
  -> ELU
  -> Linear(256, 128)
  -> ELU
  -> Linear(128, 1)
output [B, 1]
```

### 4.3.3 需要注意的实现细节

1. 构造函数虽然接收了 `num_critic_obs`，但内部立即写死为 `282`。
2. config 里的 `scan_encoder_dims` 和 `priv_encoder_dims` 在当前 `ActorCriticWithEstimator` 中没有实际参与建图。
3. 该类的 docstring 还是旧版本表述，看起来像“自己从扁平观测里解析 estimator 输入”，但当前真实实现已经改为“runner 先算出 `mcp_code`，再把 `mcp_code` 直接传给 actor”。

## 4.4 `Estimator`，当前主链路核心网络

`Estimator` 是当前真正训练的 MoE 估计器。

### 4.4.1 输入

- `proprio_seq`: `[B, 530]`
- `depth_seq`: `[B, 2, 58, 87]`
- `mask_vision`: `[B]`，`True` 表示使用视觉，`False` 表示盲走
- `obs_now`: `[B, 53]`

### 4.4.2 前端 `ParkourEstimatorCore`

#### 视觉分支

```text
input [B, 2, 58, 87]
  -> Conv2d(2, 16, k=3, s=2, p=1)  -> [B, 16, 29, 44]
  -> ELU
  -> Conv2d(16, 32, k=3, s=2, p=1) -> [B, 32, 15, 22]
  -> ELU
  -> Conv2d(32, 64, k=3, s=2, p=1) -> [B, 64, 8, 11]
  -> ELU
  -> flatten spatial
  -> permute
output image tokens [B, 88, 64]
```

因为：

- `8 * 11 = 88`

所以会得到 88 个视觉 token。

#### 本体历史分支

```text
input [B, 530]
  -> reshape -> [B, 10, 53]
  -> Linear(53, 256)
  -> ELU
  -> Linear(256, 128)
  -> ELU
  -> Linear(128, 64)
output proprio tokens [B, 10, 64]
```

#### vision flag token

`mask_vision` 会先变成：

- `vision_flag = mask_vision.float().unsqueeze(-1)`，形状 `[B, 1]`

然后编码成：

- `vision_input [B, 1, 64]`

#### token 拼接

最终拼接结果：

- image tokens: `88`
- proprio tokens: `10`
- vision flag token: `1`

总 token 数：

- `88 + 10 + 1 = 99`

即：

- `fused_input = [B, 99, 64]`
- `token_padding_mask = [B, 99]`

当 `mask_vision=False` 时：

1. 所有 image token 先被乘 0
2. 这些 image token 再被 `token_padding_mask=True` 屏蔽掉
3. 但 proprio token 和 vision token 仍然保留

所以 blind 模式下并不是整个 encoder 失效，而是“视觉 token 被静音并在 transformer 中被忽略”。

### 4.4.3 shared transformer + pooled feature

`Estimator` 在 `ParkourEstimatorCore` 后面接一个共享 transformer：

```text
input [B, 99, 64]
  -> TransformerEncoderLayer(d_model=64, nhead=1, ff=256)
  -> TransformerEncoder(num_layers=1)
output [B, 99, 64]
```

随后做 masked average pooling：

```text
[B, 99, 64] -> [B, 64]
```

再过一个共享头：

```text
[B, 64]
  -> Linear(64, 64)
  -> ELU
output gru_input [B, 64]
```

### 4.4.4 router

router 的输入是：

- `obs_now`: 53 维
- `vision_flag`: 1 维
- `gru_input`: 64 维

拼接后：

- `router_input = [B, 118]`

router 结构：

```text
input [B, 118]
  -> Linear(118, 128)
  -> ReLU
  -> Linear(128, 4)
  -> Softmax
output gating_weights [B, 4]
```

### 4.4.5 4 个 expert

当前共有 `4` 个 expert，每个 expert 结构相同：

```text
gru_input [B, 64]
  -> GRU(input_size=64, hidden_size=64, num_layers=1)
  -> output [1, B, 64]
  -> squeeze(0)
  -> Linear(64, 64)
expert_output [B, 64]
```

4 个 expert 堆叠后得到：

- `expert_outputs = [B, 4, 64]`

再通过门控加权：

```text
gating_weights.unsqueeze(1) [B, 1, 4]
  @ expert_outputs        [B, 4, 64]
  -> weighted_output      [B, 1, 64]
  -> squeeze(1)
  -> h_t                  [B, 64]
```

### 4.4.6 输出头

从 `h_t [B, 64]` 出发：

```text
v_t      = Linear(64, 3)   -> [B, 3]
h_tf     = Linear(64, 4)   -> [B, 4]
z_mu     = Linear(64, 16)  -> [B, 16]
z_logvar = Linear(64, 16)  -> [B, 16]
z_tm     = Linear(64, 32)  -> [B, 32]
```

然后执行 VAE reparameterization：

```text
std = exp(0.5 * z_logvar) -> [B, 16]
eps = randn_like(std)     -> [B, 16]
z_t = z_mu + eps * std    -> [B, 16]
```

### 4.4.7 decoder

#### 状态重建 decoder

输入：

- `[v_t, h_tf, z_t]`
- 维度为 `3 + 4 + 16 = 23`

结构：

```text
input [B, 23]
  -> Linear(23, 64)
  -> ELU
  -> Linear(64, 53)
output o_hat [B, 53]
```

#### 地形重建 decoder

输入：

- `z_tm [B, 32]`

结构：

```text
input [B, 32]
  -> Linear(32, 64)
  -> ELU
  -> Linear(64, 187)
output m_hat [B, 187]
```

### 4.4.8 SwAV / terrain contrast 分支

`Estimator` 还定义了一个用于地形对比学习的分支：

- `gate_terrain_projector`: `4 -> 64 -> 32`
- `elevation_terrain_projector`: `187 -> 64 -> 32`
- `terrain_prototypes`: `[32, 32]`

runner 会维护长度为 10 的时间窗：

- `swav_gate_window`: `[B, 10, 4]`
- `swav_map_window`: `[B, 10, 187]`
- `swav_valid_mask`: `[B, 10]`

然后用 Sinkhorn + prototype assignment 计算双向 SwAV loss。

### 4.4.9 当前定义了但没有进入 forward 主链的模块

下面这些层当前已经定义，但在 `Estimator.forward()` 里没有真正参与输出：

1. `fc_zt`
2. `map_unet`
3. `fc_zm_fine`

因此这些参数目前不会影响 `v_t / h_tf / z_mu / z_logvar / z_tm / o_hat / m_hat` 的计算结果。

## 4.5 `DepthAutoEncoder`，训练中作为重建器，推理中被当作 selector

### 4.5.1 结构

输入输出都是：

- 输入：`[B, 2, 58, 87]`
- 输出：`[B, 2, 58, 87]`

网络结构：

```text
input [B, 2, 58, 87]
  -> Conv2d(2, 16, k=3, s=2, p=1)  -> [B, 16, 29, 44]
  -> ELU
  -> Conv2d(16, 32, k=3, s=2, p=1) -> [B, 32, 15, 22]
  -> ELU
  -> Conv2d(32, 64, k=3, s=2, p=1) -> [B, 64, 8, 11]
  -> ELU
  -> flatten                           -> [B, 5632]
  -> Linear(5632, 128)                -> [B, 128]
  -> Linear(128, 5632)                -> [B, 5632]
  -> reshape                          -> [B, 64, 8, 11]
  -> ConvTranspose2d(64, 32, ...)     -> [B, 32, 15, 22]
  -> ELU
  -> ConvTranspose2d(32, 16, ...)     -> [B, 16, 29, 44]
  -> ELU
  -> ConvTranspose2d(16, 2, ...)      -> [B, 2, H, W]
  -> F.interpolate(size=(58, 87))
output x_hat [B, 2, 58, 87]
```

### 4.5.2 在 runner 中的角色

在训练时：

- 它只做深度图重建
- 更新目标就是 `MSE(depth_recon, depth_camera)`
- 不参与 actor/critic 前向
- 也不参与 `mask_vision` 切换决策

在推理/`play.py` 里：

- `get_selector_inference_policy()` 返回的就是这个 `DepthAutoEncoder`
- `play.py` 用它的 reconstruction loss 与阈值 `0.001` 比较
- 如果 loss 大于阈值，就把 easy terrain 的 `mask` 设为 0，切到 blind 模式
- 否则保留视觉模式

所以：

- “selector” 这个名字在当前代码里，本质上指的就是 `DepthAutoEncoder`
- 但这个 selector 决策逻辑只在推理脚本里显式使用，不在 `learn_rl()` 里使用

## 5. 关键中间张量

### 5.1 `mcp_code`

runner 在每次 estimator 前向后构造：

```text
mcp_code = cat(v_t, h_tf, z_mu, z_tm)
```

维度：

- `3 + 4 + 16 + 32 = 55`

即：

- `mcp_code = [B, 55]`

这是 actor 的直接条件输入。

### 5.2 `mcp_code_mu_log_mt`

runner 还构造了另一个张量用于 estimator loss：

```text
mcp_code_mu_log_mt = cat(z_mu, z_logvar, m_hat, o_hat)
```

维度：

- `16 + 16 + 187 + 53 = 272`

即：

- `mcp_code_mu_log_mt = [B, 272]`

在 loss 中切片方式是：

- `0:16` -> `z_mu`
- `16:32` -> `z_logvar`
- `32:219` -> `m_hat`
- `219:272` -> `o_hat`

### 5.3 SwAV 历史缓存

当 estimator 具有 `decoder_map` 和 `terrain_window_length` 时，runner 会创建：

- `swav_gate_hist`: `[num_envs, 9, 4]`
- `swav_map_hist`: `[num_envs, 9, 187]`
- `swav_valid_hist`: `[num_envs, 9]`

每次 estimator 更新时，会拼成当前窗口：

- `swav_gate_window`: `[num_envs, 10, 4]`
- `swav_map_window`: `[num_envs, 10, 187]`
- `swav_valid_mask`: `[num_envs, 10]`

## 6. OnPolicyRunner 完整训练逻辑

## 6.1 初始化阶段

`OnPolicyRunnerWithExtractor.__init__()` 的主要工作顺序如下：

1. 读取 `policy / estimator / algorithm / depth_encoder` 配置。
2. 通过 `env.get_observations()` 取第一批观测，得到 `num_obs`。
3. 初始化 `mask = 全 1`。
4. 读取 terrain name，构造：
   - `easy_idx_tensor = parkour_step / parkour_slope / parkour_flat`
   - `flat_idx_tensor = parkour_flat`
5. 判断当前是否有 `critic` 或 `teacher` group。
6. 构造 `DefaultEstimator`。
7. 构造 `ActorCriticWithEstimator`。
8. 在 PPOWithExtractor 内部重新构造真正使用的 `Estimator` 与 `DepthAutoEncoder`。
9. 初始化 rollout storage。
10. 初始化 SwAV 历史缓存。

## 6.2 每个 learning iteration 的全局节奏

每个 learning iteration 的结构是：

```text
for it in range(num_learning_iterations):
    1. 根据 terrain 切换 easy terrain 的 mask
    2. rollout 24 步
    3. compute_returns()
    4. PPO update()
    5. 记录 estimator/depth AE 的均值 loss
    6. 日志与保存
```

### 6.2.1 每 20 个 iteration 切换 easy terrain 的 blind/vision 状态

代码逻辑是：

```python
if it % 20 == 0:
    self.mask[self.easy_idx_tensor] = ~self.mask[self.easy_idx_tensor]
```

这里有两个关键点：

1. `easy terrain` 不是只有 flat，而是：
   - `parkour_step`
   - `parkour_slope`
   - `parkour_flat`
2. `mask` 初始是全 1，但在 `it == 0` 时就会立刻翻转一次

因此如果从头训练，实际节奏是：

- 第 `0~19` 轮：easy terrain 的 `mask_vision = False`
- 第 `20~39` 轮：easy terrain 的 `mask_vision = True`
- 第 `40~59` 轮：再次切回 `False`
- 以此类推

而非 easy terrain 会始终保持 `True`。

## 6.3 rollout 阶段，24 步/iteration

当前 `num_steps_per_env = 24`。

如果 teacher env config 保持默认 `6144` 个环境不变，则每个 iteration 会收集：

- `6144 * 24 = 147456` 条 transition

### 6.3.1 每逢 `common_step_counter % 5 == 0` 的步

这一步是“视觉/高度图更新步”，会做完整 estimator 更新。

顺序如下：

1. `privileged_obs = privileged_obs.clone().detach()`
2. `obs_now, proprio_hist = extract_obs_and_history(obs)`
3. 从 `privileged_obs` 切 ground truth：
   - `gt_ht_step = privileged_obs[..., 53:57]`
   - `gt_mt_step = privileged_obs[..., 57:244]`
   - `gt_vt_step = privileged_obs[..., 244:247]`
4. 读取 `camera_depth = additional_obs["depth_camera"]`
5. 调用 estimator：

```text
est_out = estimator(
    proprio_hist [B, 530],
    camera_depth [B, 2, 58, 87],
    mask_vision = self.mask [B],
    gt_mt_step = [B, 187],
    obs_now = [B, 53],
)
```

6. 调用 depth autoencoder：

```text
depth_recon, _ = depth_autoencoder(camera_depth)
```

7. 立刻更新 depth autoencoder：

```text
update_image_recon(depth_recon, camera_depth)
```

8. 构造 actor 用的 `mcp_code [B, 55]`
9. 构造 estimator loss 用的 `mcp_code_mu_log_mt [B, 272]`
10. 在 `torch.inference_mode()` 下调用 `self.alg.act(mcp_code, obs, privileged_obs)` 取动作
11. 执行 `env.step(actions)`
12. 取得下一时刻 `obs / rewards / dones / infos`
13. 如有需要做 obs normalizer
14. 构造下一时刻监督目标：

```text
gt_next_step = obs[..., :53]
gt_next_step[:, 6:8] = 0
gt_next_step[:, 10] = 0
```

15. 根据 `est_out["swav_gating_weights"]` 与 `gt_mt_step` 构造 SwAV 时间窗
16. 调用 `update_estimator(...)`
17. 更新 expert usage / gating entropy / SwAV 历史统计
18. `self.alg.estimator.detach_hidden_state()`

### 6.3.2 非 `% 5 == 0` 的步

这些步不会重新运行 estimator，而是直接复用上一次 `% 5 == 0` 时得到的 `mcp_code`：

1. `actions = self.alg.act(mcp_code.clone().detach(), obs, privileged_obs)`
2. `env.step(actions)`
3. 更新 `obs / rewards / dones / infos`
4. 处理 normalizer

因此：

- actor 在这 4 个中间步里依赖的是“最近一次 estimator 输出”
- estimator 并不是每一步都重算

## 6.4 每个 env step 的通用后处理

无论是不是 `% 5 == 0`，每步最后都会执行：

1. `self.alg.process_env_step(rewards, dones, infos)`
2. `self.alg.estimator.reset(dones)`
3. `self._reset_swav_history(dones)`
4. 更新 episode reward/length 日志缓存

其中 `process_env_step()` 会做：

- 记录 reward / done
- 处理 `time_outs` bootstrapping
- 把 transition 写入 rollout storage
- 清理临时 transition
- 调用 `policy.reset(dones)`

## 6.5 rollout 结束后

24 步 rollout 完成后：

1. `self.alg.compute_returns(privileged_obs)`
2. 调用 `self.alg.update()` 执行 PPO 更新
3. 把 rollout 期间累计的 estimator / AE / SwAV / gating 统计写入 `loss_dict`
4. 记录日志
5. 按 `save_interval=100` 保存模型

## 6.6 PPO update 细节

当前 PPO update 使用：

- `num_learning_epochs = 5`
- `num_mini_batches = 4`

所以每个 iteration 的 policy 参数更新次数是：

- `5 * 4 = 20`

### 6.6.1 rollout storage 中实际存了什么

当前 RL storage 关键字段包括：

- `observations`: `[24, num_envs, 812]`
- `privileged_observations`: `[24, num_envs, 812]`
- `mcp_code`: `[24, num_envs, 55]`
- `actions`: `[24, num_envs, 12]`
- `values`: `[24, num_envs, 1]`
- `actions_log_prob`: `[24, num_envs, 1]`
- `mu`: `[24, num_envs, 12]`
- `sigma`: `[24, num_envs, 12]`

### 6.6.2 PPO mini-batch 更新中真正重新前向了什么

在 `PPOWithExtractor.update()` 中，每个 mini-batch 会：

1. 用存下来的 `mcp_code_batch` 和 `obs_batch` 重新跑 actor
2. 用 `critic_obs_batch[:, :282]` 重新跑 critic
3. 计算 PPO surrogate loss
4. 计算 clipped value loss
5. 加上 entropy 正则
6. 更新 policy 参数

注意：

- estimator 不参与 PPO mini-batch 的反向传播
- estimator 已经在 rollout 过程中 online update 完成

## 7. Estimator 损失计算逻辑

## 7.1 监督目标来源

当前 estimator 的监督目标来自以下切片：

### `gt_ht_step`

```text
gt_ht_step = privileged_obs[..., 53:57]  -> [B, 4]
```

对应 4 个足端相对地面高度。

### `gt_mt_step`

```text
gt_mt_step = privileged_obs[..., 57:244] -> [B, 187]
```

对应地形高度图。

### `gt_vt_step`

```text
gt_vt_step = privileged_obs[..., 244:247] -> [B, 3]
```

对应 `priv_explicit` 的前 3 维，也就是 `base_lin_vel * 2.0`。

### `gt_next_step`

```text
gt_next_step = next_obs[..., :53]
gt_next_step[:, 6:8] = 0
gt_next_step[:, 10] = 0
```

这是下一时刻的 53 维本体观测，但仍保持与历史缓冲区一致的“yaw/command 清零版本”。

## 7.2 `update_estimator()` 的输入对应关系

`update_estimator()` 收到两个关键张量：

### `est_buffer = mcp_code`

结构为：

```text
0:3   -> v_t
3:7   -> h_tf
7:23  -> z_mu
23:55 -> z_tm
```

### `est_buffer_add = mcp_code_mu_log_mt`

结构为：

```text
0:16    -> z_mu
16:32   -> z_logvar
32:219  -> m_hat
219:272 -> o_hat
```

## 7.3 分项 loss

### 7.3.1 `ht_loss`

```text
ht_loss = MSE(est_buffer[..., 3:7], gt_ht_buffer)
```

即：

- `h_tf [B, 4]` 对齐 `gt_ht_step [B, 4]`

### 7.3.2 `mt_loss`

```text
mt_loss = MSE(est_buffer_add[..., 32:32+187], gt_mt_buffer)
```

即：

- `m_hat [B, 187]` 对齐 `gt_mt_step [B, 187]`

### 7.3.3 `vt_loss`

```text
vt_loss = MSE(est_buffer[..., :3], gt_vt_buffer)
```

即：

- `v_t [B, 3]` 对齐 `gt_vt_step [B, 3]`

### 7.3.4 `reconstruction_loss`

```text
reconstruction_loss = MSE(
    est_buffer_add[..., 32+187:32+187+53],
    gt_next_obs_buffer
)
```

即：

- `o_hat [B, 53]` 对齐 `gt_next_step [B, 53]`

## 7.4 VAE KL 项

KL 使用的是：

```text
z_mu     = est_buffer_add[..., :16]
z_logvar = est_buffer_add[..., 16:32]
```

公式为：

```text
KL = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
```

再对 batch 求均值：

```text
z_kl_loss = gaussian_kl(z_mu, z_logvar).mean()
```

## 7.5 总损失组合

### 7.5.1 estimation_loss

```text
estimation_loss = ht_loss + mt_loss + vt_loss
```

### 7.5.2 vae_loss

```text
vae_loss = reconstruction_loss + z_kl_loss
```

### 7.5.3 terrain_swav_loss

这个分支不是始终开启的，而是随 `estimator_update_counter` 逐步 ramp up：

- 前 `2000` 次 estimator 更新，权重为 `0`
- 之后在 `8000` 步内线性增长
- 最大权重为 `0.02`

即：

```text
terrain_swav_weight = 0,                                step < 2000
terrain_swav_weight = 0.02 * (step-2000)/8000,          2000 <= step < 10000
terrain_swav_weight = 0.02,                             step >= 10000
```

SwAV loss 本身是：

```text
terrain_swav_loss = cross_view_swav(gate_window, map_window)
```

### 7.5.4 最终 estimator 总损失

```text
vae_all_vp_loss = estimation_loss + vae_loss + terrain_swav_weight * terrain_swav_loss
```

展开后就是：

```text
total_estimator_loss =
    ht_loss
  + mt_loss
  + vt_loss
  + reconstruction_loss
  + z_kl_loss
  + terrain_swav_weight * terrain_swav_loss
```

没有额外系数，全部默认权重为 1，只有 `terrain_swav_loss` 带 schedule weight。

## 7.6 optimizer 与更新方式

当前 estimator 更新方式是：

1. `self.estimator_optimizer.zero_grad()`
2. `total_estimator_loss.backward()`
3. `clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)`
4. `self.estimator_optimizer.step()`
5. `self.estimator_update_counter += 1`

这是 rollout 内在线更新，不等 PPO mini-batch。

## 8. DepthAutoEncoder 的损失逻辑

当前 `DepthAutoEncoder` 的训练非常直接：

```text
recon_loss = MSE(pred_depth, gt_depth)
```

然后：

1. `zero_grad()`
2. `recon_loss.backward()`
3. `clip_grad_norm_()`
4. `optimizer.step()`

需要注意两个细节：

1. `DepthAutoEncoder.forward()` 自己已经返回了一个内部 `loss`
2. 但 runner 并不使用这个内部 `loss`
3. runner 会再次调用 `update_image_recon(pred_depth, gt_depth)` 重新算一遍 MSE 并回传

因此当前真正生效的是 `update_image_recon()` 里的 loss。

## 9. 当前实现与配置不完全一致的地方

这一节非常重要，因为它解释了“配置里写了，但当前代码并没有按那个配置执行”的部分。

| 项目 | 配置/表面现象 | 当前实际行为 |
| --- | --- | --- |
| `estimator.class_name = DefaultEstimator` | 看起来 estimator 是一个小 MLP | 当前训练真正用的是 `PPOWithExtractor` 内部硬编码新建的 MoE `Estimator` |
| `estimator.learning_rate = 1e-4` | 看起来 estimator 学习率是 `1e-4` | 实际 `estimator_optimizer = Adam(..., lr=1e-3)`，被硬编码覆盖 |
| `scan_encoder_dims / priv_encoder_dims` | 看起来 actor 还带 scan encoder / priv encoder | 当前 `ActorCriticWithEstimator` 没有使用这些层 |
| `num_critic_obs` 由外部传入 | 看起来 critic 输入维度跟 env 一致 | 实际在 `ActorCriticWithEstimator` 中直接写死为 `282` |
| selector | 看起来是一个独立 selector 网络 | 当前就是 `DepthAutoEncoder`，训练中只做重建，推理时才用 loss 做选择 |
| `train_with_estimated_states` | 看起来会影响训练路径 | 当前只是被保存到成员变量，没有真正控制训练分支 |
| `fc_zt / map_unet / fc_zm_fine` | 已经定义在 estimator 中 | 当前 forward 不使用 |
| docstring 中的 actor/estimator 描述 | 看起来 actor 会直接解析完整扁平观测 | 当前实际上是 runner 外部先算 `mcp_code`，再传入 actor |

## 10. 一句话总结

当前系统的真实结构可以概括成：

- `policy obs(812)` 提供当前本体状态、地形图、privileged 信息和 10 帧历史
- `depth_camera(2x58x87)` 提供最近两帧深度图
- `Estimator(MoE)` 每 5 步融合 `proprio_hist + depth_camera`，输出 `mcp_code(55)`
- actor 用 `mcp_code(55) + obs_now(53)` 输出 12 维动作
- critic 只看 `policy obs` 的前 282 维
- easy terrain 每 20 个 iteration 在 vision / blind 之间切换一次
- estimator 在 rollout 内在线更新，损失由 `vt + ht + mt + next_obs 重建 + KL + SwAV` 组成
- `DepthAutoEncoder` 只做深度图重建，推理阶段被当作 selector 使用
