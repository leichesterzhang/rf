# Parkour MoE Estimator-Actor-Critic 框架说明

## 1. 文档范围

本文档基于当前工作区里的最新实现，说明这套 `estimator-actor-critic` 框架在代码中的实际结构、各模块作用、设计意图、输入输出维度，以及训练与推理时的整体数据流向。

本文档重点对应以下文件：

- `rsl_rl/rsl_rl/modules/actor_critic_parkour_moe.py`
- `rsl_rl/rsl_rl/algorithms/ppo_parkour_moe.py`
- `rsl_rl/rsl_rl/runners/onpolicyrunner_parkour_moe.py`
- `rsl_rl/rsl_rl/storage/rollout_storage_parkour_moe.py`
- `legged_gym/envs/go2/go2_parkour_env.py`
- `legged_gym/envs/go2/go2_config_parkour_moe.py`
- `legged_gym/envs/go2/go2_env.py`
- `legged_gym/envs/base/legged_robot_config.py`
- `deploy/deploy_mujoco/deploy_go2.py`
- `deploy/deploy_real/deploy_real_go2_parkour_moe.py`

本文档强调的是“当前代码实际生效的路径”，不是历史方案，也不是仅在配置里出现但当前并未真正参与计算的预留项。

## 2. 一句话总览

这套框架的核心思想可以概括为：

1. 用环境提供的当前本体观测、历史本体观测、代理深度图和特权观测，构成训练所需的多源信息。
2. 用 `ParkourEstimator` 把“历史 proprio + 深度图 + 视觉开关”编码成一个紧凑的中间控制表征 `mcp_code`。
3. 用 `ActorCriticParkourMoE` 把 `mcp_code` 与当前 actor 观测拼接后输出动作，同时用 critic 的特权观测做价值评估。
4. 用 `PPOParkourMoE` 同时训练 actor-critic、估计器和深度自编码器，其中估计器除了监督速度/足端高度/地形图，还带有 latent KL、负载均衡和跨视角 SwAV 约束。
5. 在推理阶段，depth autoencoder 还额外承担一个 selector 的角色，用来判断某些简单地形上是否继续信任视觉输入。

如果画成最简主链路，可以写成：

```text
obs_now + obs_history + depth_buffer + privileged_obs
    -> ParkourEstimator
    -> mcp_code
    -> Actor
    -> action

privileged_obs
    -> Critic
    -> value
```

但真正完整的实现比这个更复杂，因为它还包含：

- 视觉与本体 token 化
- Transformer 编码
- attention pooling
- 共享记忆 `shared_state`
- Read/Write expert 机制
- 地形 SwAV 约束
- 深度重建 selector
- easy terrain 上的 vision on/off 训练策略

## 3. 代码中的整体模块分层

当前实现可以分成 7 层。

### 3.1 环境层

环境层负责生成：

- actor 当前观测 `obs`
- critic 特权观测 `privileged_obs`
- 附加观测 `additional_obs["depth_camera"]`
- 地形类型、easy terrain 索引、reset 信息

其中 Parkour 版本环境 `Go2ParkourRobot` 还额外做了三件事：

1. 把深度相机改为 `proxy` 模式，而不是直接走真实图像传感器。
2. 用高度场 raycast 合成深度图，并维护最近 `buffer_len=2` 帧的深度缓存。
3. 在普通 privileged observation 基础上追加 4 维足端高度，供 estimator 监督。

### 3.2 Runner 层

`OnPolicyRunnerParkourMoE` 负责把环境、算法和网络真正串起来。

它的职责包括：

- 初始化 actor-critic、estimator、depth autoencoder 和 PPO 算法对象
- 维护本体历史缓存 `history`
- 维护 vision mask 与 easy terrain 索引
- 控制 estimator 的刷新频率
- 组织一轮 rollout 的采样、估计器更新、PPO 更新与日志记录
- 暴露推理 policy 和导出 bundle

### 3.3 Estimator 层

`ParkourEstimator` 是整套框架里最关键的“感知到控制中间表征”模块。

它内部又分成：

1. `ParkourEstimatorCore`
2. `TransformerEncoder`
3. `attention_pool`
4. `router`
5. `ReadWriteExpert` 多专家模块
6. `shared_state` 共享记忆更新器
7. 多个预测头与解码器

### 3.4 Policy 层

`ActorCriticParkourMoE` 很清晰：

- actor 只看 `mcp_code + obs_now`
- critic 只看 `privileged_obs`

也就是说这是一个典型的不对称训练结构：

- actor 学的是可部署输入
- critic 用的是训练期特权信息

### 3.5 Algorithm 层

`PPOParkourMoE` 不是只训练 PPO。

它同时维护三套优化对象：

1. actor-critic optimizer
2. estimator optimizer
3. depth autoencoder optimizer

所以这套系统实际上是“PPO 主策略 + estimator 辅助学习 + depth selector 自监督”的联合训练框架。

### 3.6 Storage 层

`RolloutStorageParkourMoE` 除了存 PPO 常规的：

- obs
- critic obs
- actions
- rewards
- dones
- values
- log prob

还会额外存：

- `mcp_code`

这说明 PPO 更新时 actor 重新前向所使用的 estimator 输出不是临时再算一遍，而是直接复用 rollout 采样阶段保存下来的中间编码。

### 3.7 Deployment 层

训练后的推理链路通过 bundle 导出到：

- MuJoCo 部署适配器
- Real robot 部署适配器

部署时也会加载：

- `ActorCriticParkourMoE`
- `ParkourEstimator`
- `DepthAutoEncoder`

因此训练和部署之间没有额外的结构替换，基本保持同构。

## 4. 输入、观测与监督目标

## 4.1 Actor 当前观测 `obs`

在 `Go2Robot.compute_observations()` 中，actor 当前观测维度为 `45`：

```text
base_ang_vel                  3
projected_gravity             3
commands[:3]                  3
dof_pos error                12
dof_vel                      12
previous actions             12
-------------------------------
total                        45
```

这 45 维是 actor 在每一步真正可见的当前本体状态摘要。

## 4.2 Critic 特权观测 `privileged_obs`

普通 Go2 环境的 privileged observation 为 `263` 维：

```text
base_lin_vel                  3
base_ang_vel                  3
projected_gravity             3
commands[:3]                  3
dof_pos error                12
dof_vel                      12
previous actions             12
foot contact forces           4
torques                      12
motor accelerations          12
height measurements         187
-------------------------------
total                       263
```

Parkour 环境又在尾部追加了 `4` 维足端高度，因此最终 critic 输入维度变成 `267`：

```text
263 + feet_heights(4) = 267
```

这也正是 `GO2ParkourCfg.env.num_privileged_obs = 267` 的来源。

## 4.3 Estimator 的 proprio history

runner 里维护的历史长度默认为 `10`，每帧是当前 `obs` 的 45 维，所以：

```text
history_length = 10
obs_now_dim = 45
history_dim = 45 * 10 = 450
```

也就是说 estimator 吃进去的本体历史输入张量是：

```text
proprio_seq: [B, 450]
```

在 estimator 内部再 reshape 成：

```text
[B, 10, 45]
```

## 4.4 深度观测 `depth_camera`

Parkour 环境中相机采用 `proxy` 模式，当前默认维护两帧深度图缓存：

```text
depth_seq: [B, 2, 58, 87]
```

这个设计有两个特点：

1. 输入不是 RGB，而是两帧归一化深度图。
2. 输入不是直接物理相机渲染，而是根据地形高度场 raycast 合成的代理深度。

因此当前 estimator 的视觉分支学到的是一种“代理几何感知”，而不是传统意义上的纹理视觉。

## 4.5 Vision mask

估计器前向还有一个很关键的输入：

```text
mask_vision: [B]
```

它表示该环境当前是否允许使用视觉 token。

这个 mask 在不同阶段来源不同：

- 训练时：对 easy terrain 的一部分环境按 iteration 周期翻转 vision on/off
- 推理时：对 easy terrain 先用 depth autoencoder 的重建误差判断是否关闭视觉

这意味着模型不是单纯地“永远依赖视觉”，而是显式训练和部署了视觉失效/关闭分支。

当前视觉 token dropout 也不再假设噪声会等概率出现在任意位置。默认实现会在 token 网格上按空间先验采样块状 dropout，优先覆盖图像下部和远处区域，更接近真实深度相机中的地面反光、近场污染和远处过曝失真。

## 4.6 Estimator 的监督目标切片

runner 从 `privileged_obs` 中抽取三个 supervision target：

### 4.6.1 `v_t` 目标

```text
critic_obs[:, 0:3]
```

对应机身线速度 `base_lin_vel`，维度 `3`。

### 4.6.2 `m_t` 目标

地形高度图来自 privileged observation 倒数 `4 + 187` 之前的 `187` 维：

```text
mt_privileged_slice = slice(76, 263)
```

维度 `187`。

### 4.6.3 `h_t` 目标

最后 `4` 维是 feet heights：

```text
ht_privileged_slice = slice(263, 267)
```

维度 `4`。

也就是说 estimator 主要学习三种显式物理量：

1. 线速度
2. 足端高度
3. 周围地形高度图

除此之外，它还通过 `o_hat` 预测下一步 proprio observation。

## 5. Estimator 的结构设计

## 5.1 总体结构

`ParkourEstimator` 可以概括为：

```text
proprio history + depth history + vision flag
    -> token 化
    -> Transformer
    -> shared pooling
    -> expert pooling
    -> router
    -> Read/Write Experts
    -> shared_state updater
    -> prediction head
    -> {v_t, h_tf, z_mu, z_logvar, z_tm}
    -> {o_hat, m_hat}
    -> mcp_code
```

和传统“单一 MLP estimator”相比，它的关键变化不是简单增大网络，而是显式引入了：

- 多模态 token 编码
- 全局上下文 pooling
- 专家路由
- 共享记忆
- 多头辅助预测

## 5.2 `ParkourEstimatorCore` 的作用

`ParkourEstimatorCore` 负责把多模态原始输入转成统一 token 序列。

### 5.2.1 图像编码

深度图分支是 3 层卷积：

```text
2 x 58 x 87
 -> Conv(2,16,stride=2)
 -> Conv(16,32,stride=2)
 -> Conv(32,64,stride=2)
 -> 64 x 8 x 11
```

flatten 之后得到：

```text
88 个 image tokens，每个 token 维度 64
```

因为：

```text
8 * 11 = 88
```

### 5.2.2 Proprio 历史编码

每一帧 45 维 proprio 会经过一个 3 层 MLP 编到 64 维：

```text
45 -> 256 -> 128 -> 64
```

总共得到：

```text
10 个 proprio tokens，每个 token 维度 64
```

### 5.2.3 Vision flag token

视觉开关不是只喂给 router，而是先通过线性层编码成一个额外 token：

```text
1 -> 64
```

因此 estimator 输入给 transformer 的 token 总数是：

```text
88 image tokens
+10 proprio tokens
+ 1 vision token
=99 tokens
```

### 5.2.4 为什么这样设计

这一步的设计意图很清晰：

1. 让视觉和本体信息都以 token 方式进入同一个序列模型。
2. 让“视觉是否可用”本身也成为显式建模对象，而不是隐藏在数据分布里。
3. 让模型在后续 routing 时，既能感知几何场景，又能感知 proprio 时序，还能感知视觉开关状态。

## 5.3 位置编码与模态编码

core 里没有使用可学习位置编码，而是构造了：

- 1D sin/cos 编码
- 2D sin/cos 编码

分别用于：

1. image token 的二维位置
2. proprio token 的时间位置
3. image/proprio/vision_flag 的模态类型

这样做的优点是：

- 结构简单
- 不依赖额外可学习参数
- 对图像 patch、历史步序和模态类别都有显式区分

## 5.4 Vision masking 的实现方式

当前代码并不是把深度图置零后硬塞给 transformer，而是两步一起做：

1. `image_tokens` 乘上 `mask_vision`
2. `image_padding_mask` 把视觉 token 标记为 padding

因此当 `mask_vision=False` 时：

- 图像 token 内容会被清空
- attention 里也不会再把它们当作有效 token

这比单纯置零更干净，因为 transformer 不会把无效视觉位置当成“有意义的零值图像”。

## 5.5 Transformer 编码层

当前 estimator 使用的是一个标准 `TransformerEncoder`：

- `d_model = 64`
- `nhead = 4`
- `num_layers = 1`
- `ff_dim = 256`

层数不深，但由于 token 数量已经接近 100，因此它主要承担的是：

1. 跨模态融合
2. 跨时间步信息交互
3. 对视觉失效时的 token 结构重新加权

它不是那种特别重的视觉 backbone，更像一个“轻量多模态融合器”。

## 6. Attention Pooling、Router 与专家系统

## 6.1 为什么不用只取 transformer 最后一个 token

当前 estimator 没有用 `[CLS] token`，而是实现了一个通用的 `attention_pool()`。

做法是：

1. 用查询向量 `queries`
2. 对所有 encoded tokens 做 attention score
3. 加权求和得到 pooled feature

这样可以把“读哪些 token”本身做成可学习行为。

## 6.2 Shared pooling

共享 pooling query 只有一个：

```text
self.shared_pool_query: [1, 64]
```

经过 attention pooling 后得到：

```text
shared_pooled_feature: [B, 1, 64]
context_feature: [B, 64]
```

它可以理解为：

- 当前时刻从所有有效 token 中抽出的全局上下文摘要

这个 `context_feature` 是后续 router、shared memory 更新和最终预测头的公共输入。

## 6.3 Expert pooling

专家 pooling query 有 `expert_num=4` 个：

```text
self.expert_pool_queries: [4, 64]
```

经过 pooling 得到：

```text
expert_pooled_features: [B, 4, 64]
```

然后再与共享上下文相加：

```text
expert_context_features = expert_pooled_features + context_feature.unsqueeze(1)
```

这意味着每个专家都拿到：

1. 一份偏向自己关注区域的 token 摘要
2. 一份共享全局上下文

所以它不是 4 个完全互不通信的 expert，而是“共享全局语境 + 专家特异性关注”的结构。

## 6.4 Router 的输入与作用

router 输入由四部分拼接而成：

```text
obs_now           45
vision_flag        1
context_feature   64
shared_state_prev 64
---------------------
total            174
```

再经过：

```text
174 -> 128 -> 4 -> Softmax
```

输出：

```text
gating_weights: [B, 4]
```

这个设计表达了一个很明确的路由逻辑：

1. 看当前本体状态
2. 看当前是否允许视觉
3. 看多模态融合后的全局上下文
4. 看前一时刻共享记忆

然后决定 4 个 expert 当前各占多少权重。

## 6.5 `ReadWriteExpert` 的设计

每个 expert 不是独立维护一条完全私有的长期 hidden state，而是通过共享状态做读写。

单个 `ReadWriteExpert` 的结构是：

```text
shared_state
 -> read MLP
 -> hidden_init

context_feature
 + hidden_init
 -> GRUCell
 -> hidden_state

hidden_state
 -> write MLP
 -> shared_delta
```

即：

1. 从共享状态里“读”出自己的初始隐状态
2. 用自己的上下文特征做一步 GRU 更新
3. 再把结果“写回”为一个共享状态增量

它体现的是一种“共享记忆空间 + 专家特定读写映射”的设计，而不是简单的 `4 个 GRU hidden 相加`。

## 6.6 Shared state 更新

所有 expert 写回的 `shared_delta` 会按 gate 权重加权求和：

```text
mixed_delta = sum_i gate_i * delta_i
```

然后与共享上下文拼接，喂给另一个 `GRUCell` 更新共享记忆：

```text
[context_feature, mixed_delta] + shared_state_prev
    -> shared_state_updater
    -> shared_state
```

最后再做一层 `LayerNorm`。

这一步很关键，因为它把“专家输出”重新整合进了一个统一、可跨时刻传播的公共状态。

## 6.7 这个专家设计想解决什么问题

它主要想解决两个问题：

1. 不同专家之间切换时，纯私有 hidden state 容易不连续。
2. 视觉有效和视觉失效之间切换时，需要一个共享的动态上下文桥梁。

因此当前设计不是把 expert 当成四个互相隔离的技能库，而是把它们当成：

- 针对不同场景的状态更新器

而共享记忆 `shared_state` 则承担跨专家切换时的时序连续性。

## 7. 预测头、隐变量与 `mcp_code`

## 7.1 Predictor head

共享状态更新后，会和 `context_feature` 再拼接一次，通过 `prediction_head` 得到最终预测特征：

```text
[shared_state, context_feature] -> prediction_head -> predictor_input
```

这个 `predictor_input` 维度仍然是 `64`，是所有下游头的公共输入。

## 7.2 显式预测头

从 `predictor_input` 直接预测：

- `v_t`: 3 维，预测机身线速度
- `h_tf`: 4 维，预测足端高度
- `z_mu`: 16 维，latent 均值
- `z_logvar`: 16 维，latent 对数方差
- `z_tm`: 32 维，terrain latent

这些头体现了 estimator 的双重身份：

1. 它是控制中间编码器
2. 它也是一个带自监督/辅助监督的状态估计器

## 7.3 变分 latent

`z_mu` 和 `z_logvar` 会通过 reparameterization 得到 `z_t`：

- 训练时：采样
- 测试时：直接用 `mu`

所以这里引入了一个轻量 VAE 风格的 latent 建模。

它的目的不是做高保真生成，而是让中间表示带有一定的分布约束，避免 latent 完全无约束地漂移。

## 7.4 解码器

当前 estimator 有两个解码器。

### 7.4.1 `decoder_obs`

输入：

```text
[v_t, h_tf, z_t] = 3 + 4 + 16 = 23
```

输出：

```text
o_hat: 45
```

它要重构的是下一步 proprio observation。

### 7.4.2 `decoder_map`

输入：

```text
z_tm: 32
```

输出：

```text
m_hat: 187
```

它要重构的是地形高度图。

## 7.5 `mcp_code` 的组成

真正送给 actor 的中间控制表征是：

```text
mcp_code = cat(v_t, h_tf, z_mu, z_tm)
```

维度为：

```text
3 + 4 + 16 + 32 = 55
```

也就是说 actor 最终使用的是：

1. 估计的线速度
2. 估计的足端高度
3. VAE latent 的均值
4. 地形 latent

而不是直接使用 `z_logvar`、`o_hat` 或 `m_hat`。

这说明 `mcp_code` 是一个面向控制的紧凑摘要，而不是 estimator 全部输出的简单拼接。

## 8. Actor-Critic 的结构与作用分工

## 8.1 Actor 输入

actor 输入由两部分构成：

```text
mcp_code   55
obs_now    45
---------------
total     100
```

所以 actor 不是直接看深度图，也不是直接看 10 步历史，而是看：

- 当前可部署本体观测
- estimator 压缩后的高层地形/状态表征

这符合“感知前端负责压缩，policy 后端负责决策”的分工。

## 8.2 Actor 结构

默认 actor MLP 结构为：

```text
100 -> 512 -> 256 -> 128 -> 12
```

最后输出动作均值 `action_mean`，并与可学习标准差 `std` 组成高斯分布：

```text
Normal(mean, std)
```

训练时采样动作，推理时直接走均值输出。

## 8.3 Critic 输入

critic 直接吃 `267` 维 privileged observation。

默认结构为：

```text
267 -> 512 -> 256 -> 128 -> 1
```

它不额外依赖 `mcp_code`。

这意味着当前 critic 的价值估计完全建立在特权观测上，而不是 actor 的压缩中间状态上。

## 8.4 为什么这是合理的

这是非常典型的不对称训练策略：

1. actor 只学部署时能拿到的信息
2. critic 利用训练期特权信息提高价值估计质量

对于复杂地形 locomotion，这通常比 actor/critic 共用完全相同输入更稳定。

## 9. Depth AutoEncoder 的作用

## 9.1 结构

`DepthAutoEncoder` 的视觉 backbone 与 estimator 的图像编码分支很相似：

```text
2 x 58 x 87
 -> Conv -> Conv -> Conv
 -> 64 x 8 x 11
 -> FC latent(128)
 -> FC
 -> Deconv -> Deconv -> Deconv
 -> Bilinear resize
 -> recon depth
```

## 9.2 训练阶段作用

训练时它每次 estimator 刷新后都会执行一轮重建：

```text
depth_camera -> depth_autoencoder -> depth_recon
```

并最小化：

```text
MSE(depth_recon, depth_camera)
```

这是一个纯自监督视觉重建任务。

## 9.3 推理阶段作用

推理时它的角色更像 selector。

当前逻辑是：

1. 对 easy terrain 环境先重建当前深度图
2. 计算 per-env 重建误差
3. 若误差超过 `selector_threshold`，则把该环境的 `mask_vision=False`

因此它实际上承担的是：

- 视觉可靠性门控器

这一步尤其适合 simple terrain 上的 blind-walk/fallback 逻辑。

## 9.4 为什么 estimator 本身不直接做 selector

因为 depth autoencoder 的重建误差天然能反映“当前视觉输入是否落在训练期已知分布附近”。

它和 estimator 主干分开有两个好处：

1. 不干扰控制主链路
2. 可以把视觉异常检测和控制编码解耦

## 10. Runner 的训练组织方式

## 10.1 初始化

`OnPolicyRunnerParkourMoE` 会初始化：

- actor-critic
- `DefaultEstimator`
- `PPOParkourMoE`
- history buffer
- vision mask
- SwAV 窗口缓存

其中需要特别指出的是：

- `DefaultEstimator` 在 runner 里被实例化并传入算法构造函数
- 但当前 `PPOParkourMoE` 内部直接重新创建了 `ParkourEstimator`
- 因此这个 `DefaultEstimator` 当前并未进入主训练路径

这是当前代码中的一个“保留但未生效”部分。

## 10.2 History buffer

runner 自己维护：

```text
history: [num_envs, 10, 45]
```

每步环境交互后会：

1. 对 done 环境清零
2. 丢弃最老一帧
3. 追加当前 obs

它不是存到 estimator 里，而是 runner 显式管理。

## 10.3 Vision on/off 训练策略

runner 会先根据 terrain 名称构建 easy terrain 环境索引。

然后在训练主循环中，每隔：

```text
vision_toggle_interval = 20
```

对这些 easy terrain 环境执行一次：

```text
self.mask[easy_envs] = ~self.mask[easy_envs]
```

这等价于周期性制造：

- vision on 样本
- vision off 样本

从而逼迫 estimator 和 actor 学会在简单地形上既能用视觉，也能退化到非视觉策略。

## 10.4 为什么 estimator 不每步刷新

runner 用：

```text
refresh_estimator =
    common_step_counter % camera_update_interval == 0
    or mcp_code is None
```

来控制 estimator 刷新。

默认 `camera_update_interval = 5`，也就是：

- 深度图和 estimator 编码不是每个控制步都更新
- 中间若不刷新，会复用上一次的 `mcp_code`

这种做法的意义是：

1. 和相机更新频率保持一致
2. 降低 estimator 计算成本
3. 更贴近真实部署中低频视觉、高频控制的结构

## 11. 单步训练数据流

下面按一轮 rollout 中的一个环境 step 来说明。

## 11.1 Step 开始前

runner 已经持有：

- 当前 `obs`
- 当前 `privileged_obs`
- 当前 `history`
- 当前 `depth_camera`
- 当前 `mask`

## 11.2 若本步需要刷新 estimator

会执行：

```text
obs_now = obs
proprio_hist = history.flatten(1)
gt_vt_step = critic_obs[vt_slice]
gt_ht_step = critic_obs[ht_slice]
gt_mt_step = critic_obs[mt_slice]
camera_depth = additional_obs["depth_camera"]
```

然后前向：

```text
est_out = estimator(proprio_hist, camera_depth, mask, obs_now=obs_now)
```

同时深度自编码器做一次重建：

```text
depth_recon = depth_autoencoder(camera_depth)
```

并立刻更新 depth autoencoder 参数。

## 11.3 动作生成

actor 使用当前 `mcp_code` 和 `obs` 生成动作：

```text
actions = actor_critic.act(mcp_code, obs)
```

而 critic 同时基于 `critic_obs` 评估 value。

这些信息会被打包到 transition 里，稍后写入 rollout storage。

## 11.4 环境推进

执行：

```text
obs, privileged_obs, rewards, dones, infos = env.step(actions)
```

随后更新：

- `history`
- `swav` 历史窗口
- estimator hidden state reset
- rollout buffer

## 11.5 若本步刷新过 estimator

则还会基于这一步拿到的 next obs 做 estimator 参数更新：

```text
gt_next_step = obs
update_estimator(est_out, gt_ht, gt_mt, gt_vt, gt_next_step, ...)
```

注意这里 estimator 的 supervision 既包含当前时刻显式目标：

- `v_t`
- `h_tf`
- `m_hat`

也包含一步前向后的 next proprio reconstruction：

- `o_hat`

## 12. 一轮迭代的数据流

把整个 `learn()` 展开，可以概括为：

```text
reset env
 -> 初始化 obs / privileged_obs / history

for each learning iteration:
    如有需要切换 easy terrain 的 vision mask

    for step in num_steps_per_env:
        如有需要刷新 estimator
        actor 采样动作
        env.step(actions)
        更新 history / swav window / estimator / depth AE
        把 obs, critic_obs, mcp_code, action, value 等写入 rollout storage

    critic 计算最后一步 bootstrap value
    PPO update actor_critic
    记录 estimator 与 PPO 指标
    定期保存 checkpoint
```

这里要注意：

1. estimator 更新发生在 rollout 采样阶段。
2. actor-critic 的 PPO 更新发生在 rollout 结束之后。
3. 两者虽然在同一 iteration 中交替出现，但优化器是分开的。

## 13. Loss 设计

## 13.1 PPO 主损失

actor-critic 使用标准 PPO 损失：

1. surrogate loss
2. value loss
3. entropy regularization
4. adaptive KL 调节学习率

这部分没有使用 estimator 的辅助 loss，只依赖 rollout buffer 中保存的：

- obs
- critic_obs
- mcp_code
- actions
- old log prob
- old mean/std
- returns
- advantages

## 13.2 Estimator 损失

`update_estimator()` 中的 estimator 总损失由以下几部分组成：

### 13.2.1 `vt_loss`

```text
MSE(v_t, gt_vt)
```

监督机身线速度。

### 13.2.2 `ht_loss`

```text
MSE(h_tf, gt_ht)
```

监督足端高度。

### 13.2.3 `mt_loss`

```text
MSE(m_hat, gt_mt)
```

监督地形高度图。

### 13.2.4 `reconstruction_loss`

```text
MSE(o_hat, gt_next_obs)
```

监督下一步 proprio observation。

### 13.2.5 `z_kl_loss`

```text
KL(q(z|x) || N(0, I))
```

并带有 warmup 权重。

### 13.2.6 `load_balance_loss`

对 gate 平均使用率做均衡约束：

```text
mean_usage -> 尽量接近 1 / expert_num
```

这样可以减少 router 长期塌缩到单一 expert。

### 13.2.7 `terrain_swav_loss`

这是当前实现里比较有特点的一项。

它把：

- gate 权重时间窗
- 地形图时间窗

分别投影到同一个 prototype 空间，并用 Sinkhorn 生成分配，再做双向匹配约束。

它的直觉是：

- 如果某段时间里 gate 使用模式相似，那么对应的地形窗口也应该相似
- 反过来也成立

因此这是一个“专家路由模式”和“地形结构模式”之间的跨视角对齐约束。

## 13.3 Depth AutoEncoder 损失

depth autoencoder 只用：

```text
image_recon_loss = MSE(depth_recon, depth_camera)
```

单独更新，不和 estimator 共用 optimizer。

## 13.4 为什么分三个 optimizer

当前设计将：

- actor-critic
- estimator
- depth autoencoder

分开优化，是合理的，因为三者的目标并不完全一致：

1. actor-critic 关注长期回报
2. estimator 关注中间状态建模和结构化监督
3. depth AE 关注视觉重建与异常检测能力

分开可以减少梯度相互污染。

## 14. SwAV 窗口机制

## 14.1 Window 的组成

runner 为每个 env 维护三个滑动窗口：

1. `swav_gate_hist`
2. `swav_map_hist`
3. `swav_valid_hist`

长度默认为：

```text
terrain_window_length = 10
```

## 14.2 每次 estimator 刷新时做什么

会把当前：

- `gating_weights`
- `gt_mt_step`
- `valid_mask`

拼到窗口末尾，并把旧值向前滚动。

于是 estimator update 时可以拿到一个时间窗，而不是单帧。

## 14.3 设计意义

单帧 gate 权重和单帧地形图之间的对应关系可能很弱。

但一个时间窗口里的：

- 专家使用模式
- 地形变化模式

更容易具有一致的统计结构。

SwAV 正是在这个层面上对齐两种模态。

## 15. 推理与部署路径

## 15.1 `get_inference_policy()` 的推理逻辑

runner 的推理 policy 做的事情是：

1. 根据 `reset_buf` 重置 estimator hidden state 和 SwAV 历史
2. 更新 proprio history
3. 到 estimator 刷新时再取当前深度图
4. 若是 easy terrain，则先跑 depth autoencoder selector
5. 根据 selector 结果决定 `mask_vision`
6. estimator 输出 `latest_mcp_code`
7. actor 用 `latest_mcp_code + obs` 直接输出动作

也就是说，控制频率可以高于视觉刷新频率，而 policy 在两次视觉更新之间会复用上一个 `mcp_code`。

## 15.2 训练与部署的一致性

导出的 bundle 里包含：

- actor_critic state dict
- estimator state dict
- depth_autoencoder state dict
- policy_cfg
- estimator_cfg
- depth_encoder_cfg

MuJoCo 和 real robot 适配器都会直接重建相同结构后加载参数。

因此部署链路和训练链路在网络拓扑上高度一致，差别主要在：

- 深度输入的来源
- 机器人状态观测的来源

## 15.3 推理态的状态缓存

在部署适配器里也都显式维护：

- `history`
- `latest_mcp_code`
- `latest_m_hat`
- `latest_gating_weights`

这再次说明这套框架本质上是“低频估计器 + 高频 actor”的两级系统，而不是一个每步都从零开始的无状态 policy。

## 16. 当前实现里的设计亮点

可以把当前框架最有特色的设计总结成 6 点。

### 16.1 感知与控制显式解耦

视觉和历史感知先进入 estimator，被压成 `mcp_code` 后再交给 actor。

这让 actor 的输入规模和复杂度被有效控制住了。

### 16.2 多模态 token 化而不是直接拼接

深度图、本体历史和 vision flag 先转成 token 再融合，比直接 flatten 拼接更适合表达不同模态间的结构关系。

### 16.3 共享记忆 + 专家读写

这比“完全独立的多 expert hidden state”更适合 vision on/off 切换和地形模式切换。

### 16.4 显式 fallback 训练

easy terrain 上周期性关视觉，以及推理时基于 AE 的 selector，都是为了让模型具备“能用视觉时用视觉，不能用时也能走”的能力。

### 16.5 辅助目标足够贴近 locomotion

监督项不是抽象分类标签，而是：

- 线速度
- 足端高度
- 地形高度图
- 下一步 proprio

这些都和运动控制强相关。

### 16.6 训练与部署链路一致

同一套 estimator、actor、selector 可以直接导出到 MuJoCo 和真机侧，减少了训练部署鸿沟。

## 17. 当前代码中已经预留但尚未真正使用的部分

结合当前源码，下面这些项值得单独说明。

### 17.1 `DefaultEstimator`

已经实例化，但当前主路径没有使用。

### 17.2 `shared_state_alpha`

在 config 和 estimator 构造函数中保留，但当前更新 shared state 时实际上走的是 `GRUCell`，没有再使用固定 alpha blend。

### 17.3 `gate_smooth_coef`

在 `PPOParkourMoE` 构造函数里有参数，但当前损失中没有实际使用。

### 17.4 `cycle_consistency_coef`

已保留参数，但当前没有进入损失。

### 17.5 `vision_consistency_coef`

已保留参数，但当前没有进入损失。

### 17.6 `gt_mt_step`

传入 estimator forward，但当前 forward 内部并未实际使用。

### 17.7 `mcp_code_additional`

已输出，但当前 actor 和 PPO 主链路没有使用。

### 17.8 `fc_zt`、`map_unet`、`fc_zm_fine`

这些层已经定义，但当前 forward 路径没有调用。

这部分不代表设计错误，更像是当前版本还处在持续迭代中的痕迹。写文档时把它们区分出来，能避免后续阅读代码时误判“所有定义的模块都已经在生效”。

## 18. 最终总结构图

```text
Environment
  -> obs_now [45]
  -> privileged_obs [267]
  -> depth_buffer [2,58,87]
  -> terrain / dones / infos

Runner
  -> history buffer [10,45]
  -> vision mask
  -> swav windows

Estimator
  proprio history [10,45]
  + depth buffer [2,58,87]
  + vision flag
    -> tokenization
    -> transformer
    -> shared pooling / expert pooling
    -> router
    -> 4 read-write experts
    -> shared_state update
    -> predictor heads
    -> mcp_code [55]

Actor
  mcp_code [55] + obs_now [45]
    -> action [12]

Critic
  privileged_obs [267]
    -> value [1]

Training losses
  PPO loss
  + vt/ht/mt reconstruction losses
  + next obs reconstruction
  + latent KL
  + load balance
  + terrain SwAV
  + depth image reconstruction

Inference fallback
  depth autoencoder recon error
    -> selector threshold
    -> mask_vision on/off
```

## 19. 结论

当前这套 Parkour MoE `estimator-actor-critic` 框架，本质上不是一个“普通 PPO 再加一点视觉输入”的结构，而是一套分层感知控制系统：

1. 环境提供当前本体观测、特权观测、代理深度图和地形信息。
2. estimator 通过多模态 token、transformer、共享记忆和多专家路由，把历史 proprio 与深度感知压缩成 `mcp_code`。
3. actor 只接收 `mcp_code + 当前 obs`，输出动作。
4. critic 独立使用特权观测做价值评估。
5. depth autoencoder 既做自监督重建，也在推理时承担视觉可靠性 selector。
6. runner 通过 vision toggle 和 easy terrain fallback，把“视觉有效”和“视觉失效”都纳入了训练闭环。

从设计取向上看，这个框架最核心的目标并不是单纯追求更大的模型，而是追求三件事：

1. 多地形 parkour 场景下的中间状态建模能力
2. 视觉存在与缺失之间的平滑切换能力
3. 训练期到部署期结构一致的可落地性

如果后续继续迭代，这份文档里第 17 节列出的“预留未使用项”会是最值得优先清理或激活的部分，因为它们最直接决定这套框架之后是继续向“更完整的结构化 estimator”演进，还是向“更精简、可部署的感知控制骨架”收敛。
