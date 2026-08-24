# ParkourMoE 改造分析与设计建议

## 1. 文档范围

本文档基于当前仓库中的 `ParkourMoE` 实现，对以下三部分内容进行整理：

1. 当前 `ParkourMoE` 架构与任务目标的匹配关系。
2. 用户提出的“公共状态 `z_t` + 专家读写映射 `W_i / W_i'`”机制的合理性分析。
3. 结合当前任务目标后，更推荐的落地设计方案与训练约束。

本文档讨论的当前实现主要对应：

- `rsl_rl/rsl_rl/modules/actor_critic_parkour_moe.py`
- `rsl_rl/rsl_rl/algorithms/ppo_parkour_moe.py`
- `rsl_rl/rsl_rl/runners/onpolicyrunner_parkour_moe.py`

## 2. 任务目标

当前任务要求机器人具备以下能力：

1. 具有前向深度视觉。
2. 具有爬箱子、跳 gap、上下楼梯、上下斜坡、粗糙地面行走、摔倒恢复能力。
3. 在简单地形上具有全向运动能力。
4. 在视觉失效时仍能保持基础运动能力。

对应地，当前设计希望达到：

1. MoE 结构能够利用视觉信息应对多类运动场景。
2. GRU 多专家能够编码运动序列。
3. MoE 结构能够依据视觉是否失效切换专家。

## 3. 当前 ParkourMoE 架构概述

### 3.1 当前主链路

当前 `ParkourEstimator` 的结构可以概括为：

```text
proprio history + depth history + vision flag
  -> ParkourEstimatorCore
  -> Transformer
  -> shared_head
  -> router(obs_now, vision_flag, gru_input)
  -> 4 个独立 MemoryExpert(GRU)
  -> soft mixture
  -> h_t
  -> 多个预测头
  -> mcp_code
  -> actor
```

其中：

- `ParkourEstimatorCore` 负责视觉和本体 token 融合。
- `Transformer` 负责时空融合后的编码。
- `router` 输出 `expert_num` 维 softmax 权重。
- `experts` 是 `ModuleList([MemoryExpert(...)])`，每个 expert 持有独立的 GRU hidden state。
- 最终 `mcp_code = cat(v_t, h_tf, z_mu, z_tm)` 供 actor 使用。

### 3.2 当前结构的优点

当前方案已经具备以下优势：

1. 视觉和本体是多模态融合的，不是简单拼接。
2. expert 是显式 GRU memory，具备时序建模能力。
3. `router` 显式看到了 `vision_flag`，理论上可用于 vision on/off 模式切换。
4. 估计器输出同时覆盖速度、足端高度、地形图、下一步本体重建等多种辅助目标。

### 3.3 当前结构的核心问题

当前 expert 机制更接近：

```text
h_i^t = GRU_i(x_t, h_i^{t-1})
g_t = router(x_t)
z_t = Σ_i g_i^t * h_i^t
```

其中每个 expert 都维护各自独立的 hidden state。

这会带来两个重要问题：

1. 专家之间的 hidden space 没有显式对齐。
   - 如果某些时刻 router 权重显著变化，最终混合表示可能发生跳变。

2. “视觉有效 expert” 到 “视觉失效 fallback expert” 的切换不一定平滑。
   - 因为 fallback expert 并不是从当前活跃 expert 的状态连续接过去，而是在自己的私有 hidden state 轨迹上继续演化。

需要强调的是，当前实现是 soft mixture，不是硬 Top-1 选择。
因此问题不是“某一步只跑一个 expert，其他 expert 停止更新”，而是：

- 4 个 expert 都会前向；
- 4 个 hidden 都各自持续演化；
- 但最终提供给 actor 的融合表示可能由于 gate 变化而发生不连续。

## 4. 用户提出的改动

### 4.1 基础版：公共状态 `z_t`

用户首先提出：

```text
1. 输入 x_t
2. 共享前一个时刻的公共状态 z_{t-1}
3. 4 个 GRU 分别计算:
   h_i^t = GRU_i(x_t, z_{t-1})
4. 每个专家独立映射:
   f_i^t = W_i h_i^t
5. 门控 g_t
6. 融合:
   z_t = Σ_i g_i^t f_i^t
7. z_t 作为下一时刻公共状态
```

这个版本的核心思想是：

- 不再让 4 个 expert 各自维护完全独立的长期记忆；
- 而是通过一个共享公共状态 `z_t`，把跨专家切换的时序连续性建立起来。

### 4.2 增强版：专家读写映射 `W_i / W_i'`

随后又进一步提出：

```text
t 时刻激活 expert i:
  h_i^t = GRU_i(x_t, h_{i,t-1})
  z_t = write_i(h_i^t)

t+1 时刻激活 expert j:
  h_{j,t-1} = read_j(z_t)
  h_j^t = GRU_j(x_{t+1}, h_{j,t-1})
  z_{t+1} = write_j(h_j^t)
```

其本质是：

- `z_t` 作为公共记忆空间；
- 每个 expert 有自己的内部状态空间 `h_i`；
- 通过 `read_i(z)` 和 `write_i(h_i)` 完成公共空间和专家空间之间的通信。

这个版本比“所有 expert 直接共享同一个 hidden 几何”更合理，因为它允许：

- 专家内部保留自己的状态表达方式；
- 同时仍然具备跨专家切换时的时序连续性。

## 5. 结合任务目标的合理性分析

## 5.1 对多类视觉场景的适应性

你的任务包含楼梯、坡面、gap、箱体、粗糙地形和 recovery。

这些任务有一个共同特点：

- 场景模式可能切换；
- 但机器人本身的动力学状态是连续的。

因此，从任务层面看，“通过公共状态承载连续运动上下文，再由 expert 学习不同状态更新策略”是合理的。

它比“4 条完全独立的长期私有记忆”更符合任务物理结构，因为：

- 场景可以变；
- 技能模式可以切；
- 但机器人不能在动力学状态上突然失忆。

## 5.2 对 GRU 多专家编码运动序列的影响

如果采用公共状态方案，expert 的角色会发生变化：

- 当前方案中，expert 更像“各自维护一条私有时序轨迹的记忆器”。
- 改造后，expert 更像“在共享记忆基础上的不同状态更新器”。

这意味着：

- 时序连续性会更强；
- 但每个 expert 保留独立长期记忆的能力会变弱。

对于你的任务，这种变化总体上是可接受的，但要注意：

- 跳 gap 和 recovery 这类强阶段性技能，往往需要较强的时序内部结构；
- 如果公共状态维度过小，可能会把这些关键信息压坏。

因此，公共状态机制是合理的，但不宜把它做成过窄的瓶颈。

## 5.3 对简单地形全向运动的影响

简单地形全向运动更依赖：

- 低延迟；
- 平滑性；
- 对视觉的低依赖。

从这个角度看，公共状态方案是加分项，因为它可以降低专家切换带来的融合表示抖动。

特别是在 vision on/off 切换时，如果没有公共状态桥接，基础 locomotion 可能会因为 expert 语义空间不一致而出现突变。

## 5.4 对视觉失效 fallback 能力的影响

这一点是该改动最合理的地方。

你的目标明确要求：

- 视觉失效时仍保持基础运动能力。

而当前 router 已经显式使用 `vision_flag`，说明你本来就希望它参与模式切换。

在这种设定下，如果：

- 视觉可用时主要使用“视觉主导 expert”；
- 视觉失效时切到“proprio fallback expert”；

那么跨专家时序连续性就变得非常重要。

因此，使用公共状态 `z_t` 加 `read_i / write_i` 机制来支持 vision on/off 切换，是和任务目标强一致的。

## 6. 该改动的优缺点

### 6.1 优点

1. 减少专家切换时的表示跳变。
2. 更适合 vision on/off 模式切换。
3. 更符合“动力学连续、模式可变”的运动任务结构。
4. 让 expert 更像“技能更新器”而不是“互不通信的记忆孤岛”。

### 6.2 风险

1. 4 个 expert 不再拥有完全独立的长期私有记忆。
2. 如果公共空间过窄，跳 gap / recovery 等强阶段性技能信息可能会丢失。
3. 读写映射 `read_i / write_i` 本身会增加训练难度。
4. 如果没有额外约束，expert 更容易收敛到相似更新器。

### 6.3 一个关键判断

如果你的目标是：

- 让 expert 像“4 个并行长期记忆库”一样工作，

那么公共状态方案会削弱这一点。

如果你的目标是：

- 让 expert 能在多类场景和 vision on/off 之间平滑切换，

那么公共状态方案是更合理的。

结合当前任务目标，第二种更重要，因此总体上支持改动。

## 7. 推荐的最终落地方案

不建议直接采用“纯公共状态替代一切”的极端版本。

更推荐采用一个混合式结构：

### 7.1 高层结构

```text
depth history + proprio history + vision flag
  -> multimodal core
  -> transformer
  -> context feature c_t

公共状态:
  z_{t-1}

router input:
  [obs_now, vision_flag, c_t, z_{t-1}]
  -> router
  -> gate g_t

for each expert i:
  h_i^{init} = read_i(z_{t-1})
  h_i^t = GRU_i(c_t, h_i^{init})
  delta_i^t = write_i(h_i^t)

公共状态更新:
  z_t = LN((1 - alpha) * z_{t-1} + alpha * Σ_i g_i^t * delta_i^t)

prediction heads:
  z_t or [z_t, c_t]
  -> v_t, h_tf, z_mu, z_tm, ...
  -> mcp_code
```

### 7.2 为什么推荐残差更新

不建议直接使用：

```text
z_t = Σ_i g_i^t * delta_i^t
```

更推荐：

```text
z_t = LN((1 - alpha) * z_{t-1} + alpha * Σ_i g_i^t * delta_i^t)
```

原因是：

1. 能保留上一时刻记忆，避免完全覆盖。
2. 更稳。
3. 更适合基础 locomotion 的连续性。

### 7.3 为什么不建议把 `W_i'` 称为“逆映射”

在实现上，更建议显式定义：

- `read_i: z -> h_i_init`
- `write_i: h_i -> delta_i`

而不是叫“逆映射”。

因为在训练过程中它们通常并不满足严格数学逆关系。
更准确地说，它们是：

- 专家读适配器；
- 专家写适配器。

### 7.4 更推荐的 expert 语义分化

如果保持 4 个 expert，建议在训练目标和 curriculum 上引导出如下偏向：

1. 基础 locomotion / vision-off fallback expert
2. 楼梯 / 斜坡 / 粗糙地面 expert
3. 箱体 / gap / 强视觉障碍 expert
4. recovery / 大姿态修正 expert

不需要硬编码标签，但建议：

- curriculum
- gate regularization
- vision dropout

都朝这个方向施加归纳偏置。

## 8. 训练与损失建议

当前 `ParkourMoE` 的一个关键短板是：

- 结构比 `MoECTS` 强；
- 但对 gate / expert 的直接约束不够强。

因此，在引入公共状态方案时，建议同步加入以下训练项。

### 8.1 显式 load balance loss

参考 `MoECTS` 中的思路：

```text
mean_usage = mean(gating_weights, dim=0)
target_usage = uniform
L_lb = ||mean_usage - target_usage||^2
```

它的作用是：

- 避免某一个 expert 长期独占；
- 提高 expert 真正分工的机会。

### 8.2 Gate 时间平滑约束

如果你的目标是降低专家切换抖动，建议显式加入：

```text
L_gate_smooth = mean(||g_t - g_{t-1}||^2)
```

或者使用带滞回的 routing 规则。

### 8.3 Vision dropout consistency

对简单地形或低风险场景，可加入：

- 视觉可用；
- 视觉屏蔽；

两种条件下 latent / mcp_code 不应大幅漂移的约束。

这会直接增强：

- vision-failure fallback 能力；
- 基础 locomotion 稳定性。

### 8.4 公共状态循环一致性

如果用了 `read_i / write_i`，建议加入轻量一致性约束：

```text
L_cycle = ||z - write_i(read_i(z))||^2
```

它能帮助公共空间和专家空间更稳定对齐。

### 8.5 可选：teacher latent 蒸馏

如果后续训练仍不稳定，最值得借鉴 `MoECTS` 的地方是：

- 加一个 privileged teacher encoder；
- 用 teacher latent 监督公共状态或 student latent。

这是最强、最直接的表征塑形信号之一。

### 8.6 不建议同时强推“高熵”和“高稀疏”

如果 gate 设计中同时写：

- 熵正则；
- 稀疏路由；

需要注意两者方向相反。

更建议使用：

1. 前期 soft routing；
2. temperature 或 top-k 渐进收缩；
3. 再辅以 load balance；

而不是简单把“高熵”和“高稀疏”一起拉满。

## 9. 工程实现建议

### 9.1 使用 `GRUCell` 或显式 hidden 输入输出

当前 `MemoryExpert` 把 hidden state 保存在模块内部。

如果改成公共状态读写机制，更推荐用显式接口：

```text
h_i^t = GRUCell_i(c_t, read_i(z_{t-1}))
```

而不是继续依赖每个 expert 内部保存独立 persistent hidden。

这样更符合新机制的设计语义，也更便于导出和调试。

### 9.2 公共状态建议单独命名

当前网络里已经有：

- `z_mu`
- `z_logvar`
- `z_tm`

因此新增公共状态时，不建议继续命名为 `z_t` 出现在代码变量名里。

更推荐使用：

- `memory_state`
- `shared_state`
- `state_t`

文档中可继续沿用 `z_t`，但代码中建议避免歧义。

### 9.3 更新频率建议分离

当前 estimator 刷新频率与 camera update 对齐。

如果未来公共状态承担更强的运动记忆功能，建议考虑：

- 视觉上下文低频刷新；
- 本体和公共状态高频更新。

否则 GRU memory 的潜力会被低刷新频率限制。

## 10. 最终结论

结合当前任务目标，用户提出的改动方向总体上是合理的，而且是有针对性的：

- 它直接瞄准了多场景切换与 vision on/off 切换时的时序连续性问题；
- 它比当前“4 个独立 hidden 并行混合”的结构更适合承担统一运动上下文；
- 它尤其适合你的“视觉失效仍保持基础运动能力”目标。

但它也不是没有代价：

- 它会削弱 expert 作为独立长期私有记忆体的属性；
- 如果设计过于极端，可能伤害 jump gap、recovery 等强阶段技能的表达能力。

因此，最推荐的不是“纯公共状态替代方案”，而是：

> 保留多模态前端，使用公共状态承担跨专家连续性，用 `read_i / write_i` 连接公共空间和专家空间，再配合 `load_balance`、`gate_smoothness`、`vision_dropout_consistency` 等训练约束。

这个混合版本最符合当前任务目标，也最有希望同时满足：

1. 多类视觉场景下的技能切换；
2. GRU 对运动序列的建模；
3. 视觉失效时的平滑 fallback；
4. 简单地形上的稳定全向运动。

