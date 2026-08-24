# `actor_critic_moe_cts.py` 网络结构说明

## 1. 文档范围

本文档只说明当前源码中 `rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py` 的网络设计结构，以及它直接依赖的：

- `rsl_rl/rsl_rl/modules/utils.py`
- `rsl_rl/rsl_rl/algorithms/moe_cts.py`

重点回答 4 个问题：

1. 这个文件里一共定义了哪些网络模块。
2. teacher 分支和 student 分支分别怎么走。
3. MoE student encoder 的内部结构是什么。
4. actor、critic、teacher encoder、student encoder 之间的梯度关系是什么。

本文档关注的是“当前实际代码”，不是论文中的一般形式，也不是历史版本。

## 2. 文件定位

`ActorCriticMoECTS` 是一个 **Concurrent Teacher-Student + MoE student encoder** 的 actor-critic 模块。

它的核心思想是：

- **teacher 分支**：用 `privileged_obs` 经过 `teacher_encoder` 得到 latent。
- **student 分支**：用历史观测 `history` 经过 `student_moe_encoder` 得到 latent。
- **共享 actor**：不管 latent 来自 teacher 还是 student，最后都和 `obs` 拼接，喂给同一个 actor。
- **共享 critic**：不管 latent 来自 teacher 还是 student，最后都和 `privileged_obs` 拼接，喂给同一个 critic。

也就是说，这个模块不是“两套 policy”，而是：

- 两个 encoder
- 一个共享 actor
- 一个共享 critic

## 3. 总体结构

整体张量流可以概括为：

```text
teacher path:
privileged_obs
  -> teacher_encoder
  -> latent_t

student path:
history
  -> student_moe_encoder
  -> latent_s

actor path:
latent + obs
  -> actor
  -> action_mean
  -> Normal(action_mean, std)

critic path:
detach(latent) + privileged_obs
  -> critic
  -> value
```

其中 student encoder 不是普通 MLP，而是一个 MoE：

```text
history
  -> gating network -> expert weights
  -> experts -> per-expert latent
  -> weighted sum
  -> normalization
  -> student latent
```

## 4. 输入输出维度

在构造函数中，代码先定义了 4 个关键输入维度：

```python
mlp_input_dim_t = num_critic_obs
mlp_input_dim_s = num_obs * history_length
mlp_input_dim_a = latent_dim + num_obs
mlp_input_dim_c = latent_dim + num_critic_obs
```

因此：

- teacher encoder 输入维度：`num_critic_obs`
- student encoder 输入维度：`num_obs * history_length`
- actor 输入维度：`latent_dim + num_obs`
- critic 输入维度：`latent_dim + num_critic_obs`

如果使用当前 `LeggedRobotCfgMoECTS` 默认配置，则关键超参数通常是：

- `history_length = 5`
- `latent_dim = 32`
- `expert_num = 8`
- `teacher_encoder_hidden_dims = [512, 256]`
- `student_encoder_hidden_dims = [512, 256, 256]`
- `actor_hidden_dims = [512, 256, 128]`
- `critic_hidden_dims = [512, 256, 128]`

## 5. 模块拆解

### 5.1 历史缓存 `history`

文件里先注册了一个 buffer：

```python
self.register_buffer("history", torch.zeros((num_envs, history_length, num_obs)), persistent=False)
```

它的作用是：

- 在 `act_inference()` 时维护推理历史
- 在 `reset()` 时对 done 环境清零

注意：

- 训练时的 `act()` 和 `evaluate()` 并不依赖这个内部 buffer
- 训练路径直接使用外部传进来的 `history`

所以这个内部 `history` 更偏向 **部署/推理态状态缓存**，不是训练主数据源。

### 5.2 Teacher Encoder

teacher encoder 定义为：

```python
self.teacher_encoder = nn.Sequential(
    MLP([num_critic_obs, *teacher_encoder_hidden_dims, latent_dim], activation),
    L2Norm() or SimNorm()
)
```

默认结构可写成：

```text
privileged_obs [B, num_critic_obs]
  -> Linear(num_critic_obs, 512)
  -> ELU
  -> Linear(512, 256)
  -> ELU
  -> Linear(256, latent_dim)
  -> L2Norm / SimNorm
  -> latent [B, latent_dim]
```

功能上它负责：

- 从特权观测中提取 teacher latent
- 作为 student latent 的监督目标

### 5.3 Student MoE Encoder

student encoder 不是普通 MLP，而是 `StudentMoEEncoder`：

```python
self.student_moe_encoder = StudentMoEEncoder(
    expert_num=expert_num,
    input_dim=num_obs * history_length,
    hidden_dims=student_encoder_hidden_dims,
    output_dim=latent_dim,
    activation=activation,
    norm_type=norm_type,
)
```

它内部又分为两层：

1. `MoE`
2. 输出归一化层 `L2Norm` 或 `SimNorm`

#### 5.3.1 `MoE` 的结构

`MoE` 由两部分组成：

- `gating_network`
- `experts`

##### A. Gating Network

定义为：

```python
self.gating_network = nn.Sequential(
    MLP([input_dim, *hidden_dims[:-1], expert_num], activation),
    nn.Softmax(dim=-1)
)
```

若默认 `hidden_dims = [512, 256, 256]`，则 gating 网络结构为：

```text
history [B, num_obs * history_length]
  -> Linear(input_dim, 512)
  -> ELU
  -> Linear(512, 256)
  -> ELU
  -> Linear(256, expert_num)
  -> Softmax
  -> weights [B, expert_num]
```

它输出每个样本对各个 expert 的混合权重。

##### B. Experts

`Experts` 结构不是 `expert_num` 个完全独立的 Python 子网络，而是：

1. 先经过一个共享 backbone MLP
2. 再用 `groups=expert_num` 的 `Conv1d(kernel_size=1)` 生成各 expert 输出

定义为：

```python
self.backbone = MLP(
    [input_dim, *backbone_hidden_dims, expert_num * expert_hidden_dim],
    activation,
    last_activation=True
)
self.experts = nn.Conv1d(
    in_channels=expert_num * expert_hidden_dim,
    out_channels=expert_num * output_dim,
    kernel_size=1,
    groups=expert_num,
)
```

若默认 `hidden_dims = [512, 256, 256]`，则：

- `backbone_hidden_dims = [512, 256]`
- `expert_hidden_dim = 256`

因此 experts 分支可写成：

```text
history [B, input_dim]
  -> shared backbone:
     Linear(input_dim, 512)
     ELU
     Linear(512, 256)
     ELU
     Linear(256, expert_num * 256)
     ELU
  -> reshape to [B, expert_num * 256, 1]
  -> grouped Conv1d( expert_num * 256 -> expert_num * latent_dim, groups=expert_num )
  -> reshape to [B, expert_num, latent_dim]
  -> per-expert latent
```

##### C. MoE 汇聚

MoE 前向为：

```python
weights = gating_network(x)          # [B, expert_num]
expert_outs = experts(x)             # [B, expert_num, latent_dim]
output = sum(weights * expert_outs)  # [B, latent_dim]
```

因此 student latent 的核心计算是：

```text
student latent = Σ_i gate_i(x) * expert_i(x)
```

#### 5.3.2 Student Encoder 输出归一化

MoE 输出之后还会过一个归一化层：

```python
latent = self.norm_layer(latent)
```

当前支持两种：

- `L2Norm`
- `SimNorm`

默认是 `L2Norm`。

所以 student encoder 的完整路径是：

```text
history
  -> MoE
  -> mixed latent
  -> L2Norm / SimNorm
  -> student latent
```

### 5.4 Actor

actor 定义为：

```python
self.actor = MLP([latent_dim + num_obs, *actor_hidden_dims, num_actions], activation)
```

默认结构：

```text
[latent, obs] [B, latent_dim + num_obs]
  -> Linear(input, 512)
  -> ELU
  -> Linear(512, 256)
  -> ELU
  -> Linear(256, 128)
  -> ELU
  -> Linear(128, num_actions)
  -> action_mean
```

随后动作分布定义为：

```python
Normal(mean, mean * 0.0 + self.std)
```

这表示：

- 动作分布是对角高斯
- 均值由 actor 输出
- 标准差 `std` 是一个可学习参数向量
- `std` 与输入无关，不是状态条件方差

### 5.5 Critic

critic 定义为：

```python
self.critic = MLP([latent_dim + num_critic_obs, *critic_hidden_dims, 1], activation)
```

默认结构：

```text
[detach(latent), privileged_obs] [B, latent_dim + num_critic_obs]
  -> Linear(input, 512)
  -> ELU
  -> Linear(512, 256)
  -> ELU
  -> Linear(256, 128)
  -> ELU
  -> Linear(128, 1)
  -> value
```

这里有一个重要细节：

```python
x = torch.cat([latent.detach(), privileged_obs], dim=1)
```

即 critic 明确使用了 **detach 之后的 latent**，因此 critic 的梯度不会回传到 encoder。

## 6. 前向路径

### 6.1 `act()`：训练时 teacher 分支

当 `is_teacher=True` 时：

```text
privileged_obs
  -> teacher_encoder
  -> latent_t
obs + latent_t
  -> actor
  -> Normal(mean, std)
  -> sample action
```

对应代码：

```python
latent = self.teacher_encoder(privileged_obs)
x = torch.cat([latent, obs], dim=1)
```

### 6.2 `act()`：训练时 student 分支

当 `is_teacher=False` 时：

```text
history
  -> student_moe_encoder
  -> latent_s
obs + latent_s
  -> actor
  -> Normal(mean, std)
  -> sample action
```

但这里有一个关键实现细节：

```python
with torch.no_grad():
    latent, _ = self.student_moe_encoder(history)
```

也就是说：

- student 分支在 `act()` 中前向时不保留梯度
- actor 的 PPO policy loss 不会直接更新 student encoder

### 6.3 `act_inference()`：部署/推理路径

推理时不会区分 teacher/student，而是永远走 student encoder：

```text
obs
  -> 更新内部 history buffer
  -> flatten history
  -> student_moe_encoder
  -> latent_s
  -> concat(latent_s, obs)
  -> actor
  -> actions_mean
```

这里返回的是 **动作均值**，不是采样动作。

### 6.4 `evaluate()`：价值评估路径

无论 teacher 还是 student，都会先得到 latent，然后：

```text
detach(latent) + privileged_obs
  -> critic
  -> value
```

区别只在 latent 来源：

- teacher：`teacher_encoder(privileged_obs)`
- student：`student_moe_encoder(history)`

但最终喂 critic 前都会 `detach()`。

## 7. 梯度与训练职责

这一点是理解该结构最关键的地方。

### 7.1 这个模块文件本身只定义网络，不定义 latent loss

`actor_critic_moe_cts.py` 只负责：

- 定义 encoder / actor / critic
- 定义 `act()` / `act_inference()` / `evaluate()`

student latent 对 teacher latent 的对齐损失，以及 load balance loss，不在这个文件里，而是在：

- `rsl_rl/rsl_rl/algorithms/moe_cts.py`

### 7.2 当前梯度关系

结合 `actor_critic_moe_cts.py` 和 `moe_cts.py`，当前训练关系如下：

- `teacher_encoder`
  - 会被 `optimizer1` 更新
  - 会受到 actor 侧 PPO loss 的影响
  - 不会通过 critic loss 接收梯度，因为 `evaluate()` 中 latent 被 detach

- `student_moe_encoder`
  - 不会通过 `act()` 中的 actor PPO loss 更新，因为 `act()` 的 student 分支用了 `torch.no_grad()`
  - 不会通过 critic loss 更新，因为 `evaluate()` 中 latent 被 detach
  - 会通过 `moe_cts.py` 中的 `latent_loss + load_balance_loss` 单独更新

- `actor`
  - teacher/student 两个分支共享同一个 actor
  - 由 PPO policy loss 更新

- `critic`
  - teacher/student 两个分支共享同一个 critic
  - 由 PPO value loss 更新

因此，这个结构本质上是：

```text
teacher encoder --(actor侧PPO)--> shared actor
student encoder --(latent mimic + load balance)--> mimic teacher latent
critic 只把 latent 当额外输入特征，不反向塑形 encoder
```

## 8. 这个文件的设计特点

从当前实现看，`ActorCriticMoECTS` 有 6 个非常鲜明的设计点：

1. **teacher/student 共享 actor 和 critic**
   - 不是两套 policy，而是两套 latent 生成器共享下游控制头。

2. **student 只看历史观测，不看 privileged_obs**
   - student encoder 输入是 `history = num_obs * history_length`。

3. **student encoder 使用 MoE，而不是单一 MLP**
   - 通过 gating network 给每个 expert 分配权重，再做加权融合。

4. **critic 用 privileged_obs，不用普通 obs**
   - actor 和 critic 的输入信息级别不同。

5. **critic 不反向更新 encoder**
   - `latent.detach()` 明确切断了 value 分支到 encoder 的梯度。

6. **student encoder 不直接吃 PPO 梯度**
   - student 主要靠 teacher latent 蒸馏和 load balance loss 学习。

## 9. 一句话总结

`actor_critic_moe_cts.py` 当前实现的不是“普通 actor-critic + 一个 MoE 模块”，而是：

> 一个共享 actor-critic 框架，上面挂了一个 **privileged teacher encoder** 和一个 **history-based MoE student encoder**；训练时 teacher 负责提供高质量 latent，student 通过 MoE 从历史观测中模仿这个 latent，最终两者共用同一个 actor 和 critic。

如果只看这一个文件，可以把它理解成：

```text
teacher latent generator
student MoE latent generator
        -> shared actor
        -> shared critic
```

