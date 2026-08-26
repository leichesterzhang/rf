# RM75 平地 3 m/s Trot 冷启动训练

## 任务定义

任务名为 `RM75_flat_trot_3ms`，使用 RM75 固定机械臂模型、标准 PPO 和纯平面地形。
策略的目标命令最终固定为：

```text
lin_vel_x = 3.0 m/s
lin_vel_y = 0.0 m/s
ang_vel_yaw = 0.0 rad/s
```

这是一次冷启动训练：`resume = False`，actor、critic 和优化器均从随机初始化状态开始，
不会加载 `RM75_parkour_moe` 或其他历史 checkpoint。

## 速度课程

训练仍然属于同一次冷启动，但为了避免随机策略直接面对 3 m/s 时大量摔倒，前 18000 次
迭代逐步提高前进速度：

| PPO 迭代 | 前进速度采样范围（m/s） |
| ---: | ---: |
| 0 | 0.5–1.0 |
| 1500 | 0.8–1.3 |
| 3500 | 1.2–1.7 |
| 6000 | 1.6–2.1 |
| 9000 | 2.0–2.5 |
| 12000 | 2.4–2.8 |
| 15000 | 2.7–3.0 |
| 18000–30000 | 固定 3.0 |

## Trot 约束

环境按 `FL、FR、RL、RR` 的脚序计算触地状态，奖励两种交替支撑相：

- `FL + RR` 同时支撑；
- `FR + RL` 同时支撑。

奖励还使用每只脚触地占空比的指数滑动平均，防止策略长期只依赖一组对角腿。
此外，支撑脚水平滑动会被单独惩罚。训练日志中的 `rew_trot_gait`、
`rew_feet_slip` 和 `rew_tracking_lin_vel` 是主要观测指标。

## 启动命令

先做小规模连通性检查：

```bash
python -m legged_gym.scripts.train \
  --task=RM75_flat_trot_3ms \
  --headless \
  --num_envs=64 \
  --max_iterations=2
```

正式冷启动训练：

```bash
python -m legged_gym.scripts.train \
  --task=RM75_flat_trot_3ms \
  --headless \
  --num_envs=4096
```

正式训练命令不要添加 `--resume`。显存不足时可以把 `--num_envs` 降为 `2048` 或
`1024`，速度课程的 PPO 迭代节点不需要改变。checkpoint 默认每 100 次迭代保存到：

```text
logs/RM75_flat_trot_3ms/<run>/model_<iteration>.pt
```

训练后加载最新 checkpoint，以固定 3 m/s 命令进行 Isaac Gym 验证和策略导出：

```bash
python -m legged_gym.scripts.play --task=RM75_flat_trot_3ms --num_envs=1
```

这里使用 `python -m` 是为了确保 Python 优先加载当前工作区代码，避免本机旧的 editable
install 指向另一个同名仓库副本。
