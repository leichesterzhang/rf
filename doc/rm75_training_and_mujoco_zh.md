# RM75 训练与 MuJoCo 验证

RM75 任务名为 `RM75_parkour_moe`。训练端使用
`RM75/urdf/RM75_locked_arm.urdf`，该文件由
`tools/build_rm75_assets.py` 从原始 RM75 URDF 生成。

锁定姿态为：

- `link1_joint`、`link3_joint`、`link4_joint`、`link5_joint`、`link7_joint`: `0 rad`
- `link2_joint`: `-pi/2 rad`
- `link6_joint`: `+pi/2 rad`

机械臂关节已全部改为固定关节，因此训练和策略推理均只有 12 维腿部动作；机械臂的 link、质量和惯量仍保留在物理模型中。

## 训练

先用小规模并行环境确认 Isaac Gym 可以加载模型：

```bash
python legged_gym/scripts/train.py \
  --task=RM75_parkour_moe \
  --headless \
  --num_envs=128 \
  --max_iterations=2
```

正式训练可从 `1024` 或 `2048` 个并行环境开始，并根据显存调整：

```bash
python legged_gym/scripts/train.py \
  --task=RM75_parkour_moe \
  --headless \
  --num_envs=2048
```

训练配置位于 `legged_gym/envs/rm75/rm75_config_parkour_moe.py`。其中平地、粗糙平地、斜坡地形各占 8%，总占比为 24%；地形池其余部分保留间隙、石块、横梁、台阶等多地形训练。

## Gym 验证与策略导出

```bash
python legged_gym/scripts/play.py --task=RM75_parkour_moe
```

该命令会加载最新 RM75 checkpoint，并导出：

```text
logs/RM75_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt
```

RM75 默认只做常规行走验证，不会被 `play.py` 强制初始化为侧翻姿态。

## MuJoCo 验证与状态记录

MuJoCo 使用 `resources/robots/RM75/RM75.xml` 和同目录下的 13 个场景。部署配置为：

```text
deploy/deploy_mujoco/configs/RM75_parkour_moe.yaml
```

运行粗糙平地验证并同时导出状态文件和 HTML 查看器：

```bash
python deploy/deploy_mujoco/deploy_go2.py \
  --config RM75_parkour_moe.yaml \
  --export-state-html
```

状态记录默认开启，每 10 个 MuJoCo 仿真步采样一次，输出为：

```text
action_data/RM75/<terrain>_<timestamp>.npz
action_data/RM75/<terrain>_<timestamp>.html
```

切换测试场景时，在 YAML 中修改 `xml_path`，可选场景包括
`flat`、`rough_flat`、`ramp`、`stairs_up`、`single_gap`、`step_stone`、
`two_row_stones`、`one_row_stones`、`single_bridge`、`air_beams`、
`air_stones`、`hurdle` 和 `corridor`。

若原始 `RM75/urdf/四足机器人RM75.urdf` 或网格文件发生更新，重新执行：

```bash
python tools/build_rm75_assets.py
```
