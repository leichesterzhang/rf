# dog_502 四足机器狗训练项目软件使用说明

## 1. 软件概述

本软件用于 dog_502 四足机器狗在复杂非结构化地形上的强化学习训练、仿真测试、策略导出和 MuJoCo sim2sim 验证。项目基于 `legged_gym`、Isaac Gym、`rsl_rl` 和 MuJoCo 构建，当前 dog_502 训练任务名为 `dog_502_parkour_moe`。该任务复用 Go2 跑酷训练框架中的 MGDP 地形生成、Parkour-MoE 策略结构、代理深度相机、跌倒起立训练逻辑和批量仿真测试流程，并针对 dog_502 的 URDF、关节名、初始姿态、PD 参数、相机位姿和 MuJoCo 模型进行了适配。

软件支持在 Isaac Gym 中进行并行强化学习训练，在 Play 模式下加载 checkpoint 并导出推理策略，在 MuJoCo 中使用独立物理引擎进行 sim2sim 复核。当前支持的主要训练与测试地形包括 `single_gap`、`step_stone`、`two_row_stones`、`one_row_stones`、`single_bridge`、`air_beams`、`air_stones`、`hurdle`、`ramp`、`corridor`、`stairs_up` 和 `flat`。

## 2. 运行环境要求

推荐在 Ubuntu 18.04 或更高版本上运行本项目，并使用 NVIDIA GPU 进行 Isaac Gym 并行仿真训练。建议驱动版本为 525 或更高，Python 版本推荐 3.8。训练环境需要安装 Isaac Gym、PyTorch、CUDA、项目内修改版 `rsl_rl`、MuJoCo、PyYAML、TensorBoard、ONNX 和 pygame 等依赖。

项目的 Python 包依赖在 [setup.py](/home/gh/GYM/5-28/setup.py) 中声明，其中 MuJoCo 版本固定为 `mujoco==3.2.3`。如果只进行代码阅读或配置检查，不需要完整 GPU 环境；如果要实际训练或运行 MuJoCo 仿真，则必须确保 `isaacgym` 和 `mujoco` Python 包可以正常导入。

## 3. 主要目录说明

dog_502 的机器人描述文件位于 [dog_502_description](/home/gh/GYM/5-28/dog_502_description)，其中训练端使用 [dog_502_description.urdf](/home/gh/GYM/5-28/dog_502_description/urdf/dog_502_description.urdf)，网格文件位于 `dog_502_description/meshes/`。Isaac Gym 训练时直接加载该 URDF，不使用 MuJoCo XML。

训练配置位于 [legged_gym/envs/dog_502/dog_502_config_parkour_moe.py](/home/gh/GYM/5-28/legged_gym/envs/dog_502/dog_502_config_parkour_moe.py)。任务注册位于 [legged_gym/envs/__init__.py](/home/gh/GYM/5-28/legged_gym/envs/__init__.py)，其中 `dog_502_parkour_moe` 被注册为可训练任务。

MuJoCo 使用的 dog_502 主模型位于 [resources/robots/dog_502/dog_502.xml](/home/gh/GYM/5-28/resources/robots/dog_502/dog_502.xml)，各类 MuJoCo 测试场景位于 [resources/robots/dog_502](/home/gh/GYM/5-28/resources/robots/dog_502)。MuJoCo 部署配置位于 [deploy/deploy_mujoco/configs/dog_502_parkour_moe.yaml](/home/gh/GYM/5-28/deploy/deploy_mujoco/configs/dog_502_parkour_moe.yaml)。

训练入口脚本为 [train.py](/home/gh/GYM/5-28/legged_gym/scripts/train.py)，Play 与策略导出入口为 [play.py](/home/gh/GYM/5-28/legged_gym/scripts/play.py)，MuJoCo 仿真入口为 [deploy_go2.py](/home/gh/GYM/5-28/deploy/deploy_mujoco/deploy_go2.py)。虽然 MuJoCo 脚本文件名仍保留 `go2`，但通过 `dog_502_parkour_moe.yaml` 可加载 dog_502 模型和策略。

## 4. 安装与环境配置

建议使用 Conda 创建独立环境：

```bash
conda create -n dog502-rl python=3.8
conda activate dog502-rl
```

安装 PyTorch 时请根据本机 CUDA 版本选择合适命令。项目文档推荐的 CUDA 12.1 环境可使用：

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Isaac Gym 需要从 NVIDIA 官网下载。解压后进入 `isaacgym/python` 目录安装：

```bash
cd isaacgym/python
pip install -e .
```

安装项目和项目内的 `rsl_rl`：

```bash
cd /home/gh/GYM/5-28
pip install -e .

cd rsl_rl
pip install -e .
cd ..
```

如果启动训练或可视化时遇到 Vulkan、OpenGL 或动态库问题，可在终端中设置：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1
```

## 5. 训练 dog_502 策略

dog_502 的训练任务名为 `dog_502_parkour_moe`。常规训练命令如下：

```bash
cd /home/gh/GYM/5-28
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 4096
```

显存不足时可以降低并行环境数量：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 2048
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 1024
```

首次接入或修改配置后，建议先进行小规模冒烟测试：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 128 --max_iterations 2
```

训练日志默认保存到：

```text
logs/dog_502_parkour_moe/<日期时间>_<run_name>/
```

训练过程中会周期性保存模型，例如：

```text
logs/dog_502_parkour_moe/Jun24_10-30-00_/model_100.pt
logs/dog_502_parkour_moe/Jun24_10-30-00_/model_200.pt
```

## 6. 继续训练与加载 checkpoint

如果需要从最近一次训练继续运行，可以使用：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 4096 --resume
```

如果需要指定某个 run 和 checkpoint，可以使用：

```bash
python legged_gym/scripts/train.py \
  --task=dog_502_parkour_moe \
  --headless \
  --num_envs 4096 \
  --resume \
  --load_run Jun24_10-30-00_ \
  --checkpoint 3000
```

其中 `--load_run` 对应 `logs/dog_502_parkour_moe/` 下的某个运行目录名，`--checkpoint` 对应模型编号。如果不指定 `--checkpoint`，程序会尝试加载该 run 下最新 checkpoint。

## 7. Play 测试与策略导出

训练完成后，可以使用 Play 模式查看策略表现并导出推理策略：

```bash
python legged_gym/scripts/play.py --task=dog_502_parkour_moe
```

Play 默认会从 `logs/dog_502_parkour_moe/` 下加载最新训练结果。如果需要指定 run 或 checkpoint，可以使用：

```bash
python legged_gym/scripts/play.py \
  --task=dog_502_parkour_moe \
  --load_run Jun24_10-30-00_ \
  --checkpoint 3000
```

对于 parkour-MoE 策略，Play 后应关注导出的 bundle 文件：

```text
logs/dog_502_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt
```

该文件是 MuJoCo sim2sim、ONNX 导出和 TensorRT 部署流程的主要输入。如果该文件不存在，通常说明还没有成功执行 Play 导出，或者 Play 加载的 checkpoint 不是 parkour-MoE 训练结果。

## 8. MuJoCo sim2sim 仿真测试

MuJoCo 测试使用统一脚本：

```bash
python deploy/deploy_mujoco/deploy_go2.py --config dog_502_parkour_moe.yaml
```

该命令会读取 [dog_502_parkour_moe.yaml](/home/gh/GYM/5-28/deploy/deploy_mujoco/configs/dog_502_parkour_moe.yaml)。默认策略路径为：

```text
logs/dog_502_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt
```

默认测试场景为：

```text
resources/robots/dog_502/single_gap.xml
```

如需切换测试地形，修改 YAML 中的 `xml_path` 字段即可。例如切换到斜坡：

```yaml
xml_path: "{LEGGED_GYM_ROOT_DIR}/resources/robots/dog_502/ramp.xml"
```

切换到碎石/踏石类场景：

```yaml
xml_path: "{LEGGED_GYM_ROOT_DIR}/resources/robots/dog_502/step_stone.xml"
```

可选场景包括：

```text
single_gap.xml
step_stone.xml
two_row_stones.xml
one_row_stones.xml
single_bridge.xml
air_beams.xml
air_stones.xml
hurdle.xml
ramp.xml
corridor.xml
stairs_up.xml
flat.xml
parkour_hard_line.xml
```

MuJoCo 配置中的 `mujoco_joint_names` 和 `model_joint_names` 必须保持一致，并与训练端 12 个关节顺序一致。当前 dog_502 配置已经按 `FL, FR, RL, RR` 每条腿 `hip, thigh, calf` 的顺序设置。

## 9. ONNX 与 TensorRT 导出

从 Play 导出的 parkour-MoE bundle 可以进一步导出为 ONNX：

```bash
python tools/export_parkour_moe_trt_assets.py \
  --bundle logs/dog_502_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt \
  --output-dir logs/dog_502_parkour_moe/exported/policies/parkour_moe_trt_assets
```

导出后可验证 ONNX 与 PyTorch bundle 的推理一致性：

```bash
python tools/validate_parkour_moe_onnx.py \
  --bundle logs/dog_502_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt \
  --onnx-dir logs/dog_502_parkour_moe/exported/policies/parkour_moe_trt_assets \
  --trials 20 \
  --steps 50 \
  --refresh-every 5
```

如果目标平台支持 TensorRT，可继续构建 TensorRT engine：

```bash
python tools/build_parkour_moe_trt_engines.py \
  --onnx-dir logs/dog_502_parkour_moe/exported/policies/parkour_moe_trt_assets \
  --output-dir logs/dog_502_parkour_moe/exported/policies/trt_engines \
  --precision both \
  --trtexec /usr/src/tensorrt/bin/trtexec
```

## 10. 地形与测试场景说明

Isaac Gym 训练端使用 MGDP parkour 地形生成器，地形配置位于 `GO2ParkourCfg.terrain` 并由 dog_502 配置继承。训练端的地形集合为：

```text
[single_gap, step_stone, two_row_stones, one_row_stones, single_bridge,
 air_beams, air_stones, hurdle, ramp, corridor, stairs_up, flat]
```

其中 `ramp` 用于斜坡训练与测试，最大坡面角配置为 30°；`step_stone`、`two_row_stones`、`one_row_stones` 可用于碎石、踏石和离散落脚区域测试；`flat` 可用于普通步态、跌倒起立和基础控制稳定性测试；`parkour_hard_line.xml` 可在 MuJoCo 中用于复合路线测试。

如果需要临时只训练某一种地形，可以修改 [go2_config_parkour_moe.py](/home/gh/GYM/5-28/legged_gym/envs/go2/go2_config_parkour_moe.py) 或 dog_502 子类配置中的 `terrain_proportions`。例如只训练 `stairs_up` 时，将第 10 类设置为 1.0，其余设置为 0.0。修改后建议先用小规模命令进行冒烟测试。

## 11. 关键配置说明

dog_502 的训练配置主要覆盖机器人本体参数。`asset.file` 指向 dog_502 的 URDF，`foot_name = "foot"` 用于匹配 `FL_foot`、`FR_foot`、`RL_foot`、`RR_foot`，`terminate_after_contacts_on = ["base"]` 表示 base 撞地会终止 episode。`default_joint_angles` 必须覆盖 URDF 中所有 12 个可控关节，否则训练初始化时会出现 `KeyError`。

控制参数当前设置为：

```python
stiffness = {"joint": 20.0}
damping = {"joint": 0.5}
action_scale = 0.25
```

相机参数当前设置为：

```python
local_pos = [0.255, 0.0, 0.04185]
local_euler_xyz = [math.pi, math.radians(65.0), math.radians(-90.0)]
```

如果后续 dog_502 的 URDF、相机安装位置、足端 link 名、关节顺序或关节限位发生变化，应同步检查训练配置、MuJoCo XML 和 YAML 配置，避免 Isaac Gym 与 MuJoCo 中的状态顺序不一致。

## 12. 日志、结果与可视化

训练日志位于：

```text
logs/dog_502_parkour_moe/
```

TensorBoard 可用于查看训练曲线：

```bash
tensorboard --logdir logs/dog_502_parkour_moe
```

模型 checkpoint 文件通常命名为 `model_<iteration>.pt`。Play 导出的推理文件位于：

```text
logs/dog_502_parkour_moe/exported/policies/
```

MuJoCo 测试时，如果开启深度预览窗口，可以观察策略深度图、token mask、地形预测信息和重建高度点。若使用 `--save-video` 参数，视频会保存到部署脚本配置的输出目录中。

## 13. 常见问题处理

如果训练时报 `Task with name: dog_502_parkour_moe was not registered`，应检查 `legged_gym/envs/__init__.py` 中是否导入了 `DOG502ParkourCfg` 和 `DOG502CfgParkourMoE`，并确认存在 `task_registry.register("dog_502_parkour_moe", ...)`。同时需要确认是在仓库根目录运行命令，并且已经执行过 `pip install -e .`。

如果 Isaac Gym 报 URDF 或 mesh 找不到，应检查 `dog_502_description/urdf/dog_502_description.urdf` 是否存在，以及 URDF 内部 `../meshes/*.STL` 引用的文件是否都存在于 `dog_502_description/meshes/`。训练端不要把 `asset.file` 改成 MuJoCo XML。

如果训练初始化时报 `KeyError: xxx_joint`，说明 URDF 中某个可控 DOF 没有出现在 `default_joint_angles` 中。需要检查 URDF 中所有 `revolute` 关节，并确保 dog_502 配置完整覆盖这 12 个关节名。

如果足端数量不正确，应检查 URDF 的足端 link 名是否包含 `foot`。当前 dog_502 的足端为 `FL_foot`、`FR_foot`、`RL_foot`、`RR_foot`，与 `foot_name = "foot"` 匹配。

如果 MuJoCo 启动时报策略文件不存在，应先执行 Play 导出：

```bash
python legged_gym/scripts/play.py --task=dog_502_parkour_moe
```

如果 MuJoCo 启动时报 XML 或 mesh 找不到，应检查 `resources/robots/dog_502/dog_502.xml` 中的 `meshdir` 路径，以及场景 XML 是否包含 `<include file="dog_502.xml"/>`。

如果机器人在初始姿态下穿地、抖动或站不稳，优先检查 `DOG502ParkourCfg.init_state.pos`、`default_joint_angles`、`rewards.base_height_target`、`control.stiffness` 和 `control.damping`。调整后应先用 `--num_envs 128 --max_iterations 2` 进行冒烟测试。

## 14. 推荐使用流程

首次使用时，建议按以下顺序执行。先完成 Python 环境、Isaac Gym、项目包和 `rsl_rl` 安装。然后运行小规模训练冒烟测试，确认 dog_502 URDF、mesh、关节名和任务注册无误。接着启动正式训练，训练完成后用 Play 加载 checkpoint 并导出 `parkour_moe_inference_bundle.pt`。最后使用 MuJoCo 配置文件加载导出的 bundle，在 `flat`、`ramp`、`step_stone`、`stairs_up` 和 `parkour_hard_line` 等场景中做 sim2sim 检查。

推荐命令顺序如下：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 128 --max_iterations 2
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 4096
python legged_gym/scripts/play.py --task=dog_502_parkour_moe
python deploy/deploy_mujoco/deploy_go2.py --config dog_502_parkour_moe.yaml
```

完成上述流程后，即可根据需要进行 ONNX/TensorRT 导出、批量测试统计和后续实机部署适配。
