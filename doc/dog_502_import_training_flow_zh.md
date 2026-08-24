# dog_502 机器狗模型导入与训练流程说明

本文档说明本项目中 `dog_502` 机器狗模型如何导入到 `legged_gym` / Isaac Gym 训练链路中，以及如何启动训练、测试和后续 Mujoco 部署验证。

## 1. 当前导入方式总览

`dog_502` 训练任务的入口是：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 4096
```

这条命令背后的调用链如下：

```text
legged_gym/scripts/train.py
  -> from legged_gym.envs import *
  -> legged_gym/envs/__init__.py
  -> task_registry.register("dog_502_parkour_moe", Go2ParkourRobot, DOG502ParkourCfg(), DOG502CfgParkourMoE())
  -> legged_gym/envs/dog_502/dog_502_config_parkour_moe.py
  -> DOG502ParkourCfg.asset.file
  -> dog_502_description/urdf/dog_502_description.urdf
```

也就是说，训练端并不是直接读取 `resources/robots/dog_502/*.xml`，而是通过 Isaac Gym 加载 `dog_502_description/urdf/dog_502_description.urdf`。`resources/robots/dog_502/*.xml` 主要用于 Mujoco sim2sim 部署测试。

## 2. 关键目录与文件

```text
dog_502_description/
  urdf/dog_502_description.urdf       # Isaac Gym 训练加载的机器人 URDF
  meshes/*.STL                        # URDF 中引用的外观/碰撞网格
  xacro/                              # ROS/xacro 源文件和历史版本

legged_gym/envs/dog_502/
  dog_502_config_parkour_moe.py       # dog_502 训练配置，继承 Go2 跑酷配置

legged_gym/envs/__init__.py           # 注册 dog_502_parkour_moe task
legged_gym/scripts/train.py           # 训练入口
legged_gym/scripts/play.py            # Play 测试和策略导出入口

resources/robots/dog_502/
  dog_502.xml                         # Mujoco 主机器人模型
  flat.xml / stairs.xml / ...         # Mujoco 场景文件

deploy/deploy_mujoco/configs/
  dog_502_parkour_moe.yaml            # Mujoco 部署配置
```

## 3. URDF 和 mesh 准备

训练使用的 URDF 是：

```text
dog_502_description/urdf/dog_502_description.urdf
```

该 URDF 内部使用相对路径引用 mesh，例如：

```xml
<mesh filename="../meshes/base_link.STL"/>
<mesh filename="../meshes/FL_hip.STL"/>
```

因为 `LeggedRobot._create_envs()` 会把 `asset_root` 设置为 URDF 所在目录 `dog_502_description/urdf`，所以上面的 `../meshes/*.STL` 会解析到：

```text
dog_502_description/meshes/*.STL
```

检查要点：

- `dog_502_description/urdf/dog_502_description.urdf` 必须存在。
- URDF 中引用的所有 STL 必须能在 `dog_502_description/meshes/` 下找到。
- 12 个可控关节名必须和训练配置一致：
  - `FL_hip_joint`, `FL_thigh_joint`, `FL_calf_joint`
  - `FR_hip_joint`, `FR_thigh_joint`, `FR_calf_joint`
  - `RL_hip_joint`, `RL_thigh_joint`, `RL_calf_joint`
  - `RR_hip_joint`, `RR_thigh_joint`, `RR_calf_joint`
- 4 个足端 link 名中需要包含 `foot`，当前为 `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`。
- base link 名需要包含 `base`，用于跌倒终止接触检测。

## 4. dog_502 训练配置

核心配置文件是：

```text
legged_gym/envs/dog_502/dog_502_config_parkour_moe.py
```

其中 `DOG502ParkourCfg` 继承 `GO2ParkourCfg`，复用 Go2 的跑酷环境、观测、奖励和 parkour_moe 策略结构，只覆盖与机器人本体强相关的参数。

关键配置如下：

```python
class DOG502ParkourCfg(GO2ParkourCfg):
    class asset(GO2ParkourCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/dog_502_description/urdf/dog_502_description.urdf"
        name = "dog_502"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        flip_visual_attachments = False
```

含义：

- `asset.file` 指定 Isaac Gym 加载的 dog_502 URDF。
- `name` 是 Isaac Gym actor 名称。
- `foot_name = "foot"` 会自动匹配 `FL_foot` 等足端刚体。
- `penalize_contacts_on` 用于接触惩罚。
- `terminate_after_contacts_on = ["base"]` 表示 base 撞地时终止 episode。
- `flip_visual_attachments = False` 是为了避免 dog_502 的 STL 可视化网格被 Isaac Gym 额外旋转。

初始站姿配置：

```python
class init_state(GO2ParkourCfg.init_state):
    pos = [0.0, 0.0, 0.31]
    default_joint_angles = {
        "FL_hip_joint": 0.1,
        "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "FR_thigh_joint": 0.8,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    }
```

注意：`default_joint_angles` 的 key 必须覆盖 URDF 中 Isaac Gym 识别出来的全部 12 个可控 DOF，否则初始化 `default_dof_pos` 时会报 `KeyError`。

控制参数：

```python
class control(GO2ParkourCfg.control):
    stiffness = {"joint": 20.0}
    damping = {"joint": 0.5}
    action_scale = 0.25
```

这里利用 `"joint"` 子串匹配所有 dog_502 关节名，因为 12 个关节名都包含 `_joint`。

相机参数：

```python
class camera(GO2ParkourCfg.camera):
    local_pos = [0.255, 0.0, 0.04185]
    local_euler_xyz = [math.pi, math.radians(65.0), math.radians(-90.0)]
```

该位姿来自 `dog_502_description.urdf` 中的 `camera_joint`，用于 parkour_moe 的 proxy depth camera。

训练日志配置：

```python
class DOG502CfgParkourMoE(GO2CfgParkourMoE):
    class runner(GO2CfgParkourMoE.runner):
        experiment_name = "dog_502_parkour_moe"
        run_name = ""
```

训练结果会保存到：

```text
logs/dog_502_parkour_moe/
```

## 5. task 注册

`dog_502_parkour_moe` 在 `legged_gym/envs/__init__.py` 中注册：

```python
from legged_gym.envs.go2.go2_parkour_env import Go2ParkourRobot
from legged_gym.envs.dog_502.dog_502_config_parkour_moe import DOG502ParkourCfg, DOG502CfgParkourMoE

task_registry.register(
    "dog_502_parkour_moe",
    Go2ParkourRobot,
    DOG502ParkourCfg(),
    DOG502CfgParkourMoE(),
)
```

这里复用 `Go2ParkourRobot` 环境类，原因是 dog_502 和 Go2 都是 12 DOF 四足机器人，当前项目的跑酷观测、地形、奖励、视觉 proxy depth、parkour_moe runner 都写在 Go2 跑酷环境上。dog_502 的差异通过 `DOG502ParkourCfg` 覆盖。

## 6. 训练前安装

在仓库根目录执行：

```bash
pip install -e .
cd rsl_rl
pip install -e .
cd ..
```

建议使用项目匹配的 Isaac Gym / PyTorch / CUDA 环境。若动态库报错，可以设置：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1
```

## 7. 启动训练

常用训练命令：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 4096
```

显存不足时可以降低并行环境数量：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 2048
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 1024
```

快速冒烟测试可以限制迭代数：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 128 --max_iterations 2
```

继续训练：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 4096 --resume
```

指定某个 run 和 checkpoint：

```bash
python legged_gym/scripts/train.py \
  --task=dog_502_parkour_moe \
  --headless \
  --num_envs 4096 \
  --resume \
  --load_run ZApr21_15-05-55_ \
  --checkpoint 9900
```

## 8. Play 测试与策略导出

Play 命令：

```bash
python legged_gym/scripts/play.py --task=dog_502_parkour_moe
```

`play.py` 会把环境数量限制到最多 100 个，并设置 `train_cfg.runner.resume = True`，默认从 `logs/dog_502_parkour_moe/` 下加载 checkpoint。

如果 runner 支持 `export_policy()`，Play 时会导出 parkour_moe 推理 bundle：

```text
logs/dog_502_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt
```

该文件是后续 Mujoco / ONNX / TensorRT 流程的主要输入。

## 9. Mujoco 部署验证

Mujoco 端不读取训练 URDF，而读取 `resources/robots/dog_502/*.xml`。当前项目使用统一的部署脚本：

```bash
python deploy/deploy_mujoco/deploy_go2.py --config dog_502_parkour_moe.yaml
```

配置文件：

```text
deploy/deploy_mujoco/configs/dog_502_parkour_moe.yaml
```

关键字段：

```yaml
policy_type: "parkour_moe_bundle"
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/dog_502_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt"
xml_path: "{LEGGED_GYM_ROOT_DIR}/resources/robots/dog_502/stairs.xml"
action_scale: 0.25
num_actions: 12
num_obs: 45
```

要保持训练和 Mujoco 配置一致：

- `default_angles` 要和 `DOG502ParkourCfg.init_state.default_joint_angles` 顺序一致。
- `kps`, `kds`, `action_scale` 要和训练控制参数一致。
- `mujoco_joint_names` 和 `model_joint_names` 要覆盖同一组 12 个关节。
- `camera.local_pos` 和 `camera.local_euler_xyz` 要和训练端 dog_502 相机位姿一致。

## 10. ONNX / TensorRT 导出

从 Play 导出的 bundle 生成 ONNX：

```bash
python tools/export_parkour_moe_trt_assets.py \
  --bundle logs/dog_502_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt \
  --output-dir logs/dog_502_parkour_moe/exported/policies/parkour_moe_trt_assets
```

验证 ONNX 与 PyTorch bundle 输出：

```bash
python tools/validate_parkour_moe_onnx.py \
  --bundle logs/dog_502_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt \
  --onnx-dir logs/dog_502_parkour_moe/exported/policies/parkour_moe_trt_assets \
  --trials 20 \
  --steps 50 \
  --refresh-every 5
```

生成 TensorRT engine：

```bash
python tools/build_parkour_moe_trt_engines.py \
  --onnx-dir logs/dog_502_parkour_moe/exported/policies/parkour_moe_trt_assets \
  --output-dir logs/dog_502_parkour_moe/exported/policies/trt_engines \
  --precision both \
  --trtexec /usr/src/tensorrt/bin/trtexec
```

## 11. 常见问题检查

### 11.1 task 未注册

现象：

```text
ValueError: Task with name: dog_502_parkour_moe was not registered
```

检查：

- `legged_gym/envs/__init__.py` 是否 import 了 `DOG502ParkourCfg, DOG502CfgParkourMoE`。
- 是否有 `task_registry.register("dog_502_parkour_moe", ...)`。
- 是否从仓库根目录运行，或是否已经 `pip install -e .`。

### 11.2 URDF 或 mesh 找不到

检查：

- `DOG502ParkourCfg.asset.file` 指向的 URDF 是否存在。
- URDF 内部 mesh 的相对路径是否正确。
- 不要把训练端 `asset.file` 改成 Mujoco XML；Isaac Gym 训练端应加载 URDF。

### 11.3 `KeyError: xxx_joint`

原因通常是 URDF 中的 DOF 名称没有出现在 `default_joint_angles` 中。

处理：

- 检查 URDF 中 12 个 revolute joint 的名字。
- 确认 `DOG502ParkourCfg.init_state.default_joint_angles` 完整覆盖这些名字。

### 11.4 足端数量不对

`feet_indices` 由下面逻辑匹配：

```python
feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
```

当前 `foot_name = "foot"`，因此 body 名中需要包含 `foot`。正常应匹配到 4 个足端：`FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`。

### 11.5 初始姿态穿地或站不稳

优先调：

- `DOG502ParkourCfg.init_state.pos`
- `DOG502ParkourCfg.init_state.default_joint_angles`
- `DOG502ParkourCfg.rewards.base_height_target`
- `DOG502ParkourCfg.control.stiffness`
- `DOG502ParkourCfg.control.damping`

当前配置把初始 base 高度设为 `0.31`，base height target 设为 `0.315`，用于适配 dog_502 默认站姿。

### 11.6 Mujoco 命令找不到脚本

本项目当前没有 `deploy/deploy_mujoco/deploy_dog_502.py`。dog_502 使用现有统一脚本：

```bash
python deploy/deploy_mujoco/deploy_go2.py --config dog_502_parkour_moe.yaml
```

## 12. 推荐最小验证顺序

1. 静态检查 Python 语法：

```bash
python -m py_compile \
  legged_gym/envs/dog_502/dog_502_config_parkour_moe.py \
  legged_gym/envs/__init__.py \
  legged_gym/scripts/train.py \
  legged_gym/utils/task_registry.py
```

2. 检查 URDF 和 mesh 文件是否齐全。

3. 小规模冒烟训练：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 128 --max_iterations 2
```

4. 正式训练：

```bash
python legged_gym/scripts/train.py --task=dog_502_parkour_moe --headless --num_envs 4096
```

5. Play 导出推理 bundle：

```bash
python legged_gym/scripts/play.py --task=dog_502_parkour_moe
```

6. Mujoco sim2sim：

```bash
python deploy/deploy_mujoco/deploy_go2.py --config dog_502_parkour_moe.yaml
```
