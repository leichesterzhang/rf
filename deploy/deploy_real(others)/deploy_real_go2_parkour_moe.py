from pathlib import Path

LEGGED_GYM_ROOT_DIR = str(Path(__file__).resolve().parents[2])

import argparse
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation
from config_go2 import Config, resolve_repo_path
from parkour_moe_policy import (
    ParkourMoEPolicyAdapter,
    ParkourMoETRTPolicyAdapter,
    parkour_moe_engine_assets_available,
)

HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0


def init_cmd_go2(cmd: LowCmdGo):
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(12):
        cmd.motor_cmd[i].mode = 0x0A
        cmd.motor_cmd[i].q = PosStopF
        cmd.motor_cmd[i].dq = VelStopF
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0


def resize_float_image(image, output_height, output_width):
    if image.shape == (output_height, output_width):
        return image.astype(np.float32)

    image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        image_tensor,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def load_image_as_depth_frame(image_path, output_height, output_width):
    image = imageio.imread(resolve_repo_path(image_path))
    if image.ndim == 3:
        image = image[..., :3].astype(np.float32).mean(axis=2)
    elif image.ndim > 3:
        image = image[0].astype(np.float32)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
    else:
        image = image.astype(np.float32)

    image = image.astype(np.float32)
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max - image_min > 1.0e-6:
        image = (image - image_min) / (image_max - image_min)
    elif image_max > 1.0:
        image = image / image_max
    image = np.clip(image, 0.0, 1.0)
    image = resize_float_image(image, output_height, output_width)
    return image.astype(np.float32) - 0.5


def build_random_depth_frame(rng, output_height, output_width):
    return rng.random((output_height, output_width), dtype=np.float32) - 0.5


def normalize_depth_image(image, output_height, output_width, clipping_range):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]

    valid_mask = np.isfinite(image)
    if not np.any(valid_mask):
        return np.zeros((output_height, output_width), dtype=np.float32)

    image = image.copy()
    image[~valid_mask] = clipping_range
    if float(np.max(image)) > clipping_range * 10.0:
        image = image / 1000.0

    image = np.clip(image, 0.0, clipping_range)
    image = resize_float_image(image, output_height, output_width)
    return image.astype(np.float32) / clipping_range - 0.5


class Ros2DepthInputSubscriber:
    def __init__(self, camera_cfg):
        self.camera_cfg = dict(camera_cfg)
        self.output_height = int(self.camera_cfg.get("output_height", 58))
        self.output_width = int(self.camera_cfg.get("output_width", 87))
        self.clipping_range = max(1.0e-6, float(self.camera_cfg.get("clipping_range", 2.0)))
        self.topic = str(self.camera_cfg.get("ros2_topic", "/forward_depth_image"))
        self.msg_type = str(self.camera_cfg.get("ros2_msg_type", "float32_multi_array")).lower()
        self.initial_wait_sec = max(0.0, float(self.camera_cfg.get("ros2_initial_wait_sec", 2.0)))
        self.frame_timeout_sec = max(0.0, float(self.camera_cfg.get("ros2_frame_timeout_sec", 0.5)))
        self.spin_timeout_sec = max(0.0, float(self.camera_cfg.get("ros2_spin_timeout_sec", 0.0)))
        self.latest_frame = None
        self.last_msg_time = None
        self._stale_warning_printed = False
        self._missing_frame_warning_printed = False
        self._owns_rclpy = False

        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            from std_msgs.msg import Float32MultiArray
        except ImportError as exc:  # pragma: no cover - depends on robot ROS2 environment
            raise ImportError(
                "ROS2 depth input is enabled, but Python ROS2 dependencies are unavailable."
            ) from exc

        self.rclpy = rclpy
        self.Node = Node
        self.qos_profile_sensor_data = qos_profile_sensor_data
        self.Image = Image
        self.Float32MultiArray = Float32MultiArray

        try:
            self.rclpy.init(args=None)
            self._owns_rclpy = True
        except RuntimeError:
            self._owns_rclpy = False

        self.node = self.Node("parkour_moe_depth_subscriber")
        if self.msg_type == "image":
            message_type = self.Image
            qos_profile = self.qos_profile_sensor_data
        elif self.msg_type in ("float32_multi_array", "float32multiarray", "array"):
            message_type = self.Float32MultiArray
            qos_profile = 1
        else:
            raise ValueError(f"Unsupported ROS2 depth msg type: {self.msg_type}")

        self.subscription = self.node.create_subscription(
            message_type,
            self.topic,
            self._depth_callback,
            qos_profile,
        )
        print(
            f"ROS2 depth subscriber enabled: topic={self.topic}, msg_type={self.msg_type}, "
            f"shape=({self.output_height}, {self.output_width})"
        )

    def _float_array_to_frame(self, msg):
        flat = np.asarray(msg.data, dtype=np.float32)
        expected = self.output_height * self.output_width
        if flat.size == expected:
            return flat.reshape(self.output_height, self.output_width)
        if flat.size % expected == 0:
            return flat.reshape(-1, self.output_height, self.output_width)[-1]
        raise ValueError(
            f"Unexpected Float32MultiArray size from ROS2 depth topic: {flat.size} (expected {expected})"
        )

    def _image_msg_to_frame(self, msg):
        encoding = str(msg.encoding or "").lower()
        dtype_map = {
            "32fc1": np.float32,
            "16uc1": np.uint16,
            "mono16": np.uint16,
            "8uc1": np.uint8,
            "mono8": np.uint8,
        }
        if encoding not in dtype_map:
            raise ValueError(f"Unsupported ROS2 depth image encoding: {msg.encoding}")

        dtype = dtype_map[encoding]
        itemsize = np.dtype(dtype).itemsize
        row_elems = int(msg.step) // itemsize
        data = np.frombuffer(msg.data, dtype=dtype)
        expected_elems = int(msg.height) * row_elems
        if data.size < expected_elems:
            raise ValueError(
                f"Malformed ROS2 depth image payload: got {data.size} elements, expected at least {expected_elems}"
            )

        image = data[:expected_elems].reshape(int(msg.height), row_elems)
        image = image[:, : int(msg.width)]
        return normalize_depth_image(
            image,
            output_height=self.output_height,
            output_width=self.output_width,
            clipping_range=self.clipping_range,
        )

    def _depth_callback(self, msg):
        try:
            if isinstance(msg, self.Float32MultiArray):
                frame = self._float_array_to_frame(msg)
            else:
                frame = self._image_msg_to_frame(msg)
        except Exception as exc:
            self.node.get_logger().error(
                f"Failed to decode ROS2 depth input: {exc}",
                throttle_duration_sec=1.0,
            )
            return

        self.latest_frame = frame.astype(np.float32, copy=True)
        self.last_msg_time = time.monotonic()
        self._stale_warning_printed = False
        self._missing_frame_warning_printed = False

    def spin_once(self):
        self.rclpy.spin_once(self.node, timeout_sec=self.spin_timeout_sec)

    def wait_for_first_frame(self):
        if self.latest_frame is not None:
            return self.latest_frame.copy()

        if self.initial_wait_sec <= 0.0:
            return None

        deadline = time.monotonic() + self.initial_wait_sec
        while time.monotonic() < deadline and self.latest_frame is None:
            self.spin_once()
            time.sleep(0.01)

        if self.latest_frame is None:
            print(
                f"ROS2 depth subscriber did not receive a frame within {self.initial_wait_sec:.2f}s; "
                "falling back to the placeholder depth frame for now."
            )
            return None
        return self.latest_frame.copy()

    def has_recent_frame(self):
        if self.latest_frame is None or self.last_msg_time is None:
            return False
        if self.frame_timeout_sec <= 0.0:
            return True
        return (time.monotonic() - self.last_msg_time) <= self.frame_timeout_sec

    def get_latest_frame(self):
        self.spin_once()
        if self.latest_frame is None:
            if not self._missing_frame_warning_printed:
                print("ROS2 depth subscriber has not received any frame yet; using the fallback frame.")
                self._missing_frame_warning_printed = True
            return None
        if not self.has_recent_frame():
            if not self._stale_warning_printed:
                print(
                    f"ROS2 depth frame is stale (> {self.frame_timeout_sec:.2f}s); "
                    "using the fallback frame until new data arrives."
                )
                self._stale_warning_printed = True
            return None
        self._stale_warning_printed = False
        return self.latest_frame.copy()

    def close(self):
        if hasattr(self, "node") and self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self._owns_rclpy:
            try:
                self.rclpy.shutdown()
            except RuntimeError:
                pass


def resolve_button_index(button_name):
    if button_name is None:
        return None
    if isinstance(button_name, int):
        return button_name
    if not hasattr(KeyMap, str(button_name)):
        raise ValueError(f"Unknown blind-walk toggle button: {button_name}")
    return getattr(KeyMap, str(button_name))


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def resolve_engine_dir(config: Config):
    if getattr(config, "engine_dir", ""):
        return config.engine_dir

    policy_path = Path(config.policy_path)
    if policy_path.is_dir():
        return str(policy_path)
    if policy_path.suffix == ".engine":
        return str(policy_path.parent)
    return str(policy_path.parent / "trt_engines")


def infer_policy_type(config: Config):
    explicit_type = str(getattr(config, "policy_type", "auto") or "auto").lower()
    if explicit_type != "auto":
        return explicit_type

    engine_dir = resolve_engine_dir(config)
    if parkour_moe_engine_assets_available(
        engine_dir=engine_dir,
        enable_selector=bool(config.camera.get("enable_selector", False)),
        precision=getattr(config, "engine_precision", "auto"),
        engine_paths=getattr(config, "engine_paths", None),
    ):
        return "parkour_moe_engine"
    return "parkour_moe_bundle"


def resolve_policy_device(policy_type, requested_device):
    device = torch.device(requested_device)
    if policy_type == "parkour_moe_engine" and device.type != "cuda":
        print(
            f"TensorRT engine deployment requires CUDA. Overriding policy device from {requested_device} to cuda:0."
        )
        return "cuda:0"
    return requested_device


class RealDepthInputProvider:
    def __init__(self, camera_cfg, blind_walk_cfg):
        self.camera_cfg = dict(camera_cfg)
        self.blind_walk_cfg = dict(blind_walk_cfg)
        self.update_interval = max(1, int(self.camera_cfg.get("update_interval", 5)))
        self.buffer_len = max(1, int(self.camera_cfg.get("buffer_len", 2)))
        self.output_height = int(self.camera_cfg.get("output_height", 58))
        self.output_width = int(self.camera_cfg.get("output_width", 87))
        self.randomize_each_refresh = bool(self.blind_walk_cfg.get("randomize_each_refresh", False))
        self.rng = np.random.default_rng(int(self.blind_walk_cfg.get("random_seed", 1234)))
        self.ros2_subscriber = None
        self.ros2_init_failed = False

        self.default_frame = self._build_default_frame()
        self.blind_frame = self._build_blind_frame()
        self.blind_walk_enabled = bool(self.blind_walk_cfg.get("enabled", False))
        self.has_default_depth_source = bool(self.camera_cfg.get("image_path", "")) or self._ros2_source_requested()

        initial_frame = None
        if not self.blind_walk_enabled:
            initial_frame = self._prime_default_source()
        if initial_frame is None:
            initial_frame = self._current_frame()
        self.depth_buffer = np.repeat(initial_frame[None, :, :], self.buffer_len, axis=0).astype(np.float32)
        self.needs_policy_refresh = True

    def _build_default_frame(self):
        image_path = self.camera_cfg.get("image_path", "")
        if image_path:
            print(f"Real deploy image source: {image_path}")
            return load_image_as_depth_frame(image_path, self.output_height, self.output_width)
        print("Real deploy image source: using ROS2 depth topic when available, otherwise a zero depth placeholder.")
        return np.zeros((self.output_height, self.output_width), dtype=np.float32)

    def _build_blind_frame(self):
        image_path = self.blind_walk_cfg.get("image_path", "")
        if image_path:
            print(f"Blind-walk image source: {image_path}")
            return load_image_as_depth_frame(image_path, self.output_height, self.output_width)
        print("Blind-walk image source: using a generated random image.")
        return build_random_depth_frame(self.rng, self.output_height, self.output_width)

    def _ros2_source_requested(self):
        if self.camera_cfg.get("image_path", ""):
            return False
        return bool(self.camera_cfg.get("ros2_enabled", True))

    def _ensure_ros2_subscriber(self):
        if not self._ros2_source_requested() or self.ros2_init_failed:
            return None
        if self.ros2_subscriber is not None:
            return self.ros2_subscriber

        try:
            self.ros2_subscriber = Ros2DepthInputSubscriber(self.camera_cfg)
        except Exception as exc:
            self.ros2_init_failed = True
            print(f"Failed to initialize ROS2 depth subscriber: {exc}")
            return None
        return self.ros2_subscriber

    def _prime_default_source(self):
        subscriber = self._ensure_ros2_subscriber()
        if subscriber is None:
            return None
        return subscriber.wait_for_first_frame()

    def _get_live_default_frame(self):
        subscriber = self._ensure_ros2_subscriber()
        if subscriber is None:
            return None
        return subscriber.get_latest_frame()

    def _current_frame(self):
        if self.blind_walk_enabled:
            return self.blind_frame
        live_frame = self._get_live_default_frame()
        if live_frame is not None:
            return live_frame
        return self.default_frame

    def set_blind_walk(self, enabled):
        enabled = bool(enabled)
        if self.blind_walk_enabled == enabled:
            return
        self.blind_walk_enabled = enabled
        if self.blind_walk_enabled and self.randomize_each_refresh and not self.blind_walk_cfg.get("image_path", ""):
            self.blind_frame = build_random_depth_frame(self.rng, self.output_height, self.output_width)
        current = None
        if not self.blind_walk_enabled:
            current = self._prime_default_source()
        if current is None:
            current = self._current_frame()
        self.depth_buffer[:] = current[None, :, :]
        self.needs_policy_refresh = True

    def maybe_update(self, counter):
        if self.ros2_subscriber is not None:
            self.ros2_subscriber.spin_once()

        if counter % self.update_interval != 0 and not self.needs_policy_refresh:
            return

        if self.blind_walk_enabled and self.randomize_each_refresh and not self.blind_walk_cfg.get("image_path", ""):
            self.blind_frame = build_random_depth_frame(self.rng, self.output_height, self.output_width)

        current = self._current_frame()
        self.depth_buffer = np.roll(self.depth_buffer, shift=-1, axis=0)
        self.depth_buffer[-1] = current
        self.needs_policy_refresh = True

    def consume_refresh_flag(self):
        refresh = self.needs_policy_refresh
        self.needs_policy_refresh = False
        return refresh

    def should_use_vision(self):
        if self.blind_walk_enabled:
            return False
        if self.camera_cfg.get("image_path", ""):
            return True
        return self.ros2_subscriber is not None and self.ros2_subscriber.has_recent_frame()

    def close(self):
        if self.ros2_subscriber is not None:
            self.ros2_subscriber.close()
            self.ros2_subscriber = None


class ParkourMoEController:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.use_remote_controller = True

        self.depth_provider = RealDepthInputProvider(config.camera, config.blind_walk)
        self.policy_type = infer_policy_type(config)
        self.engine_dir = resolve_engine_dir(config)
        self.policy_device = resolve_policy_device(self.policy_type, config.device)
        if self.policy_type == "parkour_moe_engine":
            self.policy = ParkourMoETRTPolicyAdapter(
                bundle_path=config.policy_path,
                engine_dir=self.engine_dir,
                engine_precision=config.engine_precision,
                engine_paths=config.engine_paths,
                num_obs=config.num_obs,
                num_actions=config.num_actions,
                camera_cfg=config.camera,
                device=self.policy_device,
            )
        elif self.policy_type == "parkour_moe_bundle":
            self.policy = ParkourMoEPolicyAdapter(
                bundle_path=config.policy_path,
                num_obs=config.num_obs,
                num_actions=config.num_actions,
                camera_cfg=config.camera,
                device=self.policy_device,
            )
        else:
            raise ValueError(f"Unsupported parkour policy_type: {self.policy_type}")
        print(f"Loaded policy type: {self.policy_type} (device: {self.policy_device})")
        if self.policy_type == "parkour_moe_engine":
            print(f"TensorRT engine dir: {self.engine_dir}")
        self.policy.warm_up(
            self.depth_provider.depth_buffer,
            use_vision=self.depth_provider.should_use_vision(),
        )

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = config.cmd_init.copy()
        self.counter = 0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.lowcmd_publisher = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.blind_toggle_button = resolve_button_index(config.blind_walk.get("toggle_button", "F1"))
        self.last_blind_toggle_pressed = False

        self.wait_for_low_state()
        init_cmd_go2(self.low_cmd)
        print(
            f"Initial blind-walk mode: {'ON' if self.depth_provider.blind_walk_enabled else 'OFF'}"
            f" (toggle button: {config.blind_walk.get('toggle_button', 'F1')})"
        )

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def LowStateHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher.Write(cmd)

    def close(self):
        self.depth_provider.close()

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the *start* signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("Start signal received.")
        print("Press *select* button to exit.")

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        init_dof_pos = np.zeros(12, dtype=np.float32)
        for i in range(12):
            init_dof_pos[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q

        for i in range(num_step):
            alpha = i / max(num_step, 1)
            for j in range(12):
                motor_idx = self.config.joint2motor_idx[j]
                target_pos = self.config.default_angles[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1.0 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = 40.0
                self.low_cmd.motor_cmd[motor_idx].kd = 0.6
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the *Button A* signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(12):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = 40.0
                self.low_cmd.motor_cmd[motor_idx].kd = 0.6
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _maybe_toggle_blind_walk(self):
        if self.blind_toggle_button is None:
            return
        pressed = bool(self.remote_controller.button[self.blind_toggle_button])
        if pressed and not self.last_blind_toggle_pressed:
            next_state = not self.depth_provider.blind_walk_enabled
            self.depth_provider.set_blind_walk(next_state)
            print(f"Blind-walk mode {'enabled' if next_state else 'disabled'}.")
        self.last_blind_toggle_pressed = pressed

    def run(self):
        self.counter += 1
        run_start = time.perf_counter()
        for i in range(12):
            self.qj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq

        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32) * self.config.obs_scales_ang_vel
        quat = self.low_state.imu_state.quaternion
        gravity_orientation = get_gravity_orientation(quat)

        if self.use_remote_controller:
            self.cmd[0] = self.remote_controller.ly
            self.cmd[1] = -self.remote_controller.lx
            self.cmd[2] = -self.remote_controller.rx

        self._maybe_toggle_blind_walk()

        qj_obs = (self.qj.copy() - self.config.default_angles) * self.config.obs_scales_dof_pos
        dqj_obs = self.dqj.copy() * self.config.obs_scales_dof_vel

        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.command_scale
        self.obs[9:21] = qj_obs
        self.obs[21:33] = dqj_obs
        self.obs[33:45] = self.action

        self.depth_provider.maybe_update(self.counter)
        refresh_estimator = self.depth_provider.consume_refresh_flag()
        self.action = self.policy.act(
            self.obs,
            self.depth_provider.depth_buffer,
            refresh_estimator,
            use_vision=self.depth_provider.should_use_vision(),
        )

        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
        for i in range(12):
            motor_idx = self.config.joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = 40.0
            self.low_cmd.motor_cmd[motor_idx].kd = 1.0
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        self.send_cmd(self.low_cmd)
        time.sleep(max(self.config.control_dt-(time.perf_counter() - run_start),0))
        #print(time.perf_counter() - run_start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument(
        "--config",
        default=f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/go2_parkour_moe.yaml",
        help="YAML config path for real-robot parkour-MoE deployment.",
    )
    parser.add_argument("--blind-walk", action="store_true", help="Force blind-walk mode on at startup.")
    parser.add_argument("--no-blind-walk", action="store_true", help="Force blind-walk mode off at startup.")
    args = parser.parse_args()

    if args.blind_walk and args.no_blind_walk:
        raise ValueError("Choose either --blind-walk or --no-blind-walk, not both.")

    config = Config(args.config)
    if args.blind_walk:
        config.blind_walk["enabled"] = True
    if args.no_blind_walk:
        config.blind_walk["enabled"] = False

    ChannelFactoryInitialize(0, args.net)
    controller = ParkourMoEController(config)
    try:
        controller.zero_torque_state()
        controller.move_to_default_pos()
        controller.default_pos_state()

        while True:
            try:
                controller.run()
                if controller.remote_controller.button[KeyMap.select] == 1:
                    break
            except KeyboardInterrupt:
                break
    finally:
        controller.close()
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)
        print("Exit")


if __name__ == "__main__":
    main()
