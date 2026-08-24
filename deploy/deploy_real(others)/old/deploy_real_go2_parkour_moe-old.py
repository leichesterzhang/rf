from pathlib import Path

LEGGED_GYM_ROOT_DIR = "/home/unitree/3-21-MOE-SIM2REAL/"

import argparse
import sys
import time

#import imageio.v2 as imageio
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

if LEGGED_GYM_ROOT_DIR not in sys.path:
    sys.path.append(LEGGED_GYM_ROOT_DIR)

from rsl_rl.modules import ActorCriticParkourMoE, DepthAutoEncoder, ParkourEstimator

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


def to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


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


class PeriodMonitor:
    def __init__(self, name, report_interval=20):
        self.name = str(name)
        self.report_interval = max(1, int(report_interval))
        self.last_timestamp = None
        self.samples = []

    def reset(self):
        self.last_timestamp = None
        self.samples.clear()

    def tick(self, timestamp=None):
        if timestamp is None:
            timestamp = time.perf_counter()

        report = None
        if self.last_timestamp is not None:
            period = float(timestamp - self.last_timestamp)
            self.samples.append(period)
            if len(self.samples) >= self.report_interval:
                report = self._format_report()
                self.samples.clear()

        self.last_timestamp = float(timestamp)
        return report

    def record_value(self, value_s):
        self.samples.append(float(value_s))
        if len(self.samples) >= self.report_interval:
            report = self._format_report()
            self.samples.clear()
            return report
        return None

    def _format_report(self):
        arr = np.asarray(self.samples, dtype=np.float64)
        if arr.size == 0:
            return None

        mean_s = float(arr.mean())
        std_s = float(arr.std())
        min_s = float(arr.min())
        max_s = float(arr.max())
        hz = (1.0 / mean_s) if mean_s > 1.0e-9 else float("inf")
        return (
            f"[Timing:{self.name}] "
            f"mean={mean_s * 1000.0:.2f} ms, "
            f"std={std_s * 1000.0:.2f} ms, "
            f"min={min_s * 1000.0:.2f} ms, "
            f"max={max_s * 1000.0:.2f} ms, "
            f"freq={hz:.2f} Hz"
        )


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

        self.default_frame = self._build_default_frame()
        self.blind_frame = self._build_blind_frame()
        self.blind_walk_enabled = bool(self.blind_walk_cfg.get("enabled", False))
        self.has_default_depth_source = bool(self.camera_cfg.get("image_path", ""))

        initial_frame = self._current_frame()
        self.depth_buffer = np.repeat(initial_frame[None, :, :], self.buffer_len, axis=0).astype(np.float32)
        self.needs_policy_refresh = True

    def _build_default_frame(self):
        image_path = self.camera_cfg.get("image_path", "")
        if image_path:
            print(f"Real deploy image source: {image_path}")
            return load_image_as_depth_frame(image_path, self.output_height, self.output_width)
        print("Real deploy image source: using zero depth placeholder while blind-walk is disabled.")
        return np.zeros((self.output_height, self.output_width), dtype=np.float32)

    def _build_blind_frame(self):
        image_path = self.blind_walk_cfg.get("image_path", "")
        if image_path:
            print(f"Blind-walk image source: {image_path}")
            return load_image_as_depth_frame(image_path, self.output_height, self.output_width)
        print("Blind-walk image source: using a generated random image.")
        return build_random_depth_frame(self.rng, self.output_height, self.output_width)

    def _current_frame(self):
        if self.blind_walk_enabled:
            return self.blind_frame
        return self.default_frame

    def set_blind_walk(self, enabled):
        enabled = bool(enabled)
        if self.blind_walk_enabled == enabled:
            return
        self.blind_walk_enabled = enabled
        if self.blind_walk_enabled and self.randomize_each_refresh and not self.blind_walk_cfg.get("image_path", ""):
            self.blind_frame = build_random_depth_frame(self.rng, self.output_height, self.output_width)
        current = self._current_frame()
        self.depth_buffer[:] = current[None, :, :]
        self.needs_policy_refresh = True

    def maybe_update(self, counter):
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
        return (not self.blind_walk_enabled) and self.has_default_depth_source


class ParkourMoEPolicyAdapter:
    def __init__(self, config: Config):
        self.device = torch.device(config.device)
        self.num_obs = int(config.num_obs)
        self.num_actions = int(config.num_actions)

        bundle = torch.load(config.policy_path, map_location=self.device)
        actor_state_dict = bundle["actor_critic_state_dict"]
        critic_obs_dim = int(actor_state_dict["critic.0.weight"].shape[1] - 1)

        self.actor_critic = ActorCriticParkourMoE(
            self.num_obs,
            critic_obs_dim,
            self.num_actions,
            **bundle["policy_cfg"],
        ).to(self.device)
        self.estimator = ParkourEstimator(**bundle["estimator_cfg"]).to(self.device)
        self.depth_autoencoder = DepthAutoEncoder(**bundle["estimator_cfg"]).to(self.device)

        self.actor_critic.load_state_dict(actor_state_dict)
        self.estimator.load_state_dict(bundle["estimator_state_dict"])
        if "depth_autoencoder_state_dict" in bundle:
            self.depth_autoencoder.load_state_dict(bundle["depth_autoencoder_state_dict"])

        self.actor_critic.eval()
        self.estimator.eval()
        self.depth_autoencoder.eval()

        self.history_length = int(bundle["estimator_cfg"].get("history_length", 10))
        self.history = np.zeros((self.history_length, self.num_obs), dtype=np.float32)
        self.latest_mcp_code = None
        self.latest_m_hat = None
        self.latest_gating_weights = None
        self.latest_terrain_pred = None
        self.latest_estimator_output = {}
        self.enable_selector = bool(config.camera.get("enable_selector", False))
        self.selector_threshold = float(
            config.camera.get(
                "selector_threshold",
                bundle.get("depth_encoder_cfg", {}).get("selector_threshold", 1.0e-3),
            )
        )

    def reset(self):
        self.history.fill(0.0)
        self.latest_mcp_code = None
        self.latest_m_hat = None
        self.latest_gating_weights = None
        self.latest_terrain_pred = None
        self.latest_estimator_output = {}
        self.estimator.reset()

    def warm_up(self, depth_buffer, use_vision=False):
        warm_obs = np.ones(self.num_obs, dtype=np.float32)
        for _ in range(5):
            self.act(warm_obs, depth_buffer, refresh_estimator=True, use_vision=use_vision)
        self.reset()
        print("Parkour MoE network has been warmed up.")

    def _update_history(self, obs):
        self.history = np.roll(self.history, shift=-1, axis=0)
        self.history[-1] = obs

    def _append_vision_flag_to_mcp(self, mcp_code, vision_flag):
        if vision_flag.dim() == 1:
            vision_flag = vision_flag.unsqueeze(-1)
        vision_flag = vision_flag.to(device=mcp_code.device, dtype=mcp_code.dtype).detach()
        return torch.cat((mcp_code.detach(), vision_flag), dim=-1)

    def act(self, obs, depth_buffer, refresh_estimator=False, use_vision=False):
        self._update_history(obs)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if refresh_estimator or self.latest_mcp_code is None:
                proprio_hist = torch.from_numpy(self.history.reshape(1, -1)).to(self.device)
                depth_tensor = torch.from_numpy(depth_buffer).unsqueeze(0).to(self.device)
                mask = torch.full((1,), bool(use_vision), dtype=torch.bool, device=self.device)

                if self.enable_selector and bool(use_vision):
                    depth_recon, _ = self.depth_autoencoder(depth_tensor)
                    recon_error = F.mse_loss(depth_recon, depth_tensor, reduction="none").mean(dim=(1, 2, 3))
                    mask = recon_error <= self.selector_threshold

                est_out = self.estimator(
                    proprio_hist,
                    depth_tensor,
                    mask,
                    obs_now=obs_tensor,
                )
                self.latest_mcp_code = self._append_vision_flag_to_mcp(est_out["mcp_code"], mask)
                self.latest_m_hat = to_numpy(est_out["m_hat"]).squeeze().astype(np.float32)
                self.latest_gating_weights = est_out["swav_gating_weights"].detach()
                self.latest_terrain_pred = int(to_numpy(est_out["terrain_pred"]).reshape(-1)[0])
                self.latest_estimator_output = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in est_out.items()
                }

            action = self.actor_critic.act_inference(self.latest_mcp_code, obs_tensor)

        return to_numpy(action).squeeze().astype(np.float32)


class ParkourMoEController:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.use_remote_controller = True

        self.depth_provider = RealDepthInputProvider(config.camera, config.blind_walk)
        self.policy = ParkourMoEPolicyAdapter(config)
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

        self.period_report_interval = 200
        self.lowstate_monitor = PeriodMonitor("lowstate", self.period_report_interval)
        self.lowcmd_monitor = PeriodMonitor("lowcmd", self.period_report_interval)
        self.run_period_monitor = PeriodMonitor("run_period", self.period_report_interval)
        self.run_compute_monitor = PeriodMonitor("run_compute", self.period_report_interval)
        self.run_total_monitor = PeriodMonitor("run_total", self.period_report_interval)

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
        timing_report = self.lowstate_monitor.tick()
        if timing_report is not None:
            print(timing_report)
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdGo):
        timing_report = self.lowcmd_monitor.tick()
        if timing_report is not None:
            print(timing_report)
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher.Write(cmd)

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
        run_start = time.perf_counter()
        #print(run_start)
        timing_report = self.run_period_monitor.tick(run_start)
        if timing_report is not None:
            print(timing_report)

        self.counter += 1
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
        )# 133ms
       

        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
        for i in range(12):
            motor_idx = self.config.joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = 20.0
            self.low_cmd.motor_cmd[motor_idx].kd = 0.5
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        compute_report = self.run_compute_monitor.record_value(time.perf_counter() - run_start)
        if compute_report is not None:
            print(compute_report)
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt-0.003)
        total_report = self.run_total_monitor.record_value(time.perf_counter() - run_start)
        #print(time.perf_counter() - run_start)
        if total_report is not None:
            print(total_report)


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

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")


if __name__ == "__main__":
    main()
