import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import pygame
import torch
import torch.nn.functional as F
import yaml

try:
    import cv2
except ImportError:
    cv2 = None

PATH_PARENT = Path(__file__).parent
sys.path.append(str(PATH_PARENT))
from utils import MujocoRenderUtils

from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.modules import ActorCriticParkourMoE, ParkourEstimator

DEFAULT_HEIGHT_SCAN_X = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_HEIGHT_SCAN_Y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MGDP_TERRAIN_NAMES = (
    "single_gap",
    "step_stone",
    "two_row_stones",
    "one_row_stones",
    "single_bridge",
    "air_beams",
    "air_stones",
    "hurdle",
    "ramp",
    "corridor",
    "stairs_up",
    "flat",
    "rough_flat",
)


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def quat_rotate_inverse(q, v):
    q = np.array(q, np.float32)
    v = np.array(v, np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_xbox_command(joystick, max_cmd):
    pygame.event.pump()
    dead_zone = 0.1
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)
    if abs(lx) < dead_zone:
        lx = 0
    if abs(ly) < dead_zone:
        ly = 0
    if abs(rx) < dead_zone:
        rx = 0
    cmd_x = -ly * max_cmd[0]
    cmd_y = -lx * max_cmd[1]
    cmd_yaw = -rx * max_cmd[2]
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)


def clip_command(cmd, max_cmd):
    return np.clip(cmd, -max_cmd, max_cmd).astype(np.float32)


def quat_wxyz_to_yaw(quaternion):
    qw, qx, qy, qz = quaternion
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def rotate_points_yaw(local_points_xy, yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    return local_points_xy @ rot_mat.T


def build_scan_local_points(scan_cfg):
    x_points = np.array(scan_cfg.get("measured_points_x", DEFAULT_HEIGHT_SCAN_X), dtype=np.float32)
    y_points = np.array(scan_cfg.get("measured_points_y", DEFAULT_HEIGHT_SCAN_Y), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_points, y_points, indexing="ij")
    return np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1).astype(np.float32)


def reconstruct_scan_points_world(m_hat, base_pos, base_quat, scan_cfg):
    if m_hat is None:
        return None

    local_points_xy = build_scan_local_points(scan_cfg)
    pred_rel_heights = np.asarray(m_hat, dtype=np.float32).reshape(-1)
    if pred_rel_heights.shape[0] != local_points_xy.shape[0]:
        return None

    yaw = quat_wxyz_to_yaw(base_quat)
    rotated_xy = rotate_points_yaw(local_points_xy, yaw)
    world_xy = rotated_xy + np.asarray(base_pos[:2], dtype=np.float32)

    obs_scale = float(scan_cfg.get("height_measurements_scale", 2.5))
    base_height_offset = float(scan_cfg.get("base_height_offset", 0.5))
    z_offset = float(scan_cfg.get("z_offset", 0.035))
    world_z = float(base_pos[2]) - base_height_offset - pred_rel_heights / obs_scale + z_offset

    scan_points = np.zeros((local_points_xy.shape[0], 3), dtype=np.float32)
    scan_points[:, :2] = world_xy
    scan_points[:, 2] = world_z
    return scan_points


def resolve_config_path(config_ref):
    config_path = Path(config_ref)
    if config_path.is_file():
        return config_path.resolve()
    return (Path(LEGGED_GYM_ROOT_DIR) / "deploy" / "deploy_mujoco" / "configs" / config_ref).resolve()


def resolve_repo_path(path_str):
    return str(path_str).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)


def sanitize_filename(value):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))


def infer_robot_name(config, config_path, xml_path, policy_path):
    if config.get("robot_name", None):
        return str(config["robot_name"])

    xml_parts = Path(xml_path).parts
    if "robots" in xml_parts:
        robot_index = xml_parts.index("robots") + 1
        if robot_index < len(xml_parts):
            return xml_parts[robot_index]

    config_stem = Path(config_path).stem
    for suffix in ("_parkour_moe", "_mujoco", "_config"):
        if config_stem.endswith(suffix):
            return config_stem[: -len(suffix)]

    return Path(policy_path).stem


def infer_terrain_name(xml_path):
    return Path(xml_path).stem


def format_xml_vec(values):
    return " ".join(f"{float(value):.10f}" for value in values)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def quat_apply_xyzw(quat_xyzw, vec):
    quat_vec = quat_xyzw[:3]
    quat_w = quat_xyzw[3]
    uv = np.cross(quat_vec, vec)
    uuv = np.cross(quat_vec, uv)
    return vec + 2.0 * (quat_w * uv + uuv)


def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    if norm < 1.0e-8:
        return vec
    return vec / norm


def build_ros_camera_axes(camera_cfg):
    convention = str(camera_cfg.get("convention", "ros")).lower()
    if convention != "ros":
        raise ValueError(f"Unsupported camera convention: {camera_cfg.get('convention')}")

    local_euler_xyz = camera_cfg.get("local_euler_xyz", None)
    if local_euler_xyz is None:
        local_euler_zyx = camera_cfg.get("local_euler_zyx", None)
        if local_euler_zyx is None:
            raise ValueError("Camera config must provide local_euler_xyz or local_euler_zyx.")
        local_euler_xyz = [local_euler_zyx[2], local_euler_zyx[1], local_euler_zyx[0]]

    quat_ros = quat_from_euler_xyz(*[float(x) for x in local_euler_xyz])
    quat_ros[2] *= -1.0

    right_base = quat_apply_xyzw(quat_ros, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    down_base = quat_apply_xyzw(quat_ros, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    forward_base = quat_apply_xyzw(quat_ros, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    up_base = -down_base

    right_base = normalize_vec(right_base)
    forward_base = normalize_vec(forward_base)
    up_base = up_base - right_base * np.dot(up_base, right_base)
    up_base = normalize_vec(up_base)

    if np.dot(np.cross(right_base, up_base), -forward_base) < 0.0:
        up_base = -up_base

    return right_base.astype(np.float32), up_base.astype(np.float32)


def compute_camera_vertical_fov(camera_cfg):
    focal_length = camera_cfg.get("focal_length", None)
    vertical_aperture = camera_cfg.get("vertical_aperture", None)
    if focal_length is not None and vertical_aperture is not None:
        return np.degrees(2.0 * np.arctan(float(vertical_aperture) / (2.0 * float(focal_length))))

    horizontal_fov = camera_cfg.get("horizontal_fov", None)
    raw_width = float(camera_cfg.get("raw_width", camera_cfg.get("output_width", 87)))
    raw_height = float(camera_cfg.get("raw_height", camera_cfg.get("output_height", 58)))
    if horizontal_fov is not None:
        aspect = raw_width / raw_height
        return np.degrees(
            2.0 * np.arctan(np.tan(np.radians(float(horizontal_fov)) * 0.5) / max(aspect, 1.0e-6))
        )

    return 58.0


def rewrite_relative_xml_paths(root, xml_dir):
    compiler = root.find("compiler")
    meshdir = None
    texturedir = None
    assetdir = None
    if compiler is not None:
        for attr in ("meshdir", "texturedir", "assetdir"):
            value = compiler.get(attr, None)
            if value and not Path(value).is_absolute():
                compiler.set(attr, str((xml_dir / value).resolve()))
        meshdir = compiler.get("meshdir", None)
        texturedir = compiler.get("texturedir", None)
        assetdir = compiler.get("assetdir", None)

    for elem in root.findall(".//*[@file]"):
        file_path = elem.get("file", None)
        if not file_path or Path(file_path).is_absolute():
            continue

        # Keep asset file names relative when compiler directories are present.
        # MuJoCo resolves them against meshdir/texturedir/assetdir, and rewriting
        # them here would bypass that logic.
        if elem.tag == "mesh" and meshdir is not None:
            continue
        if elem.tag == "texture" and texturedir is not None:
            continue
        if elem.tag in {"hfield", "skin"} and assetdir is not None:
            continue

        elem.set("file", str((xml_dir / file_path).resolve()))


def inject_policy_camera(body_elem, camera_cfg, camera_name):
    for camera_elem in list(body_elem.findall("camera")):
        if camera_elem.get("name") == camera_name:
            body_elem.remove(camera_elem)

    right_axis, up_axis = build_ros_camera_axes(camera_cfg)
    camera_elem = ET.SubElement(body_elem, "camera")
    camera_elem.set("name", camera_name)
    camera_elem.set("pos", format_xml_vec(camera_cfg["local_pos"]))
    camera_elem.set("xyaxes", format_xml_vec(np.concatenate((right_axis, up_axis), axis=0)))
    camera_elem.set("fovy", f"{compute_camera_vertical_fov(camera_cfg):.10f}")


def prepare_model_xml(xml_path, camera_cfg, use_policy_camera):
    xml_path = Path(xml_path).resolve()
    if not use_policy_camera:
        return str(xml_path), None

    tmp_root = Path(os.environ.get("MUJOCO_TMPDIR", Path(LEGGED_GYM_ROOT_DIR) / "tmp" / "mujoco_scenes"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="go2_policy_camera_", dir=str(tmp_root)))
    camera_name = str(camera_cfg.get("camera_name", "policy_depth_camera"))
    attach_body = str(camera_cfg.get("attach_body", "base"))

    wrapper_tree = ET.parse(xml_path)
    wrapper_root = wrapper_tree.getroot()
    rewrite_relative_xml_paths(wrapper_root, xml_path.parent)

    body_elem = wrapper_root.find(f".//body[@name='{attach_body}']")
    if body_elem is not None:
        inject_policy_camera(body_elem, camera_cfg, camera_name)
    else:
        injected = False
        for include_elem in wrapper_root.findall(".//include"):
            include_file = include_elem.get("file", None)
            if include_file is None:
                continue
            include_path = Path(include_file)
            if not include_path.is_absolute():
                include_path = (xml_path.parent / include_file).resolve()
            if not include_path.is_file():
                continue

            include_tree = ET.parse(include_path)
            include_root = include_tree.getroot()
            rewrite_relative_xml_paths(include_root, include_path.parent)
            include_body = include_root.find(f".//body[@name='{attach_body}']")
            if include_body is None:
                continue

            inject_policy_camera(include_body, camera_cfg, camera_name)
            include_out_path = tmp_dir / include_path.name
            include_tree.write(include_out_path, encoding="utf-8")
            include_elem.set("file", str(include_out_path))
            injected = True
            break

        if not injected:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ValueError(f"Could not find body '{attach_body}' to attach the policy camera.")

    wrapper_out_path = tmp_dir / xml_path.name
    wrapper_tree.write(wrapper_out_path, encoding="utf-8")
    return str(wrapper_out_path), str(tmp_dir)


def depth_buffer_to_distance(depth_buffer, model):
    near = float(model.vis.map.znear) * float(model.stat.extent)
    far = float(model.vis.map.zfar) * float(model.stat.extent)
    depth_buffer = np.clip(depth_buffer, 0.0, 1.0 - 1.0e-7)
    denominator = 1.0 - depth_buffer * (1.0 - near / max(far, 1.0e-6))
    return near / np.clip(denominator, 1.0e-6, None)


def resize_depth_image(depth_image, output_height, output_width):
    if depth_image.shape == (output_height, output_width):
        return depth_image.astype(np.float32)

    depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        depth_tensor,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _as_int_pair(value, default):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        scalar = max(1, int(value))
        return scalar, scalar
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return max(1, int(value[0])), max(1, int(value[1]))
    return default


def apply_gaussian_depth_noise(depth_image, camera_cfg):
    noise_cfg = dict(camera_cfg.get("gaussian_noise", {}))
    enabled = bool(noise_cfg.get("enabled", False))
    std = float(noise_cfg.get("std", camera_cfg.get("noise_gaussian", 0.0)))
    if not enabled or std <= 0.0:
        return depth_image

    probability = float(noise_cfg.get("probability", 1.0))
    if probability <= 0.0 or np.random.random() > probability:
        return depth_image

    noisy_depth = np.array(depth_image, dtype=np.float32, copy=True)
    noisy_depth += np.random.normal(0.0, std, size=noisy_depth.shape).astype(np.float32)
    return np.clip(noisy_depth, 0.0, float(camera_cfg["clipping_range"]))


def make_irregular_reflective_mask(height, width, roughness=0.35):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = 0.5 * (width - 1) + np.random.uniform(-0.12, 0.12) * width
    cy = 0.5 * (height - 1) + np.random.uniform(-0.12, 0.12) * height
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x = (xx - cx) / max(0.5 * width, 1.0)
    y = (yy - cy) / max(0.5 * height, 1.0)
    xr = cos_a * x + sin_a * y
    yr = -sin_a * x + cos_a * y

    theta = np.arctan2(yr, xr)
    radius = np.sqrt(xr * xr + yr * yr)
    phase_1 = np.random.uniform(0.0, 2.0 * np.pi)
    phase_2 = np.random.uniform(0.0, 2.0 * np.pi)
    phase_3 = np.random.uniform(0.0, 2.0 * np.pi)
    boundary = (
        0.72
        + roughness * 0.22 * np.sin(3.0 * theta + phase_1)
        + roughness * 0.18 * np.sin(5.0 * theta + phase_2)
        + roughness * 0.12 * np.sin(8.0 * theta + phase_3)
    )
    mask = radius <= np.clip(boundary, 0.25, 1.15)

    notch_count = np.random.randint(1, 4)
    for _ in range(notch_count):
        notch_cx = np.random.uniform(0.0, width)
        notch_cy = np.random.uniform(0.0, height)
        notch_rx = np.random.uniform(0.12, 0.35) * width
        notch_ry = np.random.uniform(0.12, 0.35) * height
        notch = ((xx - notch_cx) / max(notch_rx, 1.0)) ** 2 + ((yy - notch_cy) / max(notch_ry, 1.0)) ** 2 < 1.0
        if np.random.random() < 0.5:
            mask &= ~notch
        else:
            mask |= notch

    return mask


def apply_reflective_block_noise(depth_image, camera_cfg):
    noise_cfg = dict(camera_cfg.get("reflective_noise", {}))
    if not bool(noise_cfg.get("enabled", False)):
        return depth_image

    probability = float(noise_cfg.get("probability", 1.0))
    if probability <= 0.0 or np.random.random() > probability:
        return depth_image

    noisy_depth = np.array(depth_image, dtype=np.float32, copy=True)
    height, width = noisy_depth.shape
    clipping_range = float(camera_cfg["clipping_range"])

    min_blocks = max(0, int(noise_cfg.get("min_blocks", 1)))
    max_blocks = max(min_blocks, int(noise_cfg.get("max_blocks", min_blocks)))
    if max_blocks <= 0:
        return noisy_depth
    block_count = np.random.randint(min_blocks, max_blocks + 1)

    min_h, min_w = _as_int_pair(noise_cfg.get("min_size"), (8, 8))
    max_h, max_w = _as_int_pair(noise_cfg.get("max_size"), (20, 28))
    min_h, min_w = min(min_h, height), min(min_w, width)
    max_h, max_w = min(max(max_h, min_h), height), min(max(max_w, min_w), width)
    blend_alpha = float(noise_cfg.get("blend_alpha", 1.0))
    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    mode = str(noise_cfg.get("mode", "far")).lower()
    shape = str(noise_cfg.get("shape", "irregular")).lower()
    roughness = float(noise_cfg.get("roughness", 0.35))
    value_range = noise_cfg.get("value_range", [0.0, clipping_range])

    for _ in range(block_count):
        block_h = np.random.randint(min_h, max_h + 1)
        block_w = np.random.randint(min_w, max_w + 1)
        y0 = np.random.randint(0, max(height - block_h + 1, 1))
        x0 = np.random.randint(0, max(width - block_w + 1, 1))

        if mode == "near":
            block_value = 0.0
        elif mode == "random":
            lo = float(value_range[0]) if isinstance(value_range, (list, tuple)) and len(value_range) > 0 else 0.0
            hi = float(value_range[1]) if isinstance(value_range, (list, tuple)) and len(value_range) > 1 else clipping_range
            lo, hi = sorted((max(0.0, lo), min(clipping_range, hi)))
            block_value = np.random.uniform(lo, hi)
        elif mode == "value":
            block_value = float(noise_cfg.get("value", clipping_range))
        else:
            block_value = clipping_range

        block_value = float(np.clip(block_value, 0.0, clipping_range))
        patch = noisy_depth[y0 : y0 + block_h, x0 : x0 + block_w]
        if shape == "rectangle":
            reflective_mask = np.ones((block_h, block_w), dtype=bool)
        else:
            reflective_mask = make_irregular_reflective_mask(block_h, block_w, roughness=roughness)
        patch[reflective_mask] = blend_alpha * block_value + (1.0 - blend_alpha) * patch[reflective_mask]

    return noisy_depth


def render_policy_depth_frame(renderer, model, data, camera_id, camera_cfg):
    if not hasattr(renderer, "enable_depth_rendering"):
        raise RuntimeError("This MuJoCo python binding does not support depth rendering.")

    renderer.update_scene(data, camera=camera_id)
    renderer.enable_depth_rendering()
    try:
        depth_image = renderer.render()
    finally:
        renderer.disable_depth_rendering()

    if depth_image.ndim == 3:
        depth_image = depth_image[..., 0]

    depth_format = str(camera_cfg.get("depth_format", "distance")).lower()
    if depth_format == "buffer":
        depth_image = depth_buffer_to_distance(depth_image, model)

    depth_image = np.nan_to_num(
        depth_image,
        nan=float(camera_cfg["clipping_range"]),
        posinf=float(camera_cfg["clipping_range"]),
        neginf=0.0,
    )

    if camera_cfg.get("flip_vertical", False):
        depth_image = np.flipud(depth_image)
    if camera_cfg.get("flip_horizontal", False):
        depth_image = np.fliplr(depth_image)

    top = int(camera_cfg.get("crop_top", 0))
    bottom = int(camera_cfg.get("crop_bottom", 0))
    left = int(camera_cfg.get("crop_left", 0))
    right = int(camera_cfg.get("crop_right", 0))
    h_end = depth_image.shape[0] - bottom if bottom > 0 else depth_image.shape[0]
    w_end = depth_image.shape[1] - right if right > 0 else depth_image.shape[1]
    depth_image = depth_image[top:h_end, left:w_end]

    depth_image = np.clip(depth_image, 0.0, float(camera_cfg["clipping_range"]))
    depth_image = resize_depth_image(
        depth_image.astype(np.float32),
        int(camera_cfg["output_height"]),
        int(camera_cfg["output_width"]),
    )
    depth_image = apply_gaussian_depth_noise(depth_image, camera_cfg)
    depth_image = apply_reflective_block_noise(depth_image, camera_cfg)
    return (depth_image / float(camera_cfg["clipping_range"])) - 0.5


def to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


class LegacyPolicyAdapter:
    def __init__(self, policy_path, idx_model2mj, device):
        self.device = torch.device(device)
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.idx_model2mj = idx_model2mj
        self.requires_camera = False

    def reset(self):
        pass

    def act(self, obs, _depth_buffer=None, _refresh_estimator=False):
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        result = self.policy(obs_tensor)
        extras = {}
        if isinstance(result, tuple):
            action, aux = result
            action = to_numpy(action).squeeze()[self.idx_model2mj]
            if isinstance(aux, tuple) and len(aux) == 2:
                extras["weights"] = to_numpy(aux[0]).squeeze()
                extras["latent"] = to_numpy(aux[1]).squeeze()
        else:
            action = to_numpy(result).squeeze()[self.idx_model2mj]
        return action.astype(np.float32), extras


class ParkourMoEPolicyAdapter:
    def __init__(self, bundle_path, num_obs, num_actions, camera_cfg, idx_model2mj, device):
        self.device = torch.device(device)
        self.idx_model2mj = idx_model2mj
        bundle = torch.load(bundle_path, map_location=self.device)

        actor_state_dict = bundle["actor_critic_state_dict"]
        if "critic_experts.backbone.network.0.weight" in actor_state_dict:
            critic_obs_dim = int(actor_state_dict["critic_experts.backbone.network.0.weight"].shape[1] - 1)
        else:
            critic_obs_dim = int(actor_state_dict["critic.0.weight"].shape[1] - 1)

        self.actor_critic = ActorCriticParkourMoE(
            num_obs,
            critic_obs_dim,
            num_actions,
            **bundle["policy_cfg"],
        ).to(self.device)
        self.estimator = ParkourEstimator(**bundle["estimator_cfg"]).to(self.device)

        self.actor_critic.load_state_dict(actor_state_dict)
        estimator_load_result = self.estimator.load_state_dict(
            bundle["estimator_state_dict"],
            strict=False,
        )
        missing_estimator_keys = list(estimator_load_result.missing_keys)
        unexpected_estimator_keys = list(estimator_load_result.unexpected_keys)
        allowed_unexpected_prefixes = ("core.context_patch_decoder.",)
        disallowed_unexpected_keys = [
            key for key in unexpected_estimator_keys if not key.startswith(allowed_unexpected_prefixes)
        ]
        if missing_estimator_keys or disallowed_unexpected_keys:
            raise RuntimeError(
                "Failed to load estimator state_dict cleanly. "
                f"missing={missing_estimator_keys}, unexpected={unexpected_estimator_keys}"
            )
        if unexpected_estimator_keys:
            print(
                "Estimator checkpoint contains deprecated context reconstruction weights; "
                "they will be ignored by this deployment bundle."
            )

        self.actor_critic.eval()
        self.estimator.eval()

        self.history_length = int(bundle["estimator_cfg"].get("history_length", 10))
        self.history = np.zeros((self.history_length, num_obs), dtype=np.float32)
        self.latest_mcp_code = None
        self.latest_m_hat = None
        self.latest_estimator_output = {}
        self.latest_token_mask_info = {}
        self.requires_camera = True
        timing_cfg = dict(camera_cfg.get("inference_timing", {}))
        timing_cfg.update(dict(camera_cfg.get("_inference_timing", {})))
        self.profile_inference = bool(timing_cfg.get("enabled", False))
        self.profile_print_interval = max(1, int(timing_cfg.get("print_interval", 50)))
        self.profile_cuda_synchronize = bool(timing_cfg.get("cuda_synchronize", True))
        self.profile_step_count = 0
        self.profile_estimator_count = 0
        self.profile_sums = {
            "total": 0.0,
            "actor": 0.0,
            "estimator_total": 0.0,
            "visual_encode": 0.0,
            "confidence_mask": 0.0,
            "estimator_body": 0.0,
        }
        self.manual_vision_flag = bool(
            camera_cfg.get(
                "manual_vision_flag",
                camera_cfg.get("vision_flag", camera_cfg.get("use_vision", True)),
            )
        )
        self.use_confidence_mask = bool(camera_cfg.get("use_confidence_mask", False))
        self.token_confidence_threshold = float(camera_cfg.get("token_confidence_threshold", 0.6))
        self.bad_token_ratio_threshold = float(camera_cfg.get("bad_token_ratio_threshold", 0.4))
        random_token_dropout_cfg = dict(camera_cfg.get("random_token_dropout", {}))
        self.random_token_dropout_enabled = bool(random_token_dropout_cfg.get("enabled", False))
        self.random_token_dropout_mode = str(
            random_token_dropout_cfg.get(
                "mode",
                random_token_dropout_cfg.get("token_dropout_mode", "follow_estimator"),
            )
        ).lower()
        self.random_token_dropout_min = random_token_dropout_cfg.get(
            "min",
            random_token_dropout_cfg.get("dropout_min", None),
        )
        self.random_token_dropout_max = random_token_dropout_cfg.get(
            "max",
            random_token_dropout_cfg.get("dropout_max", None),
        )
        structured_token_cfg = dict(camera_cfg.get("structured_token_protection", {}))
        self.structured_token_protection_enabled = bool(structured_token_cfg.get("enabled", False))
        self.structured_token_sobel_threshold = float(structured_token_cfg.get("sobel_threshold", 0.10))
        self.structured_token_keep_margin = float(structured_token_cfg.get("keep_margin", 0.05))
        self.structured_token_orientation_threshold = float(
            structured_token_cfg.get("orientation_consistency_threshold", 0.65)
        )
        self.structured_token_min_support_neighbors = max(
            0, int(structured_token_cfg.get("min_support_neighbors", 2))
        )
        self.structured_token_min_protected_ratio = float(
            structured_token_cfg.get("min_protected_ratio", 0.0)
        )
        self.structured_token_max_protected_ratio = float(
            structured_token_cfg.get("max_protected_ratio", 0.25)
        )
        self.structured_token_dilate = max(0, int(structured_token_cfg.get("dilate", 1)))
        token_mask_debug_cfg = dict(camera_cfg.get("token_mask_debug", {}))
        self.token_mask_debug_enabled = bool(token_mask_debug_cfg.get("enabled", False))
        self.token_mask_debug_interval = max(1, int(token_mask_debug_cfg.get("print_interval", 20)))
        self.token_mask_debug_counter = 0
        print(
            "Parkour MoE vision gating: "
            f"manual_vision_flag={self.manual_vision_flag}, "
            f"use_confidence_mask={self.use_confidence_mask}, "
            f"token_confidence_threshold={self.token_confidence_threshold:.3f}, "
            f"bad_token_ratio_threshold={self.bad_token_ratio_threshold:.3f}, "
            f"random_token_dropout={self.random_token_dropout_enabled}, "
            f"random_token_dropout_mode={self.random_token_dropout_mode}, "
            f"structured_token_protection={self.structured_token_protection_enabled}"
        )
        if self.profile_inference:
            print(
                "Inference timing enabled: "
                f"print_interval={self.profile_print_interval}, "
                f"cuda_synchronize={self.profile_cuda_synchronize}"
            )

    def reset(self):
        self.history.fill(0.0)
        self.latest_mcp_code = None
        self.latest_m_hat = None
        self.latest_estimator_output = {}
        self.latest_token_mask_info = {}
        self.estimator.reset()

    def warm_up(self, depth_buffer):
        warm_obs = np.ones(self.history.shape[1], dtype=np.float32)
        profile_inference = self.profile_inference
        self.profile_inference = False
        try:
            for _ in range(5):
                self.act(warm_obs, depth_buffer, refresh_estimator=True)
        finally:
            self.profile_inference = profile_inference
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

    def _profile_sync(self):
        if self.profile_cuda_synchronize and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _profile_start(self):
        if not self.profile_inference:
            return None
        self._profile_sync()
        return time.perf_counter()

    def _profile_elapsed_ms(self, start_time):
        if start_time is None:
            return 0.0
        self._profile_sync()
        return (time.perf_counter() - start_time) * 1000.0

    def _record_profile(self, timings, estimator_refreshed):
        if not self.profile_inference:
            return

        self.profile_step_count += 1
        self.profile_sums["total"] += timings["total"]
        self.profile_sums["actor"] += timings["actor"]
        if estimator_refreshed:
            self.profile_estimator_count += 1
            self.profile_sums["estimator_total"] += timings["estimator_total"]
            self.profile_sums["visual_encode"] += timings["visual_encode"]
            self.profile_sums["confidence_mask"] += timings["confidence_mask"]
            self.profile_sums["estimator_body"] += timings["estimator_body"]

        if self.profile_step_count % self.profile_print_interval != 0:
            return

        step_count = max(self.profile_step_count, 1)
        estimator_count = max(self.profile_estimator_count, 1)
        print(
            "\nInference timing avg "
            f"({self.profile_step_count} policy steps, {self.profile_estimator_count} estimator refreshes): "
            f"total={self.profile_sums['total'] / step_count:.3f} ms, "
            f"actor={self.profile_sums['actor'] / step_count:.3f} ms, "
            f"est_total={self.profile_sums['estimator_total'] / estimator_count:.3f} ms, "
            f"visual_encode={self.profile_sums['visual_encode'] / estimator_count:.3f} ms, "
            f"conf_mask={self.profile_sums['confidence_mask'] / estimator_count:.3f} ms, "
            f"est_body={self.profile_sums['estimator_body'] / estimator_count:.3f} ms"
        )
        self.profile_step_count = 0
        self.profile_estimator_count = 0
        for key in self.profile_sums:
            self.profile_sums[key] = 0.0

    def _build_random_token_dropout_mask(self, visual_token_data, vision_mask):
        image_height = int(visual_token_data.get("image_height", 0) or 0)
        image_width = int(visual_token_data.get("image_width", 0) or 0)
        tokens_per_frame = image_height * image_width
        frame_count = int(
            visual_token_data.get(
                "visual_frame_count",
                getattr(self.estimator.core, "depth_frame_count", 1),
            )
            or 1
        )
        dropout_mask = torch.zeros(
            vision_mask.shape[0],
            frame_count,
            tokens_per_frame,
            dtype=torch.bool,
            device=vision_mask.device,
        )
        if not self.random_token_dropout_enabled or tokens_per_frame <= 0:
            return dropout_mask

        dropout_min = self.random_token_dropout_min
        dropout_max = self.random_token_dropout_max
        if dropout_min is None:
            dropout_min = getattr(self.estimator.core, "token_dropout_min", 0.0)
        if dropout_max is None:
            dropout_max = getattr(self.estimator.core, "token_dropout_max", 0.0)
        dropout_mode = None
        if self.random_token_dropout_mode not in ("follow_estimator", "follow", "default", ""):
            dropout_mode = self.random_token_dropout_mode
        if hasattr(self.estimator.core, "build_token_dropout_mask"):
            return self.estimator.core.build_token_dropout_mask(
                vision_mask.shape[0],
                tokens_per_frame,
                vision_mask,
                vision_mask.device,
                dropout_min=dropout_min,
                dropout_max=dropout_max,
                frame_count=frame_count,
                image_height=image_height,
                image_width=image_width,
                mode=dropout_mode,
            )

        dropout_min = max(float(dropout_min), 0.0)
        dropout_max = min(max(float(dropout_max), dropout_min), 1.0)
        if dropout_max <= 0.0:
            return dropout_mask

        drop_ratio = torch.empty(vision_mask.shape[0], frame_count, device=vision_mask.device).uniform_(
            dropout_min,
            dropout_max,
        )
        drop_count = torch.round(drop_ratio * tokens_per_frame).long().clamp(min=0, max=tokens_per_frame)
        random_rank = torch.rand(
            vision_mask.shape[0],
            frame_count,
            tokens_per_frame,
            device=vision_mask.device,
        ).argsort(dim=-1).argsort(dim=-1)
        dropout_mask = random_rank < drop_count.unsqueeze(-1)
        return dropout_mask & vision_mask.view(vision_mask.shape[0], 1, 1)

    def _build_structured_token_mask(self, depth_tensor, visual_token_data):
        token_confidence = visual_token_data["token_confidence"]
        structured_mask = torch.zeros_like(token_confidence, dtype=torch.bool)
        if not self.structured_token_protection_enabled:
            return structured_mask

        image_height = int(visual_token_data.get("image_height", 0) or 0)
        image_width = int(visual_token_data.get("image_width", 0) or 0)
        if image_height <= 0 or image_width <= 0:
            return structured_mask

        depth_frames = depth_tensor.to(device=token_confidence.device, dtype=torch.float32)
        batch_size, frame_count, raw_height, raw_width = depth_frames.shape
        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            device=depth_frames.device,
            dtype=depth_frames.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            device=depth_frames.device,
            dtype=depth_frames.dtype,
        ).view(1, 1, 3, 3)
        flat_frames = depth_frames.reshape(batch_size * frame_count, 1, raw_height, raw_width)
        grad_x = F.conv2d(flat_frames, sobel_x, padding=1)
        grad_y = F.conv2d(flat_frames, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)
        grad_mag = F.adaptive_avg_pool2d(
            grad_mag,
            output_size=(image_height, image_width),
        )
        grad_x = F.adaptive_avg_pool2d(
            grad_x,
            output_size=(image_height, image_width),
        )
        grad_y = F.adaptive_avg_pool2d(
            grad_y,
            output_size=(image_height, image_width),
        )
        grad_mag = grad_mag.reshape(batch_size, frame_count, image_height * image_width)
        structured_threshold = max(self.structured_token_sobel_threshold, 0.0)
        structured_mask = grad_mag >= structured_threshold

        grad_x_2d = grad_x.reshape(batch_size * frame_count, image_height, image_width)
        grad_y_2d = grad_y.reshape(batch_size * frame_count, image_height, image_width)
        grad_mag_2d = grad_mag.reshape(batch_size * frame_count, image_height, image_width)
        candidate_mask_2d = structured_mask.reshape(batch_size * frame_count, image_height, image_width)

        if self.structured_token_orientation_threshold > 0.0:
            dir_x = grad_x_2d / (grad_mag_2d + 1.0e-6)
            dir_y = grad_y_2d / (grad_mag_2d + 1.0e-6)
            support_kernel = torch.tensor(
                [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                device=depth_frames.device,
                dtype=depth_frames.dtype,
            ).view(1, 1, 3, 3)
            dir_x_neighbors = F.conv2d(
                (dir_x * candidate_mask_2d).unsqueeze(1),
                support_kernel,
                padding=1,
            ).squeeze(1)
            dir_y_neighbors = F.conv2d(
                (dir_y * candidate_mask_2d).unsqueeze(1),
                support_kernel,
                padding=1,
            ).squeeze(1)
            neighbor_support = F.conv2d(
                candidate_mask_2d.float().unsqueeze(1),
                support_kernel,
                padding=1,
            ).squeeze(1)
            mean_dir_norm = torch.sqrt(dir_x_neighbors.square() + dir_y_neighbors.square() + 1.0e-6)
            orientation_consistency = torch.zeros_like(grad_mag_2d)
            valid_support = neighbor_support > 0.5
            orientation_consistency[valid_support] = (
                mean_dir_norm[valid_support] / neighbor_support[valid_support]
            )
            structured_mask_2d = (
                candidate_mask_2d
                & (orientation_consistency >= self.structured_token_orientation_threshold)
                & (neighbor_support >= float(self.structured_token_min_support_neighbors))
            )
        else:
            neighbor_support = F.conv2d(
                candidate_mask_2d.float().unsqueeze(1),
                torch.tensor(
                    [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                    device=depth_frames.device,
                    dtype=depth_frames.dtype,
                ).view(1, 1, 3, 3),
                padding=1,
            ).squeeze(1)
            structured_mask_2d = candidate_mask_2d & (
                neighbor_support >= float(self.structured_token_min_support_neighbors)
            )

        structured_mask = structured_mask_2d.reshape(batch_size, frame_count, image_height * image_width)

        if self.structured_token_dilate > 0:
            mask_2d = structured_mask.reshape(batch_size * frame_count, 1, image_height, image_width).float()
            kernel = 2 * self.structured_token_dilate + 1
            mask_2d = F.max_pool2d(mask_2d, kernel_size=kernel, stride=1, padding=self.structured_token_dilate)
            structured_mask = mask_2d.reshape(batch_size, frame_count, image_height * image_width) > 0.5

        flat_mask = structured_mask.reshape(batch_size * frame_count, -1)
        flat_grad = grad_mag.reshape(batch_size * frame_count, -1)
        token_count = image_height * image_width
        min_keep = 0
        if self.structured_token_min_protected_ratio > 0.0:
            min_keep = max(1, int(round(token_count * self.structured_token_min_protected_ratio)))
        max_keep = token_count
        if self.structured_token_max_protected_ratio > 0.0:
            max_keep = max(1, int(round(token_count * self.structured_token_max_protected_ratio)))
        max_keep = min(token_count, max(max_keep, min_keep))
        rank = flat_grad.argsort(dim=-1, descending=True).argsort(dim=-1)

        over_limit = flat_mask.sum(dim=-1) > max_keep
        if torch.any(over_limit):
            topk_cap_mask = rank < max_keep
            flat_mask = torch.where(
                over_limit.unsqueeze(-1),
                flat_mask & topk_cap_mask,
                flat_mask,
            )

        if min_keep > 0:
            keep_count = flat_mask.sum(dim=-1)
            need_extra = keep_count < min_keep
            if torch.any(need_extra):
                topk_floor_mask = rank < min_keep
                flat_mask = flat_mask | (topk_floor_mask & need_extra.unsqueeze(-1))

        structured_mask = flat_mask.reshape(batch_size, frame_count, token_count)

        return structured_mask

    def _build_deploy_vision_masks(self, visual_token_data, depth_tensor):
        token_confidence = visual_token_data["token_confidence"]
        manual_vision_mask = torch.full(
            (token_confidence.shape[0],),
            self.manual_vision_flag,
            dtype=torch.bool,
            device=token_confidence.device,
        )
        token_padding_mask = torch.zeros_like(token_confidence, dtype=torch.bool)
        bad_token_ratio = torch.zeros(
            token_confidence.shape[0],
            token_confidence.shape[1],
            dtype=token_confidence.dtype,
            device=token_confidence.device,
        )
        confidence_vision_mask = torch.ones_like(manual_vision_mask)
        structured_token_mask = self._build_structured_token_mask(depth_tensor, visual_token_data)
        confidence_bad_mask = torch.zeros_like(token_confidence, dtype=torch.bool)
        structured_confidence_rescue_mask = torch.zeros_like(token_confidence, dtype=torch.bool)

        if self.use_confidence_mask:
            confidence_bad_mask = token_confidence < self.token_confidence_threshold
            rescue_threshold = max(self.token_confidence_threshold - self.structured_token_keep_margin, 0.0)
            structured_confidence_rescue_mask = structured_token_mask & (token_confidence >= rescue_threshold)
            token_padding_mask = confidence_bad_mask & ~structured_confidence_rescue_mask
            bad_token_ratio = token_padding_mask.to(token_confidence.dtype).mean(dim=-1)
            confidence_vision_mask = ~(bad_token_ratio > self.bad_token_ratio_threshold).any(dim=-1)

        vision_mask = manual_vision_mask & confidence_vision_mask
        random_token_dropout_mask = self._build_random_token_dropout_mask(visual_token_data, vision_mask)
        random_token_dropout_mask = random_token_dropout_mask & ~structured_token_mask
        token_padding_mask = token_padding_mask | random_token_dropout_mask
        flat_confidence = token_confidence.reshape(token_confidence.shape[0], -1)
        flat_token_padding_mask = token_padding_mask.reshape(token_padding_mask.shape[0], -1)
        flat_random_token_dropout_mask = random_token_dropout_mask.reshape(random_token_dropout_mask.shape[0], -1)
        flat_structured_token_mask = structured_token_mask.reshape(structured_token_mask.shape[0], -1)
        flat_confidence_bad_mask = confidence_bad_mask.reshape(confidence_bad_mask.shape[0], -1)
        return vision_mask, token_padding_mask, {
            "manual_vision_flag": manual_vision_mask,
            "confidence_vision_flag": confidence_vision_mask,
            "deploy_vision_flag": vision_mask,
            "deploy_token_padding_mask": token_padding_mask,
            "deploy_random_token_dropout_mask": random_token_dropout_mask,
            "deploy_random_token_dropout_count": flat_random_token_dropout_mask.sum(dim=-1),
            "deploy_bad_token_count": flat_token_padding_mask.sum(dim=-1),
            "deploy_structured_token_mask": structured_token_mask,
            "deploy_structured_token_count": flat_structured_token_mask.sum(dim=-1),
            "deploy_structured_confidence_rescue_count": structured_confidence_rescue_mask.reshape(
                structured_confidence_rescue_mask.shape[0],
                -1,
            ).sum(dim=-1),
            "deploy_confidence_bad_token_count": flat_confidence_bad_mask.sum(dim=-1),
            "deploy_total_token_count": torch.full(
                (token_confidence.shape[0],),
                flat_token_padding_mask.shape[-1],
                dtype=torch.long,
                device=token_confidence.device,
            ),
            "deploy_bad_token_ratio": bad_token_ratio,
            "deploy_token_confidence_min": flat_confidence.min(dim=-1).values,
            "deploy_token_confidence_mean": flat_confidence.mean(dim=-1),
            "deploy_token_confidence": token_confidence,
            "image_height": int(visual_token_data.get("image_height", 0)),
            "image_width": int(visual_token_data.get("image_width", 0)),
        }

    def _maybe_print_token_mask_debug(self, deploy_mask_info):
        if not self.token_mask_debug_enabled:
            return
        self.token_mask_debug_counter += 1
        if self.token_mask_debug_counter % self.token_mask_debug_interval != 0:
            return

        bad_count = int(to_numpy(deploy_mask_info["deploy_bad_token_count"]).reshape(-1)[0])
        random_drop_count = int(to_numpy(deploy_mask_info["deploy_random_token_dropout_count"]).reshape(-1)[0])
        structured_count = int(to_numpy(deploy_mask_info["deploy_structured_token_count"]).reshape(-1)[0])
        structured_rescue_count = int(
            to_numpy(deploy_mask_info["deploy_structured_confidence_rescue_count"]).reshape(-1)[0]
        )
        confidence_bad_count = int(to_numpy(deploy_mask_info["deploy_confidence_bad_token_count"]).reshape(-1)[0])
        total_count = int(to_numpy(deploy_mask_info["deploy_total_token_count"]).reshape(-1)[0])
        bad_ratio = to_numpy(deploy_mask_info["deploy_bad_token_ratio"]).reshape(-1)
        conf_min = float(to_numpy(deploy_mask_info["deploy_token_confidence_min"]).reshape(-1)[0])
        conf_mean = float(to_numpy(deploy_mask_info["deploy_token_confidence_mean"]).reshape(-1)[0])
        manual_flag = bool(to_numpy(deploy_mask_info["manual_vision_flag"]).reshape(-1)[0])
        confidence_flag = bool(to_numpy(deploy_mask_info["confidence_vision_flag"]).reshape(-1)[0])
        vision_flag = bool(to_numpy(deploy_mask_info["deploy_vision_flag"]).reshape(-1)[0])
        ratio_text = ",".join(f"{float(value):.3f}" for value in bad_ratio)
        print(
            "\nToken mask debug: "
            f"bad_tokens={bad_count}/{total_count}, "
            f"conf_bad={confidence_bad_count}/{total_count}, "
            f"random_drop={random_drop_count}/{total_count}, "
            f"structured_keep={structured_count}/{total_count}, "
            f"structured_rescue={structured_rescue_count}/{total_count}, "
            f"bad_ratio_per_frame=[{ratio_text}], "
            f"conf_min={conf_min:.3f}, conf_mean={conf_mean:.3f}, "
            f"manual_flag={manual_flag}, confidence_flag={confidence_flag}, "
            f"vision_flag={vision_flag}"
        )

    def act(self, obs, depth_buffer, refresh_estimator=False):
        if depth_buffer is None:
            raise ValueError("Parkour MoE policy requires a depth camera buffer.")

        total_start = self._profile_start()
        timings = {
            "total": 0.0,
            "actor": 0.0,
            "estimator_total": 0.0,
            "visual_encode": 0.0,
            "confidence_mask": 0.0,
            "estimator_body": 0.0,
        }
        estimator_refreshed = False

        self._update_history(obs)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if refresh_estimator or self.latest_mcp_code is None:
                estimator_refreshed = True
                proprio_hist = torch.from_numpy(self.history.reshape(1, -1)).to(self.device)
                depth_tensor = torch.from_numpy(depth_buffer).unsqueeze(0).to(self.device)
                #depth_tensor = torch.rand_like(depth_tensor)
                timing_start = self._profile_start()
                visual_token_data = self.estimator.encode_visual_tokens(depth_tensor)
                timings["visual_encode"] = self._profile_elapsed_ms(timing_start)

                timing_start = self._profile_start()
                mask, token_padding_mask, deploy_mask_info = self._build_deploy_vision_masks(
                    visual_token_data,
                    depth_tensor,
                )
                timings["confidence_mask"] = self._profile_elapsed_ms(timing_start)

                timing_start = self._profile_start()
                est_out = self.estimator.forward_from_visual_tokens(
                    proprio_hist,
                    visual_token_data,
                    mask,
                    token_padding_mask=token_padding_mask,
                    obs_now=obs_tensor,
                    apply_training_dropout=False,
                )
                timings["estimator_body"] = self._profile_elapsed_ms(timing_start)
                timings["estimator_total"] = (
                    timings["visual_encode"]
                    + timings["confidence_mask"]
                    + timings["estimator_body"]
                )
                self._maybe_print_token_mask_debug(deploy_mask_info)
                est_out.update(deploy_mask_info)
                #print(est_out["mcp_code"])
                self.latest_mcp_code = self._append_vision_flag_to_mcp(est_out["mcp_code"], mask)
                self.latest_m_hat = to_numpy(est_out["m_hat"]).squeeze().astype(np.float32)
                self.latest_estimator_output = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in est_out.items()
                }
                self.latest_token_mask_info = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in deploy_mask_info.items()
                }
                for active_key in ("active_token_coords", "active_token_valid_mass", "active_token_enabled"):
                    active_value = est_out.get(active_key, None)
                    self.latest_token_mask_info[active_key] = (
                        active_value.detach() if torch.is_tensor(active_value) else active_value
                    )

            timing_start = self._profile_start()
            action = self.actor_critic.act_inference(self.latest_mcp_code, obs_tensor)
            timings["actor"] = self._profile_elapsed_ms(timing_start)

        timings["total"] = self._profile_elapsed_ms(total_start)
        self._record_profile(timings, estimator_refreshed)
        extras = {
            "weights": to_numpy(self.latest_estimator_output.get("swav_gating_weights", None)).squeeze()
            if "swav_gating_weights" in self.latest_estimator_output
            else None,
            "latent": to_numpy(self.latest_estimator_output.get("mcp_code", None)).squeeze()
            if "mcp_code" in self.latest_estimator_output
            else None,
            "terrain_pred": to_numpy(self.latest_estimator_output.get("terrain_pred", None)).squeeze()
            if "terrain_pred" in self.latest_estimator_output
            else None,
            "terrain_probs": to_numpy(self.latest_estimator_output.get("terrain_probs", None)).squeeze()
            if "terrain_probs" in self.latest_estimator_output
            else None,
        }
        action_np = to_numpy(action).squeeze()[self.idx_model2mj].astype(np.float32)
        return action_np, extras


class PolicyDepthCamera:
    def __init__(self, model, camera_name, camera_cfg):
        self.enabled = bool(camera_cfg.get("enabled", False))
        self.camera_cfg = camera_cfg
        self.update_interval = max(1, int(camera_cfg.get("update_interval", 1)))
        self.buffer_len = max(1, int(camera_cfg.get("buffer_len", 1)))
        self.output_height = int(camera_cfg.get("output_height", 1))
        self.output_width = int(camera_cfg.get("output_width", 1))
        self.depth_buffer = np.zeros((self.buffer_len, self.output_height, self.output_width), dtype=np.float32)
        self.initialized = False
        self.needs_policy_refresh = False
        self.renderer = None
        self.camera_id = None

        if not self.enabled:
            return

        raw_height = int(camera_cfg["raw_height"])
        raw_width = int(camera_cfg["raw_width"])
        self.renderer = mujoco.Renderer(model, height=raw_height, width=raw_width)
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            raise ValueError(f"Could not find MuJoCo camera '{camera_name}'.")

    def maybe_update(self, counter, model, data):
        if not self.enabled:
            return False

        periodic_update = counter % self.update_interval == 0 or not self.initialized
        if not periodic_update:
            return False

        frame = render_policy_depth_frame(
            self.renderer,
            model,
            data,
            self.camera_id,
            self.camera_cfg,
        )
        if not self.initialized:
            self.depth_buffer[:] = frame[np.newaxis, :, :]
            self.initialized = True
        else:
            self.depth_buffer = np.roll(self.depth_buffer, shift=-1, axis=0)
            self.depth_buffer[-1] = frame
        self.needs_policy_refresh = True
        return True

    def consume_refresh_flag(self):
        refresh = self.needs_policy_refresh or not self.initialized
        self.needs_policy_refresh = False
        return refresh


def build_observation(
    data,
    default_angles,
    dof_pos_scale,
    dof_vel_scale,
    ang_vel_scale,
    cmd,
    cmd_scale,
    action,
    idx_mj2model,
    num_obs,
):
    qj = data.qpos[7:].copy()
    dqj = data.qvel[6:].copy()
    quat = data.qpos[3:7]
    ang_vel = data.qvel[3:6]

    qj = (qj - default_angles) * dof_pos_scale
    dqj = dqj * dof_vel_scale
    gravity_orientation = get_gravity_orientation(quat)
    ang_vel = ang_vel * ang_vel_scale

    obs = np.zeros(num_obs, dtype=np.float32)
    obs[:3] = ang_vel
    obs[3:6] = gravity_orientation
    obs[6:9] = cmd * cmd_scale
    obs[9:21] = qj[idx_mj2model]
    obs[21:33] = dqj[idx_mj2model]
    obs[33:45] = action[idx_mj2model]
    return obs


class MujocoStateRecorder:
    def __init__(
        self,
        save_dir,
        robot_name,
        terrain_name,
        joint_names,
        model,
        config_path,
        xml_path,
        simulation_dt,
        policy_path=None,
    ):
        self.save_dir = Path(save_dir)
        self.robot_name = sanitize_filename(robot_name)
        self.terrain_name = sanitize_filename(terrain_name)
        self.joint_names = list(joint_names)
        self.config_path = str(config_path)
        self.xml_path = str(xml_path)
        self.simulation_dt = float(simulation_dt)
        self.policy_path = "" if policy_path is None else str(policy_path)
        self.records = []
        self.body_ids = self._resolve_joint_body_ids(model)
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.base_body_id < 0:
            self.base_body_id = 1 if model.nbody > 1 else 0

    def _resolve_joint_body_ids(self, model):
        body_ids = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Could not find MuJoCo joint '{joint_name}'.")
            body_ids.append(int(model.jnt_bodyid[joint_id]))
        return np.array(body_ids, dtype=np.int32)

    def record(self, model, data, step_index, tau, action=None, target_dof_pos=None, cmd=None):
        joint_positions = np.asarray(data.qpos[7:], dtype=np.float32).copy()
        policy_actions = (
            np.zeros_like(joint_positions)
            if action is None
            else np.asarray(action, dtype=np.float32).reshape(-1).copy()
        )
        target_dof_positions = (
            joint_positions.copy()
            if target_dof_pos is None
            else np.asarray(target_dof_pos, dtype=np.float32).reshape(-1).copy()
        )
        command = (
            np.zeros(3, dtype=np.float32)
            if cmd is None
            else np.asarray(cmd, dtype=np.float32).reshape(-1).copy()
        )
        body_xvelp = np.asarray(data.cvel[self.body_ids, 3:6], dtype=np.float32).copy()
        body_xvelr = np.asarray(data.cvel[self.body_ids, 0:3], dtype=np.float32).copy()
        body_xaccp = np.asarray(data.cacc[self.body_ids, 3:6], dtype=np.float32).copy()
        body_xaccr = np.asarray(data.cacc[self.body_ids, 0:3], dtype=np.float32).copy()
        joint_dof_velocities = np.asarray(data.qvel[6:], dtype=np.float32).copy()
        joint_torques = np.asarray(data.qfrc_actuator[6:], dtype=np.float32).copy()
        joint_instantaneous_power = joint_torques * joint_dof_velocities
        mujoco.mj_subtreeVel(model, data)

        self.records.append(
            {
                "time": float(data.time),
                "step": int(step_index),
                "joint_positions": joint_positions,
                "joint_dof_velocities": joint_dof_velocities,
                "joint_dof_accelerations": np.asarray(data.qacc[6:], dtype=np.float32).copy(),
                "joint_linear_velocities": body_xvelp,
                "joint_angular_velocities": body_xvelr,
                "joint_linear_accelerations": body_xaccp,
                "joint_angular_accelerations": body_xaccr,
                "joint_torques": joint_torques,
                "joint_torque_commands": np.asarray(tau, dtype=np.float32).copy(),
                "joint_instantaneous_power": joint_instantaneous_power,
                "total_instantaneous_power": np.array(
                    [
                        np.sum(joint_instantaneous_power),
                        np.sum(np.maximum(joint_instantaneous_power, 0.0)),
                        np.sum(np.abs(joint_instantaneous_power)),
                    ],
                    dtype=np.float32,
                ),
                "policy_actions": policy_actions,
                "target_dof_positions": target_dof_positions,
                "joint_position_errors": target_dof_positions - joint_positions,
                "command": command,
                "base_position": np.asarray(data.qpos[:3], dtype=np.float32).copy(),
                "base_quaternion_wxyz": np.asarray(data.qpos[3:7], dtype=np.float32).copy(),
                "base_linear_velocity": np.asarray(data.qvel[:3], dtype=np.float32).copy(),
                "base_angular_velocity": np.asarray(data.qvel[3:6], dtype=np.float32).copy(),
                "base_linear_acceleration": np.asarray(data.qacc[:3], dtype=np.float32).copy(),
                "base_angular_acceleration": np.asarray(data.qacc[3:6], dtype=np.float32).copy(),
                "base_com_position": np.asarray(data.xipos[self.base_body_id], dtype=np.float32).copy(),
                "base_com_velocity": np.asarray(data.cvel[self.base_body_id, 3:6], dtype=np.float32).copy(),
                "system_com_position": np.asarray(data.subtree_com[0], dtype=np.float32).copy(),
                "system_com_velocity": np.asarray(data.subtree_linvel[0], dtype=np.float32).copy(),
            }
        )

    def save(self):
        if not self.records:
            return None

        output_dir = self.save_dir / self.robot_name
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{self.terrain_name}_{timestamp}.npz"

        arrays = {
            key: np.stack([record[key] for record in self.records], axis=0)
            for key in self.records[0]
        }
        arrays.update(
            {
                "joint_names": np.array(self.joint_names),
                "robot_name": np.array(self.robot_name),
                "terrain_name": np.array(self.terrain_name),
                "config_path": np.array(self.config_path),
                "xml_path": np.array(self.xml_path),
                "simulation_dt": np.array(self.simulation_dt, dtype=np.float32),
                "policy_path": np.array(self.policy_path),
            }
        )
        np.savez_compressed(output_path, **arrays)
        return output_path


def draw_moe_weights(screen, weights, width, height):
    screen.fill((255, 255, 255))
    if weights is None:
        pygame.display.flip()
        return

    num_experts = len(weights)
    if num_experts == 0:
        pygame.display.flip()
        return

    margin = 5
    bar_width = (width - 2 * margin) / num_experts
    max_bar_height = height - 2 * margin

    for i, weight in enumerate(weights):
        weight = float(np.clip(weight, 0.0, 1.0))
        bar_height = int(weight * max_bar_height)
        x_pos = margin + i * bar_width
        y_pos = height - margin - bar_height
        pygame.draw.rect(screen, (50, 100, 255), (x_pos, y_pos, bar_width - 2, bar_height))

    pygame.display.flip()


def build_depth_preview_surface(depth_frame, preview_size):
    if depth_frame is None:
        return None

    depth_vis = np.clip(0.5 - depth_frame, 0.0, 1.0)
    depth_vis = (depth_vis * 255.0).astype(np.uint8)
    depth_rgb = np.repeat(depth_vis[:, :, None], 3, axis=2)
    surface = pygame.surfarray.make_surface(np.transpose(depth_rgb, (1, 0, 2)))
    if surface.get_size() != preview_size:
        surface = pygame.transform.smoothscale(surface, preview_size)
    return surface


def build_depth_preview_rgb(depth_frame, preview_size):
    if depth_frame is None:
        return None

    depth_vis = np.clip(0.5 - depth_frame, 0.0, 1.0)
    depth_vis = depth_vis.astype(np.float32, copy=False)
    if depth_vis.shape != (preview_size[1], preview_size[0]):
        if cv2 is not None:
            depth_vis = cv2.resize(
                depth_vis,
                (preview_size[0], preview_size[1]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            depth_vis = resize_depth_image(depth_vis, preview_size[1], preview_size[0])
    depth_vis = (depth_vis * 255.0).astype(np.uint8)
    return np.repeat(depth_vis[:, :, None], 3, axis=2)


def terrain_label_from_info(terrain_info):
    if not terrain_info:
        return None

    terrain_pred = terrain_info.get("terrain_pred", None)
    terrain_probs = terrain_info.get("terrain_probs", None)
    if terrain_pred is None and terrain_probs is None:
        return None

    confidence = None
    if terrain_probs is not None:
        terrain_probs = np.asarray(terrain_probs, dtype=np.float32).reshape(-1)
        if terrain_probs.size > 0:
            terrain_id = int(np.argmax(terrain_probs))
            confidence = float(terrain_probs[terrain_id])
        else:
            terrain_id = None
    else:
        terrain_id = None

    if terrain_id is None and terrain_pred is not None:
        terrain_array = np.asarray(terrain_pred).reshape(-1)
        if terrain_array.size == 0:
            return None
        terrain_id = int(terrain_array[0])

    if terrain_id is None:
        return None

    if 0 <= terrain_id < len(MGDP_TERRAIN_NAMES):
        terrain_name = MGDP_TERRAIN_NAMES[terrain_id]
    else:
        terrain_name = "unknown"

    if confidence is None:
        return f"terrain: {terrain_id} {terrain_name}"
    return f"terrain: {terrain_id} {terrain_name} ({confidence:.2f})"


def overlay_token_mask_on_bgr(frame_bgr, token_mask_info, overlay_cfg):
    if not overlay_cfg.get("enabled", False) or not token_mask_info or cv2 is None:
        return frame_bgr

    show_padding_mask = bool(overlay_cfg.get("show_padding_mask", True))
    show_random_dropout = bool(overlay_cfg.get("show_random_dropout", True))
    show_structured_protection = bool(overlay_cfg.get("show_structured_protection", True))

    token_mask = token_mask_info.get("deploy_token_padding_mask", None) if show_padding_mask else None
    random_token_mask = token_mask_info.get("deploy_random_token_dropout_mask", None) if show_random_dropout else None
    structured_token_mask = (
        token_mask_info.get("deploy_structured_token_mask", None) if show_structured_protection else None
    )
    if token_mask is None and random_token_mask is None and structured_token_mask is None:
        return frame_bgr

    def normalize_token_mask(mask_value):
        if mask_value is None:
            return None
        mask_value = to_numpy(mask_value).astype(bool)
        if mask_value.ndim == 3:
            return mask_value[0, -1]
        if mask_value.ndim == 2:
            return mask_value[0]
        return None

    token_mask = normalize_token_mask(token_mask)
    random_token_mask = normalize_token_mask(random_token_mask)
    structured_token_mask = normalize_token_mask(structured_token_mask)
    if token_mask is None:
        reference_mask = random_token_mask if random_token_mask is not None else structured_token_mask
        token_mask = np.zeros_like(reference_mask, dtype=bool)
    if random_token_mask is None:
        random_token_mask = np.zeros_like(token_mask, dtype=bool)
    if structured_token_mask is None:
        structured_token_mask = np.zeros_like(token_mask, dtype=bool)
    combined_token_mask = token_mask | random_token_mask
    combined_visible_mask = combined_token_mask | structured_token_mask

    token_height = int(token_mask_info.get("image_height", 0) or 0)
    token_width = int(token_mask_info.get("image_width", 0) or 0)
    if token_height <= 0 or token_width <= 0:
        if combined_visible_mask.size == 88:
            token_height, token_width = 8, 11
        else:
            token_height = 1
            token_width = int(combined_visible_mask.size)
    if combined_visible_mask.size != token_height * token_width:
        return frame_bgr

    token_mask = token_mask.reshape(token_height, token_width)
    random_token_mask = random_token_mask.reshape(token_height, token_width)
    structured_token_mask = structured_token_mask.reshape(token_height, token_width)
    combined_token_mask = combined_token_mask.reshape(token_height, token_width)
    combined_visible_mask = combined_visible_mask.reshape(token_height, token_width)
    output = frame_bgr.copy()
    overlay = output.copy()
    display_height, display_width = output.shape[:2]
    alpha = float(np.clip(overlay_cfg.get("alpha", 0.45), 0.0, 1.0))
    show_grid = bool(overlay_cfg.get("show_grid", True))
    show_confidence = bool(overlay_cfg.get("show_confidence", False))
    bad_color = tuple(int(value) for value in overlay_cfg.get("bad_color_bgr", [40, 40, 255]))
    random_color = tuple(int(value) for value in overlay_cfg.get("random_dropout_color_bgr", [0, 160, 255]))
    structured_color = tuple(int(value) for value in overlay_cfg.get("structured_protection_color_bgr", [0, 220, 0]))
    grid_color = tuple(int(value) for value in overlay_cfg.get("grid_color_bgr", [90, 90, 90]))

    confidence_grid = None
    if show_confidence:
        token_confidence = token_mask_info.get("deploy_token_confidence", None)
        if token_confidence is not None:
            token_confidence = to_numpy(token_confidence)
            if token_confidence.ndim == 3:
                token_confidence = token_confidence[0, -1]
            elif token_confidence.ndim == 2:
                token_confidence = token_confidence[0]
            if token_confidence.size == token_height * token_width:
                confidence_grid = token_confidence.reshape(token_height, token_width)

    for row in range(token_height):
        y0 = int(round(row * display_height / token_height))
        y1 = int(round((row + 1) * display_height / token_height))
        for col in range(token_width):
            if not combined_visible_mask[row, col]:
                continue
            x0 = int(round(col * display_width / token_width))
            x1 = int(round((col + 1) * display_width / token_width))
            if structured_token_mask[row, col] and not combined_token_mask[row, col]:
                color = structured_color
            else:
                color = random_color if random_token_mask[row, col] else bad_color
            cv2.rectangle(
                overlay,
                (x0, y0),
                (max(x1 - 1, x0), max(y1 - 1, y0)),
                color,
                thickness=-1,
            )

    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)
    for row in range(token_height):
        y0 = int(round(row * display_height / token_height))
        y1 = int(round((row + 1) * display_height / token_height))
        for col in range(token_width):
            if not combined_visible_mask[row, col]:
                continue
            x0 = int(round(col * display_width / token_width))
            x1 = int(round((col + 1) * display_width / token_width))
            if structured_token_mask[row, col] and not combined_token_mask[row, col]:
                outline_color = (0, 180, 0)
            else:
                outline_color = (0, 120, 255) if random_token_mask[row, col] else (0, 0, 255)
            cv2.rectangle(output, (x0, y0), (max(x1 - 1, x0), max(y1 - 1, y0)), outline_color, thickness=1)
            if confidence_grid is not None and x1 - x0 >= 22 and y1 - y0 >= 14:
                cv2.putText(
                    output,
                    f"{confidence_grid[row, col]:.2f}",
                    (x0 + 2, min(y1 - 3, y0 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    if show_grid:
        for row in range(1, token_height):
            y = int(round(row * display_height / token_height))
            cv2.line(output, (0, y), (display_width - 1, y), grid_color, 1)
        for col in range(1, token_width):
            x = int(round(col * display_width / token_width))
            cv2.line(output, (x, 0), (x, display_height - 1), grid_color, 1)

    masked_count = int(combined_token_mask.sum())
    random_count = int(random_token_mask.sum())
    if masked_count > 0:
        cv2.putText(
            output,
            f"masked tokens: {masked_count}/{combined_token_mask.size}",
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 255),
            1,
            cv2.LINE_AA,
        )
    if random_count > 0:
        cv2.putText(
            output,
            f"random drop: {random_count}/{combined_token_mask.size}",
            (8, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 160, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def overlay_active_tokens_on_bgr(frame_bgr, token_mask_info, overlay_cfg):
    if not overlay_cfg.get("enabled", False) or not token_mask_info or cv2 is None:
        return frame_bgr

    active_enabled = token_mask_info.get("active_token_enabled", True)
    if torch.is_tensor(active_enabled):
        active_enabled = bool(to_numpy(active_enabled).reshape(-1)[0])
    if not bool(active_enabled):
        return frame_bgr

    active_coords = token_mask_info.get("active_token_coords", None)
    if active_coords is None:
        return frame_bgr

    active_coords = to_numpy(active_coords).astype(np.float32)
    if active_coords.ndim == 4:
        active_coords = active_coords[0, -1]
    elif active_coords.ndim == 3:
        active_coords = active_coords[0]
    if active_coords.ndim != 2 or active_coords.shape[-1] != 2:
        return frame_bgr

    active_valid_mass = token_mask_info.get("active_token_valid_mass", None)
    if active_valid_mass is not None:
        active_valid_mass = to_numpy(active_valid_mass).astype(np.float32)
        if active_valid_mass.ndim == 3:
            active_valid_mass = active_valid_mass[0, -1]
        elif active_valid_mass.ndim == 2:
            active_valid_mass = active_valid_mass[0]
        if active_valid_mass.ndim != 1:
            active_valid_mass = None

    output = frame_bgr.copy()
    display_height, display_width = output.shape[:2]
    radius = max(1, int(overlay_cfg.get("radius", 4)))
    min_valid_mass = float(overlay_cfg.get("min_valid_mass", 0.05))
    valid_color = tuple(int(value) for value in overlay_cfg.get("valid_color_bgr", [255, 240, 40]))
    invalid_color = tuple(int(value) for value in overlay_cfg.get("invalid_color_bgr", [150, 150, 150]))
    outline_color = tuple(int(value) for value in overlay_cfg.get("outline_color_bgr", [25, 25, 25]))
    text_color = tuple(int(value) for value in overlay_cfg.get("text_color_bgr", [255, 255, 160]))
    show_count = bool(overlay_cfg.get("show_count", True))

    valid_count = 0
    for token_idx, coord in enumerate(active_coords):
        x_norm = float(np.clip(coord[0], -1.0, 1.0))
        y_norm = float(np.clip(coord[1], -1.0, 1.0))
        x = int(round((x_norm + 1.0) * 0.5 * (display_width - 1)))
        y = int(round((y_norm + 1.0) * 0.5 * (display_height - 1)))
        valid = True
        if active_valid_mass is not None and token_idx < active_valid_mass.shape[0]:
            valid = float(active_valid_mass[token_idx]) > min_valid_mass
        valid_count += int(valid)
        color = valid_color if valid else invalid_color
        cv2.circle(output, (x, y), radius + 1, outline_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(output, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)

    if show_count:
        cv2.putText(
            output,
            f"probes: {valid_count}/{active_coords.shape[0]}",
            (8, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return output


class OpenCVDepthPreview:
    def __init__(self, preview_size, enabled=True, token_overlay_cfg=None, active_token_overlay_cfg=None):
        self.enabled = bool(enabled)
        self.preview_size = tuple(int(value) for value in preview_size)
        self.token_overlay_cfg = dict(token_overlay_cfg or {})
        self.active_token_overlay_cfg = dict(active_token_overlay_cfg or {})
        self.display_size = (
            max(self.preview_size[0], 240),
            max(self.preview_size[1], 160),
        )
        self.window_name = "Depth Preview"
        self.window_created = False
        self.closed = False

        if not self.enabled:
            return

        if cv2 is None:
            raise RuntimeError(
                "depth_preview.enabled=True but OpenCV (cv2) is not installed. "
                "Please install an OpenCV package with GUI support."
            )

        self._ensure_window()
        waiting_image = self._build_waiting_image()
        cv2.imshow(self.window_name, waiting_image)
        cv2.waitKey(1)
        print(
            "Depth preview window created with OpenCV: "
            f"source={self.preview_size}, display={self.display_size}"
        )

    def _ensure_window(self):
        if self.window_created or cv2 is None:
            return
        if hasattr(cv2, "startWindowThread"):
            cv2.startWindowThread()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_size[0], self.display_size[1])
        self.window_created = True

    def _window_is_visible(self):
        if not self.window_created or cv2 is None:
            return False
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False

    def _build_waiting_image(self):
        preview_width, preview_height = self.display_size
        image = np.full((preview_height, preview_width, 3), 235, dtype=np.uint8)
        if cv2 is not None:
            cv2.putText(
                image,
                "Waiting for depth frame...",
                (12, max(preview_height // 2, 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
        return image

    def close(self):
        if cv2 is None:
            self.closed = True
            return

        self.closed = True
        if self.window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self.window_created = False

    def update(self, depth_frame, token_mask_info=None, terrain_info=None):
        if not self.enabled or self.closed or cv2 is None:
            return -1

        try:
            self._ensure_window()
            if not self._window_is_visible():
                self.close()
                return -1

            if depth_frame is None:
                frame_bgr = self._build_waiting_image()
            else:
                frame_rgb = build_depth_preview_rgb(depth_frame, self.preview_size)
                frame_rgb = cv2.resize(
                    frame_rgb,
                    self.display_size,
                    interpolation=cv2.INTER_NEAREST,
                )
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_bgr = overlay_token_mask_on_bgr(
                    frame_bgr,
                    token_mask_info,
                    self.token_overlay_cfg,
                )
                frame_bgr = overlay_active_tokens_on_bgr(
                    frame_bgr,
                    token_mask_info,
                    self.active_token_overlay_cfg,
                )
                cv2.putText(
                    frame_bgr,
                    f"min={float(depth_frame.min()):+.3f} max={float(depth_frame.max()):+.3f}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (40, 220, 40),
                    1,
                    cv2.LINE_AA,
                )
                terrain_label = terrain_label_from_info(terrain_info)
                if terrain_label is not None:
                    cv2.putText(
                        frame_bgr,
                        terrain_label,
                        (8, 104),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 220, 40),
                        1,
                        cv2.LINE_AA,
                    )

            cv2.imshow(self.window_name, frame_bgr)
            return int(cv2.waitKey(1) & 0xFF)
        except Exception as exc:
            self.closed = True
            print(f"Warning: OpenCV depth preview update failed: {exc}")
            self.close()
            return -1


def infer_policy_type(policy_path, config):
    explicit_type = config.get("policy_type", "auto")
    if explicit_type != "auto":
        return explicit_type

    policy_name = Path(policy_path).name
    if "inference_bundle" in policy_name or "parkour_moe" in policy_name:
        return "parkour_moe_bundle"
    return "jit"


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="go2_parkour_moe.yaml", help="YAML config filename or absolute path.")
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument(
        "--visualize-moe-weights",
        action="store_true",
        help="Whether to visualize mixture of experts weights.",
    )
    parser.add_argument(
        "--save-moe-latent",
        action="store_true",
        help="Whether to save mixture of experts latent vectors.",
    )
    parser.add_argument(
        "--no-record-state",
        action="store_true",
        help="Disable MuJoCo state recording.",
    )
    parser.add_argument(
        "--action-data-dir",
        default=str(Path(LEGGED_GYM_ROOT_DIR) / "action_data"),
        help="Directory used to save recorded state data.",
    )
    parser.add_argument(
        "--record-state-stride",
        type=int,
        default=10,
        help="Record one state sample every N MuJoCo steps.",
    )
    parser.add_argument("--duration", type=float, help="Override simulation duration in simulated seconds.")
    parser.add_argument("--xml-path", help="Override the terrain XML path from the YAML config.")
    parser.add_argument("--device", help="Override the policy inference device, for example cuda:1 or cpu.")
    parser.add_argument("--cmd-x", type=float, help="Override the initial forward velocity command in m/s.")
    parser.add_argument(
        "--manual-vision",
        action="store_true",
        help="Force the deployed estimator to use camera observations.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the interactive MuJoCo and depth-preview windows.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.load(file_obj, Loader=yaml.FullLoader)

    if args.duration is not None:
        config["simulation_duration"] = args.duration
    if args.xml_path:
        config["xml_path"] = args.xml_path
    if args.device:
        config["device"] = args.device
    if args.cmd_x is not None:
        config["cmd_init"][0] = args.cmd_x
    if args.manual_vision:
        config.setdefault("camera", {})["manual_vision_flag"] = True

    policy_path = resolve_repo_path(config["policy_path"])
    policy_type = infer_policy_type(policy_path, config)
    xml_path = resolve_repo_path(config["xml_path"])
    device = config.get("device", "cpu")
    robot_name = infer_robot_name(config, config_path, xml_path, policy_path)
    terrain_name = infer_terrain_name(xml_path)

    simulation_duration = float(config["simulation_duration"])
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    lin_vel_scale = float(config["lin_vel_scale"])
    ang_vel_scale = float(config["ang_vel_scale"])
    dof_pos_scale = float(config["dof_pos_scale"])
    dof_vel_scale = float(config["dof_vel_scale"])
    action_scale = float(config["action_scale"])
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    max_cmd = np.array(config["max_cmd"], dtype=np.float32)
    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    depth_preview_cfg = dict(config.get("depth_preview", {}))
    depth_preview_enabled = bool(
        depth_preview_cfg.get("enabled", policy_type == "parkour_moe_bundle")
    ) and not args.headless
    depth_preview_size = tuple(int(value) for value in depth_preview_cfg.get("size", [240, 160]))
    token_overlay_cfg = dict(depth_preview_cfg.get("token_overlay", {}))
    active_token_overlay_cfg = dict(depth_preview_cfg.get("active_token_overlay", {}))
    recon_scan_cfg = dict(config.get("recon_scan", {}))
    recon_scan_enabled = bool(recon_scan_cfg.get("enabled", False))

    idx_model2mj = idx_mj2model = list(range(num_actions))
    mujoco_joint_names = config.get("mujoco_joint_names", [f"joint_{index}" for index in range(num_actions)])
    model_joint_names = config.get("model_joint_names", mujoco_joint_names)
    if "mujoco_joint_names" in config and "model_joint_names" in config:
        idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
        idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    camera_cfg = dict(config.get("camera", {}))

    if policy_type == "parkour_moe_bundle":
        camera_cfg.setdefault("enabled", True)
        camera_cfg.setdefault("buffer_len", 1)
        camera_cfg.setdefault("output_height", 58)
        camera_cfg.setdefault("output_width", 87)
        camera_cfg.setdefault("raw_height", 60)
        camera_cfg.setdefault("raw_width", 106)
        camera_cfg.setdefault("crop_top", 0)
        camera_cfg.setdefault("crop_bottom", 2)
        camera_cfg.setdefault("crop_left", 4)
        camera_cfg.setdefault("crop_right", 4)
        camera_cfg.setdefault("clipping_range", 2.0)
        camera_cfg.setdefault("attach_body", "base")
        camera_cfg.setdefault("convention", "ros")
        if "local_pos" not in camera_cfg or "local_euler_xyz" not in camera_cfg:
            raise ValueError("Parkour MoE deployment requires camera.local_pos and camera.local_euler_xyz.")

    policy_adapter = (
        ParkourMoEPolicyAdapter(policy_path, num_obs, num_actions, camera_cfg, idx_model2mj, device)
        if policy_type == "parkour_moe_bundle"
        else LegacyPolicyAdapter(policy_path, idx_model2mj, device)
    )
    print(f"Loaded policy type: {policy_type}")

    prepared_xml_path = None
    generated_xml_dir = None
    video_writer = None
    video_path = None
    latent_path = None
    all_latents = None
    state_recorder = None
    state_data_path = None
    screen = None
    depth_preview_window = None
    if not args.headless:
        pygame.init()
    use_joystick = False
    joystick = None
    if not args.headless and pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        use_joystick = True
        print(f"Detected Joystick: {joystick.get_name()}")
    else:
        print("No Joystick detected. Using default commands from config.")

    if args.visualize_moe_weights and not args.headless:
        screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("MoE Weights")
    if depth_preview_enabled:
        depth_preview_window = OpenCVDepthPreview(
            depth_preview_size,
            enabled=True,
            token_overlay_cfg=token_overlay_cfg,
            active_token_overlay_cfg=active_token_overlay_cfg,
        )

    try:
        prepared_xml_path, generated_xml_dir = prepare_model_xml(
            xml_path,
            camera_cfg,
            policy_adapter.requires_camera,
        )

        model = mujoco.MjModel.from_xml_path(prepared_xml_path)
        data = mujoco.MjData(model)
        model.opt.timestep = simulation_dt
        if not args.no_record_state:
            state_recorder = MujocoStateRecorder(
                args.action_data_dir,
                robot_name,
                terrain_name,
                mujoco_joint_names,
                model,
                config_path,
                xml_path,
                simulation_dt,
                policy_path=policy_path,
            )
            print(f"State recording enabled: {Path(args.action_data_dir) / state_recorder.robot_name}/{state_recorder.terrain_name}_*.npz")

        video_renderer = mujoco.Renderer(model, height=360, width=640) if args.save_video else None
        policy_camera = PolicyDepthCamera(
            model,
            str(camera_cfg.get("camera_name", "policy_depth_camera")),
            camera_cfg,
        )
        mujoco_render_utils = MujocoRenderUtils(50, model.opt.timestep)

        video_save_dir = str(PATH_PARENT / "videos")
        os.makedirs(video_save_dir, exist_ok=True)
        model_name = os.path.basename(policy_path).split(".")[0]
        cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

        if args.save_video:
            video_filename = f"{model_name}_{cmd_str}.mp4"
            video_path = os.path.join(video_save_dir, video_filename)
            video_fps = 50
            sim_fps = 1.0 / model.opt.timestep
            frame_skip = max(int(sim_fps / video_fps), 1)
            video_writer = imageio.get_writer(video_path, fps=video_fps)
            print(
                f"Video recording will be saved to: {video_path}\n"
                f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}"
            )
        else:
            frame_skip = 1
            video_path = None

        if args.save_moe_latent:
            latent_save_dir = str(PATH_PARENT / "data_latents")
            os.makedirs(latent_save_dir, exist_ok=True)
            latent_path = os.path.join(latent_save_dir, f"{model_name}_{cmd_str}_latents.npy")
            all_latents = []
        else:
            latent_path = None
            all_latents = None

        action = np.zeros(num_actions, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        counter = 0
        control_counter = 0
        last_weights = None
        last_depth_frame = None
        current_recon_scan_points = None
        last_token_mask_info = None
        last_terrain_info = None
        depth_frame_announced = False
        exit_requested = False
        weights_preview_dirty = False
        depth_preview_dirty = False
        policy_adapter.reset()
        if policy_adapter.requires_camera:
            if policy_camera.maybe_update(0, model, data) and policy_camera.initialized:
                last_depth_frame = np.array(policy_camera.depth_buffer[-1], copy=True)
                depth_preview_dirty = True
                policy_adapter.warm_up(policy_camera.depth_buffer)

        viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = 1
                viewer.cam.distance = 2.0
                viewer.cam.elevation = -20.0
                viewer.cam.azimuth = 60.0

            start_sim_time = float(data.time)
            while (
                (viewer is None or viewer.is_running())
                and not exit_requested
                and float(data.time) - start_sim_time < simulation_duration
            ):
                epoch_start = time.time()
                depth_frame_updated = False
                if screen is not None:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            exit_requested = True

                vel = data.qvel[:3]
                ang_vel = data.qvel[3:6]
                local_vel = quat_rotate_inverse(data.qpos[3:7], vel)
                local_ang_vel = quat_rotate_inverse(data.qpos[3:7], ang_vel)
                show_str = (
                    f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, "
                    f"Wz={local_ang_vel[2]:.2f}"
                )

                if counter % control_decimation == 0:
                    if use_joystick:
                        cmd = get_xbox_command(joystick, max_cmd)
                    if use_joystick:
                        show_str += f", Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}"
                        print(show_str, end="\r")

                tau = pd_control(
                    target_dof_pos,
                    data.qpos[7:],
                    kps,
                    np.zeros_like(kds),
                    data.qvel[6:],
                    kds,
                )
                data.ctrl[:] = tau

                mujoco.mj_step(model, data)
                counter += 1
                if state_recorder is not None and counter % max(args.record_state_stride, 1) == 0:
                    state_recorder.record(
                        model,
                        data,
                        counter,
                        tau,
                        action=action,
                        target_dof_pos=target_dof_pos,
                        cmd=cmd,
                    )

                if counter % control_decimation == 0:
                    control_counter += 1
                    depth_frame_updated = policy_camera.maybe_update(control_counter, model, data)
                    if depth_frame_updated and policy_camera.initialized:
                        last_depth_frame = np.array(policy_camera.depth_buffer[-1], copy=True)
                        depth_preview_dirty = True
                        if not depth_frame_announced:
                            print(
                                f"\nDepth preview ready: shape={last_depth_frame.shape}, "
                                f"min={last_depth_frame.min():+.3f}, max={last_depth_frame.max():+.3f}"
                            )
                            depth_frame_announced = True

                    obs = build_observation(
                        data,
                        default_angles,
                        dof_pos_scale,
                        dof_vel_scale,
                        ang_vel_scale,
                        cmd,
                        cmd_scale,
                        action,
                        idx_mj2model,
                        num_obs,
                    )
                    refresh_estimator = policy_camera.consume_refresh_flag()
                    action, extras = policy_adapter.act(
                        obs,
                        policy_camera.depth_buffer if policy_adapter.requires_camera else None,
                        refresh_estimator,
                    )
                    if policy_adapter.requires_camera:
                        latest_token_mask_info = getattr(policy_adapter, "latest_token_mask_info", {})
                        if latest_token_mask_info:
                            last_token_mask_info = latest_token_mask_info
                            depth_preview_dirty = True
                    action = np.clip(action, -100.0, 100.0).astype(np.float32)
                    target_dof_pos = action * action_scale + default_angles
                    if extras.get("weights", None) is not None:
                        last_weights = extras["weights"]
                        weights_preview_dirty = True
                    if extras.get("terrain_pred", None) is not None:
                        last_terrain_info = {
                            "terrain_pred": extras.get("terrain_pred", None),
                            "terrain_probs": extras.get("terrain_probs", None),
                        }
                        depth_preview_dirty = True
                    if recon_scan_enabled:
                        latest_m_hat = getattr(policy_adapter, "latest_m_hat", None)
                        current_recon_scan_points = reconstruct_scan_points_world(
                            latest_m_hat,
                            data.qpos[:3],
                            data.qpos[3:7],
                            recon_scan_cfg,
                        )
                    if args.save_moe_latent and all_latents is not None and extras.get("latent", None) is not None:
                        all_latents.append(np.array(extras["latent"], copy=True))

                mujoco_render_utils.update(cmd, data, current_recon_scan_points, recon_scan_cfg if recon_scan_enabled else None)

                if video_writer is not None and counter % frame_skip == 0:
                    try:
                        video_renderer.update_scene(data, camera=viewer.cam)
                        mujoco_render_utils.update_external_rendering(video_renderer, ctype="renderer")
                        frame = video_renderer.render()
                        video_writer.append_data(frame)
                    except Exception as exc:
                        print(f"Error rendering frame: {exc}")

                if screen is not None and args.visualize_moe_weights and weights_preview_dirty:
                    draw_moe_weights(screen, last_weights, 400, 200)
                    weights_preview_dirty = False
                if depth_preview_window is not None and depth_preview_dirty:
                    preview_key = depth_preview_window.update(
                        last_depth_frame if depth_preview_enabled else None,
                        last_token_mask_info,
                        last_terrain_info,
                    )
                    if preview_key == 27:
                        exit_requested = True
                    depth_preview_dirty = False

                if viewer is not None:
                    mujoco_render_utils.update_external_rendering(viewer, ctype="viewer")
                    viewer.sync()
                #print(f"Epoch time: {time.time() - epoch_start}")

    finally:
        if video_writer is not None:
            video_writer.close()
            if video_path is not None:
                print(f"Video saved successfully to {video_path}")

        if args.save_moe_latent and all_latents:
            np.save(latent_path, np.stack(all_latents, axis=0))
            print(f"Saved MoE latents to {latent_path}")

        if state_recorder is not None:
            state_data_path = state_recorder.save()
            if state_data_path is not None:
                print(f"Saved state data to {state_data_path}")

        if depth_preview_window is not None:
            depth_preview_window.close()
        pygame.quit()
        if generated_xml_dir is not None:
            shutil.rmtree(generated_xml_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
