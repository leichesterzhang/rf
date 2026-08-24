from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover
    trt = None


LEGGED_GYM_ROOT_DIR = str(Path(__file__).parents[2])
for candidate in (Path(LEGGED_GYM_ROOT_DIR), Path(LEGGED_GYM_ROOT_DIR) / "rsl_rl"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from rsl_rl.modules import ActorCriticParkourMoE, ParkourEstimator


def to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


@dataclass
class ParkourMoEBundleMeta:
    num_obs: int
    num_actions: int
    num_actor_obs_now: int
    critic_obs_dim: int
    history_length: int
    shared_state_dim: int
    depth_frames: int
    depth_height: int
    depth_width: int
    actor_mcp_input_dim: int
    append_vision_flag: bool


ENGINE_MODEL_NAMES = {
    "estimator": "estimator",
    "actor": "actor",
}


def _validate_bundle_layout(bundle):
    estimator_state_dict = bundle["estimator_state_dict"]
    required_keys = (
        "shared_pool_query",
        "expert_pool_queries",
        "selector_head.0.weight",
        "terrain_token_encoder.0.weight",
        "router_terrain_encoder.0.weight",
        "shared_state_norm.weight",
    )
    missing_keys = [key for key in required_keys if key not in estimator_state_dict]
    if not missing_keys:
        return

    if "core.vision_flag_encoder.weight" in estimator_state_dict:
        raise RuntimeError(
            "Legacy parkour MoE bundle detected. The current deploy path and TRT export utilities only "
            "support the newer terrain-aware estimator bundle. Please export a newer bundle before using this path."
        )

    raise RuntimeError(f"Unsupported parkour MoE bundle layout. Missing keys: {missing_keys}")


def infer_bundle_meta(bundle, num_obs, num_actions, depth_height, depth_width, selector_threshold=None):
    _validate_bundle_layout(bundle)

    actor_state_dict = bundle["actor_critic_state_dict"]
    estimator_cfg = bundle["estimator_cfg"]
    policy_cfg = bundle["policy_cfg"]

    if "critic_experts.backbone.network.0.weight" in actor_state_dict:
        critic_obs_dim = int(actor_state_dict["critic_experts.backbone.network.0.weight"].shape[1] - 1)
    else:
        critic_obs_dim = int(actor_state_dict["critic.0.weight"].shape[1] - 1)
    actor_input_dim = int(actor_state_dict["actor_moe.experts.backbone.network.0.weight"].shape[1])
    num_actor_obs_now = int(policy_cfg.get("num_actor_obs_now", num_obs))
    actor_mcp_input_dim = actor_input_dim - num_actor_obs_now
    append_vision_flag = actor_mcp_input_dim == int(policy_cfg["mcp_dim"]) + 1
    if not append_vision_flag and actor_mcp_input_dim != int(policy_cfg["mcp_dim"]):
        raise RuntimeError(
            "Could not infer actor MCP input dimension from bundle. "
            f"actor_input={actor_input_dim}, "
            f"obs_now={num_actor_obs_now}, policy_mcp_dim={policy_cfg['mcp_dim']}"
        )

    return ParkourMoEBundleMeta(
        num_obs=int(num_obs),
        num_actions=int(num_actions),
        num_actor_obs_now=num_actor_obs_now,
        critic_obs_dim=critic_obs_dim,
        history_length=int(estimator_cfg.get("history_length", 10)),
        shared_state_dim=int(estimator_cfg.get("shared_state_dim", estimator_cfg.get("vision_token_dim", 64))),
        depth_frames=int(estimator_cfg.get("depth_frame_count", 1)),
        depth_height=int(depth_height),
        depth_width=int(depth_width),
        actor_mcp_input_dim=actor_mcp_input_dim,
        append_vision_flag=append_vision_flag,
    )


def infer_manifest_meta(manifest, num_obs=None, num_actions=None):
    estimator_inputs = manifest["inputs"]["estimator"]
    actor_inputs = manifest["inputs"]["actor"]
    actor_outputs = manifest["outputs"]["actor"]

    obs_dim = int(num_obs if num_obs is not None else manifest.get("num_obs", actor_inputs["obs"][1]))
    action_dim = int(num_actions if num_actions is not None else manifest.get("num_actions", actor_outputs["action"][1]))
    proprio_history_dim = int(estimator_inputs["proprio_history"][1])
    history_length = proprio_history_dim // obs_dim
    if history_length * obs_dim != proprio_history_dim:
        raise RuntimeError(
            "Invalid manifest proprio_history shape: "
            f"{estimator_inputs['proprio_history']} is not divisible by num_obs={obs_dim}"
        )

    depth_shape = estimator_inputs["depth"]
    shared_state_shape = estimator_inputs["shared_state"]
    mcp_shape = actor_inputs["mcp_code"]
    policy_cfg = dict(manifest.get("policy_cfg", {}))
    mcp_dim = int(policy_cfg.get("mcp_dim", int(mcp_shape[1]) - 1))

    return ParkourMoEBundleMeta(
        num_obs=obs_dim,
        num_actions=action_dim,
        num_actor_obs_now=int(policy_cfg.get("num_actor_obs_now", obs_dim)),
        critic_obs_dim=int(manifest.get("num_critic_obs", 0)),
        history_length=int(history_length),
        shared_state_dim=int(shared_state_shape[1]),
        depth_frames=int(depth_shape[1]),
        depth_height=int(depth_shape[2]),
        depth_width=int(depth_shape[3]),
        actor_mcp_input_dim=int(mcp_shape[1]),
        append_vision_flag=int(mcp_shape[1]) == mcp_dim + 1,
    )


def resolve_parkour_moe_manifest_path(engine_dir=None, engine_paths=None):
    candidates = []
    if engine_dir:
        engine_dir = Path(engine_dir)
        candidates.extend(
            (
                engine_dir / "manifest.json",
                engine_dir.parent / "manifest.json",
            )
        )
    for path_str in dict(engine_paths or {}).values():
        path = Path(path_str)
        candidates.append(path.parent / "manifest.json")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def resolve_parkour_moe_bundle_path(bundle_path, engine_dir=None):
    candidates = []
    if bundle_path:
        candidates.append(Path(bundle_path))
    if engine_dir:
        engine_dir = Path(engine_dir)
        candidates.extend(
            (
                engine_dir / "parkour_moe_inference_bundle.pt",
                engine_dir.parent / "parkour_moe_inference_bundle.pt",
            )
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix == ".pt":
            return candidate.resolve()

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate parkour_moe_inference_bundle.pt for metadata loading.\n"
        f"Tried:\n{tried}"
    )


def resolve_parkour_moe_engine_file(engine_dir, model_name, precision="auto"):
    engine_dir = Path(engine_dir).resolve()
    precision = str(precision or "auto").lower()
    candidate_names = []
    if precision == "auto":
        candidate_names.extend(
            (
                f"{model_name}.fp16.engine",
                f"{model_name}.fp32.engine",
                f"{model_name}.engine",
            )
        )
    else:
        candidate_names.extend((f"{model_name}.{precision}.engine", f"{model_name}.engine"))

    for candidate_name in candidate_names:
        candidate_path = engine_dir / candidate_name
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path

    tried = "\n".join(f"  - {engine_dir / candidate_name}" for candidate_name in candidate_names)
    raise FileNotFoundError(
        f"Missing TensorRT engine for {model_name}. Expected one of:\n{tried}"
    )


def parkour_moe_engine_assets_available(engine_dir, enable_selector=False, precision="auto", engine_paths=None):
    del enable_selector
    engine_paths = dict(engine_paths or {})
    try:
        for role in ("estimator", "actor"):
            if role in engine_paths:
                candidate = Path(engine_paths[role])
                if not candidate.exists() or not candidate.is_file():
                    return False
            else:
                resolve_parkour_moe_engine_file(engine_dir, ENGINE_MODEL_NAMES[role], precision=precision)
        return True
    except (FileNotFoundError, TypeError, ValueError):
        return False


def _torch_dtype_from_trt(trt_dtype):
    np_dtype = np.dtype(trt.nptype(trt_dtype))
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.bool_): torch.bool,
    }
    if np_dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")
    return mapping[np_dtype]


class TensorRTEngineRunner:
    def __init__(self, engine_path, device):
        if trt is None:
            raise ImportError(
                "TensorRT Python bindings are not installed. Install `tensorrt` on the target machine "
                "before using `.engine` deployment."
            )

        self.engine_path = Path(engine_path).resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"TensorRT engine deployment requires a CUDA device, but got: {self.device}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, so TensorRT engines cannot be executed.")

        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context for: {self.engine_path}")

        self._uses_tensor_api = hasattr(self.engine, "num_io_tensors")
        self._name_to_index = {}
        self.input_names = []
        self.output_names = []
        self._output_buffers = {}
        self._bindings = [0] * getattr(self.engine, "num_bindings", 0)

        if self._uses_tensor_api:
            for tensor_idx in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(tensor_idx)
                mode = self.engine.get_tensor_mode(tensor_name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(tensor_name)
                else:
                    self.output_names.append(tensor_name)
        else:
            for binding_idx in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(binding_idx)
                self._name_to_index[binding_name] = binding_idx
                if self.engine.binding_is_input(binding_idx):
                    self.input_names.append(binding_name)
                else:
                    self.output_names.append(binding_name)

    def _get_dtype(self, name):
        if self._uses_tensor_api:
            return self.engine.get_tensor_dtype(name)
        return self.engine.get_binding_dtype(self._name_to_index[name])

    def _set_input_shape(self, name, shape):
        shape = tuple(int(dim) for dim in shape)
        if self._uses_tensor_api and hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(name, shape)
            return
        self.context.set_binding_shape(self._name_to_index[name], shape)

    def _get_runtime_shape(self, name):
        if self._uses_tensor_api and hasattr(self.context, "get_tensor_shape"):
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        elif not self._uses_tensor_api:
            shape = tuple(int(dim) for dim in self.context.get_binding_shape(self._name_to_index[name]))
        elif hasattr(self.engine, "get_tensor_shape"):
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
        else:
            raise RuntimeError(f"Could not query runtime shape for TensorRT tensor: {name}")

        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"Dynamic output shape for {name} was not resolved: {shape}")
        return shape

    def _bind_tensor(self, name, tensor):
        if self._uses_tensor_api and hasattr(self.context, "set_tensor_address"):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            return
        self._bindings[self._name_to_index[name]] = int(tensor.data_ptr())

    def _prepare_input_tensor(self, name, value):
        expected_dtype = _torch_dtype_from_trt(self._get_dtype(name))
        tensor = torch.as_tensor(value, device=self.device)
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(dtype=expected_dtype)
        return tensor.contiguous()

    def _prepare_output_tensor(self, name):
        shape = self._get_runtime_shape(name)
        dtype = _torch_dtype_from_trt(self._get_dtype(name))
        existing = self._output_buffers.get(name)
        if (
            existing is None
            or tuple(existing.shape) != shape
            or existing.dtype != dtype
            or existing.device != self.device
        ):
            existing = torch.empty(shape, device=self.device, dtype=dtype)
            self._output_buffers[name] = existing
        return existing

    def infer(self, inputs):
        if set(inputs.keys()) != set(self.input_names):
            raise ValueError(
                f"TensorRT engine inputs mismatch for {self.engine_path.name}. "
                f"Expected {self.input_names}, got {sorted(inputs.keys())}."
            )

        for name in self.input_names:
            tensor = self._prepare_input_tensor(name, inputs[name])
            self._set_input_shape(name, tensor.shape)
            self._bind_tensor(name, tensor)

        outputs = {}
        for name in self.output_names:
            tensor = self._prepare_output_tensor(name)
            self._bind_tensor(name, tensor)
            outputs[name] = tensor

        stream = torch.cuda.current_stream(device=self.device)
        stream_handle = int(stream.cuda_stream)
        if hasattr(self.context, "execute_async_v3"):
            ok = self.context.execute_async_v3(stream_handle)
        elif hasattr(self.context, "execute_async_v2"):
            ok = self.context.execute_async_v2(self._bindings, stream_handle)
        else:
            ok = self.context.execute_v2(self._bindings)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")

        return outputs


class ParkourMoEEstimatorStep(nn.Module):
    def __init__(self, estimator, append_vision_flag=True):
        super().__init__()
        self.estimator = estimator
        self.append_vision_flag = bool(append_vision_flag)

    def forward(self, proprio_history, depth, vision_flag, obs_now, shared_state):
        if vision_flag.dim() == 1:
            vision_flag = vision_flag.unsqueeze(-1)
        vision_flag = vision_flag.to(device=obs_now.device, dtype=obs_now.dtype)
        mask_vision = vision_flag[:, 0] > 0.5

        est_out = self.estimator(
            proprio_history,
            depth,
            mask_vision,
            obs_now=obs_now,
            shared_state_override=shared_state,
            update_state=False,
        )

        mcp_code = est_out["mcp_code"]
        if self.append_vision_flag:
            mcp_code = torch.cat((mcp_code, vision_flag.to(dtype=mcp_code.dtype)), dim=-1)

        return (
            mcp_code,
            est_out["shared_state"],
            est_out["m_hat"],
            est_out["swav_gating_weights"],
            est_out["terrain_probs"],
            est_out["fall_recovery_prob"],
        )


class ParkourMoEActorStep(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, mcp_code, obs):
        return self.actor_critic.act_inference(mcp_code, obs)


class ParkourMoEBaseAdapter:
    def _init_runtime_state(self):
        self.history = torch.zeros(
            self.meta.history_length,
            self.meta.num_obs,
            device=self.device,
            dtype=torch.float32,
        )
        self.shared_state = torch.zeros(
            1,
            self.meta.shared_state_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.latest_mcp_code = None
        self.latest_m_hat = None
        self.latest_gating_weights = None
        self.latest_terrain_pred = None
        self.latest_estimator_output = {}

    def reset(self):
        self.history.zero_()
        self.shared_state = self.shared_state.clone()
        self.shared_state.zero_()
        self.latest_mcp_code = None
        self.latest_m_hat = None
        self.latest_gating_weights = None
        self.latest_terrain_pred = None
        self.latest_estimator_output = {}

    def warm_up(self, depth_buffer, use_vision=False):
        warm_obs = np.ones(self.meta.num_obs, dtype=np.float32)
        for _ in range(5):
            self.act(warm_obs, depth_buffer, refresh_estimator=True, use_vision=use_vision)
        self.reset()
        print("Parkour MoE network has been warmed up.")

    def _to_obs_tensor(self, obs):
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)

    def _to_depth_tensor(self, depth_buffer):
        depth_array = np.asarray(depth_buffer, dtype=np.float32)
        if depth_array.ndim == 2:
            depth_array = depth_array[None, :, :]
        if depth_array.ndim != 3:
            raise ValueError(f"Expected depth buffer [frames,H,W] or [H,W], got shape {depth_array.shape}")
        if depth_array.shape[0] < self.meta.depth_frames:
            repeat_count = self.meta.depth_frames - depth_array.shape[0]
            pad = np.repeat(depth_array[:1], repeat_count, axis=0)
            depth_array = np.concatenate((pad, depth_array), axis=0)
        depth_array = depth_array[-self.meta.depth_frames :]
        return torch.as_tensor(depth_array, device=self.device, dtype=torch.float32).unsqueeze(0)

    def _update_history(self, obs_tensor):
        self.history = torch.roll(self.history, shifts=-1, dims=0)
        self.history[-1] = obs_tensor[0]

    def _build_vision_flag(self, use_vision):
        return torch.full(
            (1, 1),
            1.0 if bool(use_vision) else 0.0,
            device=self.device,
            dtype=torch.float32,
        )


class ParkourMoEPolicyAdapter(ParkourMoEBaseAdapter):
    def __init__(self, bundle_path, num_obs, num_actions, camera_cfg, device):
        self.device = torch.device(device)
        self.camera_cfg = dict(camera_cfg)

        bundle = torch.load(bundle_path, map_location=self.device)
        self.meta = infer_bundle_meta(
            bundle,
            num_obs=num_obs,
            num_actions=num_actions,
            depth_height=int(self.camera_cfg.get("output_height", 58)),
            depth_width=int(self.camera_cfg.get("output_width", 87)),
        )

        self.actor_critic = ActorCriticParkourMoE(
            self.meta.num_obs,
            self.meta.critic_obs_dim,
            self.meta.num_actions,
            **bundle["policy_cfg"],
        ).to(self.device)
        self.estimator = ParkourEstimator(**bundle["estimator_cfg"]).to(self.device)

        self.actor_critic.load_state_dict(bundle["actor_critic_state_dict"])
        self.estimator.load_state_dict(bundle["estimator_state_dict"])

        self.actor_critic.eval()
        self.estimator.eval()

        self.enable_selector = False
        self.estimator_step = ParkourMoEEstimatorStep(
            self.estimator,
            append_vision_flag=self.meta.append_vision_flag,
        ).to(self.device)
        self.actor_step = ParkourMoEActorStep(self.actor_critic).to(self.device)
        self.estimator_step.eval()
        self.actor_step.eval()

        self._init_runtime_state()

    def reset(self):
        super().reset()
        self.estimator.reset()

    def _refresh_estimator(self, obs_tensor, depth_tensor, use_vision):
        proprio_history = self.history.reshape(1, -1)
        vision_flag = self._build_vision_flag(use_vision)

        mcp_code, shared_state, m_hat, gating_weights, terrain_probs, fall_recovery_prob = self.estimator_step(
            proprio_history,
            depth_tensor,
            vision_flag,
            obs_tensor,
            self.shared_state,
        )
        self.shared_state = shared_state.detach()
        self.latest_mcp_code = mcp_code.detach()
        self.latest_m_hat = to_numpy(m_hat).squeeze().astype(np.float32)
        self.latest_gating_weights = gating_weights.detach()
        self.latest_terrain_pred = int(np.argmax(to_numpy(terrain_probs).reshape(-1)))
        self.latest_estimator_output = {
            "mcp_code": self.latest_mcp_code,
            "m_hat": m_hat.detach(),
            "swav_gating_weights": self.latest_gating_weights,
            "terrain_probs": terrain_probs.detach(),
            "fall_recovery_prob": fall_recovery_prob.detach(),
            "shared_state": self.shared_state,
        }

    def act(self, obs, depth_buffer, refresh_estimator=False, use_vision=False):
        obs_tensor = self._to_obs_tensor(obs)
        self._update_history(obs_tensor)

        with torch.inference_mode():
            if refresh_estimator or self.latest_mcp_code is None:
                depth_tensor = self._to_depth_tensor(depth_buffer)
                self._refresh_estimator(obs_tensor, depth_tensor, use_vision)

            action = self.actor_step(self.latest_mcp_code, obs_tensor)

        return to_numpy(action).squeeze().astype(np.float32)

    def export_inference_assets(
        self,
        output_dir,
        export_onnx=True,
        export_torchscript=True,
        opset_version=14,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        obs = torch.zeros(1, self.meta.num_obs, dtype=torch.float32)
        proprio_history = torch.zeros(1, self.meta.history_length * self.meta.num_obs, dtype=torch.float32)
        depth = torch.zeros(
            1,
            self.meta.depth_frames,
            self.meta.depth_height,
            self.meta.depth_width,
            dtype=torch.float32,
        )
        vision_flag = torch.ones(1, 1, dtype=torch.float32)
        shared_state = torch.zeros(1, self.meta.shared_state_dim, dtype=torch.float32)
        actor_mcp_code = torch.zeros(1, self.meta.actor_mcp_input_dim, dtype=torch.float32)

        estimator_cpu = self.estimator_step.cpu().eval()
        actor_cpu = self.actor_step.cpu().eval()

        if export_torchscript:
            estimator_ts = torch.jit.trace(
                estimator_cpu,
                (proprio_history, depth, vision_flag, obs, shared_state),
                strict=False,
                check_trace=False,
            )
            estimator_ts.save(str(output_dir / "estimator.ts"))

            actor_ts = torch.jit.trace(
                actor_cpu,
                (actor_mcp_code, obs),
                strict=False,
                check_trace=False,
            )
            actor_ts.save(str(output_dir / "actor.ts"))

        if export_onnx:
            torch.onnx.export(
                estimator_cpu,
                (proprio_history, depth, vision_flag, obs, shared_state),
                str(output_dir / "estimator.onnx"),
                export_params=True,
                opset_version=opset_version,
                input_names=[
                    "proprio_history",
                    "depth",
                    "vision_flag",
                    "obs_now",
                    "shared_state",
                ],
                output_names=[
                    "mcp_code",
                    "shared_state_next",
                    "m_hat",
                    "gating_weights",
                    "terrain_probs",
                    "fall_recovery_prob",
                ],
                dynamic_axes={},
            )
            torch.onnx.export(
                actor_cpu,
                (actor_mcp_code, obs),
                str(output_dir / "actor.onnx"),
                export_params=True,
                opset_version=opset_version,
                input_names=["mcp_code", "obs"],
                output_names=["action"],
                dynamic_axes={},
            )

        self.estimator_step.to(self.device).eval()
        self.actor_step.to(self.device).eval()


class ParkourMoETRTPolicyAdapter(ParkourMoEBaseAdapter):
    def __init__(
        self,
        bundle_path,
        num_obs,
        num_actions,
        camera_cfg,
        device,
        engine_dir=None,
        engine_precision="auto",
        engine_paths=None,
    ):
        self.device = torch.device(device)
        self.camera_cfg = dict(camera_cfg)
        self.enable_selector = False
        self.engine_precision = str(engine_precision or "auto").lower()
        self.engine_dir = Path(engine_dir).resolve() if engine_dir else None
        self.engine_paths = {name: Path(path).resolve() for name, path in dict(engine_paths or {}).items()}

        manifest_path = resolve_parkour_moe_manifest_path(self.engine_dir, self.engine_paths)
        if manifest_path is not None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.meta = infer_manifest_meta(manifest, num_obs=num_obs, num_actions=num_actions)
            print(f"Loaded TensorRT metadata from manifest: {manifest_path}")
        else:
            bundle_path = resolve_parkour_moe_bundle_path(bundle_path=bundle_path, engine_dir=self.engine_dir)
            bundle = torch.load(bundle_path, map_location="cpu")
            self.meta = infer_bundle_meta(
                bundle,
                num_obs=num_obs,
                num_actions=num_actions,
                depth_height=int(self.camera_cfg.get("output_height", 58)),
                depth_width=int(self.camera_cfg.get("output_width", 87)),
            )
            print(f"Loaded TensorRT metadata from bundle fallback: {bundle_path}")

        if self.engine_dir is None and not self.engine_paths:
            raise ValueError("TensorRT deployment requires `engine_dir` or explicit `engine_paths`.")
        if self.engine_dir is None:
            required_roles = ["estimator", "actor"]
            missing_roles = [role for role in required_roles if role not in self.engine_paths]
            if missing_roles:
                raise ValueError(
                    "Explicit TensorRT `engine_paths` are incomplete. Missing entries for: "
                    + ", ".join(missing_roles)
                )

        estimator_path = self.engine_paths.get("estimator")
        if estimator_path is None:
            estimator_path = resolve_parkour_moe_engine_file(
                self.engine_dir,
                ENGINE_MODEL_NAMES["estimator"],
                precision=self.engine_precision,
            )
        actor_path = self.engine_paths.get("actor")
        if actor_path is None:
            actor_path = resolve_parkour_moe_engine_file(
                self.engine_dir,
                ENGINE_MODEL_NAMES["actor"],
                precision=self.engine_precision,
            )

        self.estimator_engine = TensorRTEngineRunner(estimator_path, device=self.device)
        self.actor_engine = TensorRTEngineRunner(actor_path, device=self.device)
        self._init_runtime_state()

    def _refresh_estimator(self, obs_tensor, depth_tensor, use_vision):
        proprio_history = self.history.reshape(1, -1)
        vision_flag = self._build_vision_flag(use_vision)

        estimator_outputs = self.estimator_engine.infer(
            {
                "proprio_history": proprio_history,
                "depth": depth_tensor,
                "vision_flag": vision_flag,
                "obs_now": obs_tensor,
                "shared_state": self.shared_state,
            }
        )

        self.shared_state = estimator_outputs["shared_state_next"].clone()
        self.latest_mcp_code = estimator_outputs["mcp_code"].clone()
        self.latest_m_hat = to_numpy(estimator_outputs["m_hat"].float()).squeeze().astype(np.float32)
        self.latest_gating_weights = estimator_outputs["gating_weights"].clone()
        self.latest_terrain_pred = int(np.argmax(to_numpy(estimator_outputs["terrain_probs"].float()).reshape(-1)))
        self.latest_estimator_output = {
            "mcp_code": self.latest_mcp_code,
            "m_hat": estimator_outputs["m_hat"].clone(),
            "swav_gating_weights": self.latest_gating_weights,
            "terrain_probs": estimator_outputs["terrain_probs"].clone(),
            "fall_recovery_prob": estimator_outputs["fall_recovery_prob"].clone(),
            "shared_state": self.shared_state,
        }

    def act(self, obs, depth_buffer, refresh_estimator=False, use_vision=False):
        obs_tensor = self._to_obs_tensor(obs)
        self._update_history(obs_tensor)

        with torch.inference_mode():
            if refresh_estimator or self.latest_mcp_code is None:
                depth_tensor = self._to_depth_tensor(depth_buffer)
                self._refresh_estimator(obs_tensor, depth_tensor, use_vision)

            actor_outputs = self.actor_engine.infer(
                {
                    "mcp_code": self.latest_mcp_code,
                    "obs": obs_tensor,
                }
            )

        return to_numpy(actor_outputs["action"].float()).squeeze().astype(np.float32)
