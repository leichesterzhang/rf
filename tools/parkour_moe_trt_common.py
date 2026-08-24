#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "rsl_rl"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from rsl_rl.modules import ActorCriticParkourMoE, ParkourEstimator


DEFAULT_BUNDLE = REPO_ROOT / "logs/go2_parkour_moe/exported/policies/parkour_moe_inference_bundle.pt"
DEFAULT_ASSET_DIR = REPO_ROOT / "logs/go2_parkour_moe/exported/policies/parkour_moe_trt_assets"


@dataclass
class ParkourMoEMeta:
    num_obs: int
    num_critic_obs: int
    num_actions: int
    history_length: int
    depth_frame_count: int
    depth_height: int
    depth_width: int
    shared_state_dim: int
    mcp_dim: int
    mcp_dim_with_flag: int
    terrain_map_dim: int
    estimator_expert_num: int
    num_terrain_types: int


def disable_torch_export_fastpaths() -> None:
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)


def disable_transformer_nested_tensor(estimator: ParkourEstimator) -> None:
    if hasattr(estimator.transformer, "enable_nested_tensor"):
        estimator.transformer.enable_nested_tensor = False
    if hasattr(estimator.transformer, "use_nested_tensor"):
        estimator.transformer.use_nested_tensor = False


def infer_actor_dims(bundle: dict) -> tuple[int, int, int]:
    policy_cfg = dict(bundle["policy_cfg"])
    actor_state_dict = bundle["actor_critic_state_dict"]
    num_actions = int(actor_state_dict["std"].numel())
    actor_input_dim = int(actor_state_dict["actor_moe.experts.backbone.network.0.weight"].shape[1])
    mcp_dim = int(policy_cfg.get("mcp_dim", 68))
    vision_flag_dim = 1
    num_actor_obs_now = int(policy_cfg.get("num_actor_obs_now", actor_input_dim - mcp_dim - vision_flag_dim))
    num_obs = num_actor_obs_now

    if "critic_experts.backbone.network.0.weight" in actor_state_dict:
        num_critic_obs = int(actor_state_dict["critic_experts.backbone.network.0.weight"].shape[1] - vision_flag_dim)
    else:
        num_critic_obs = int(actor_state_dict["critic.0.weight"].shape[1] - vision_flag_dim)
    return num_obs, num_critic_obs, num_actions


def build_meta(bundle: dict, estimator: ParkourEstimator, depth_height: int, depth_width: int) -> ParkourMoEMeta:
    estimator_cfg = dict(bundle["estimator_cfg"])
    policy_cfg = dict(bundle["policy_cfg"])
    num_obs, num_critic_obs, num_actions = infer_actor_dims(bundle)
    mcp_dim = int(policy_cfg.get("mcp_dim", 68))
    return ParkourMoEMeta(
        num_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        history_length=int(estimator_cfg.get("history_length", 10)),
        depth_frame_count=int(estimator_cfg.get("depth_frame_count", 1)),
        depth_height=int(depth_height),
        depth_width=int(depth_width),
        shared_state_dim=int(estimator.shared_state_dim),
        mcp_dim=mcp_dim,
        mcp_dim_with_flag=mcp_dim + 1,
        terrain_map_dim=int(estimator_cfg.get("terrain_map_dim", 187)),
        estimator_expert_num=int(estimator_cfg.get("expert_num", 3)),
        num_terrain_types=int(estimator_cfg.get("num_terrain_types", 12)),
    )


def load_bundle_models(bundle_path: Path | str, device: torch.device | str = "cpu", depth_height: int = 58, depth_width: int = 87):
    bundle_path = Path(bundle_path).expanduser().resolve()
    device = torch.device(device)
    bundle = torch.load(bundle_path, map_location=device)
    num_obs, num_critic_obs, num_actions = infer_actor_dims(bundle)

    actor_critic = ActorCriticParkourMoE(
        num_obs,
        num_critic_obs,
        num_actions,
        **bundle["policy_cfg"],
    ).to(device)
    estimator = ParkourEstimator(**bundle["estimator_cfg"]).to(device)

    actor_critic.load_state_dict(bundle["actor_critic_state_dict"])
    estimator_result = estimator.load_state_dict(bundle["estimator_state_dict"], strict=False)
    missing = list(estimator_result.missing_keys)
    unexpected = list(estimator_result.unexpected_keys)
    allowed_unexpected_prefixes = ("core.context_patch_decoder.",)
    disallowed_unexpected = [key for key in unexpected if not key.startswith(allowed_unexpected_prefixes)]
    if missing or disallowed_unexpected:
        raise RuntimeError(
            "Failed to load estimator state_dict cleanly. "
            f"missing={missing}, unexpected={unexpected}"
        )

    actor_critic.eval()
    estimator.eval()
    disable_transformer_nested_tensor(estimator)
    meta = build_meta(bundle, estimator, depth_height=depth_height, depth_width=depth_width)
    return bundle, actor_critic, estimator, meta


class EstimatorExportWrapper(nn.Module):
    def __init__(self, estimator: ParkourEstimator):
        super().__init__()
        self.estimator = estimator

    def forward(self, proprio_history, depth, vision_flag, obs_now, shared_state):
        vision_mask = vision_flag.reshape(-1) > 0.5
        out = self.estimator(
            proprio_history,
            depth,
            vision_mask,
            obs_now=obs_now,
            shared_state_override=shared_state,
            update_state=False,
        )
        vision_flag = vision_flag.to(dtype=out["mcp_code"].dtype, device=out["mcp_code"].device).reshape(-1, 1)
        mcp_code = torch.cat((out["mcp_code"], vision_flag), dim=-1)
        return (
            mcp_code,
            out["shared_state"],
            out["m_hat"],
            out["swav_gating_weights"],
            out["terrain_probs"],
            out["fall_recovery_prob"],
        )


class ActorExportWrapper(nn.Module):
    def __init__(self, actor_critic: ActorCriticParkourMoE):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, mcp_code, obs):
        return self.actor_critic.act_inference(mcp_code, obs)


def make_manifest(bundle_path: Path, output_dir: Path, bundle: dict, meta: ParkourMoEMeta, opset: int, batch_size: int) -> dict:
    return {
        "bundle": str(bundle_path),
        "opset": int(opset),
        "files": {
            "estimator": "estimator.onnx",
            "actor": "actor.onnx",
            "estimator_fp32_engine": "estimator.fp32.engine",
            "estimator_fp16_engine": "estimator.fp16.engine",
            "actor_fp32_engine": "actor.fp32.engine",
            "actor_fp16_engine": "actor.fp16.engine",
        },
        "inputs": {
            "estimator": {
                "proprio_history": [batch_size, meta.history_length * meta.num_obs],
                "depth": [batch_size, meta.depth_frame_count, meta.depth_height, meta.depth_width],
                "vision_flag": [batch_size, 1],
                "obs_now": [batch_size, meta.num_obs],
                "shared_state": [batch_size, meta.shared_state_dim],
            },
            "actor": {
                "mcp_code": [batch_size, meta.mcp_dim_with_flag],
                "obs": [batch_size, meta.num_obs],
            },
        },
        "outputs": {
            "estimator": {
                "mcp_code": [batch_size, meta.mcp_dim_with_flag],
                "shared_state_next": [batch_size, meta.shared_state_dim],
                "m_hat": [batch_size, meta.terrain_map_dim],
                "gating_weights": [batch_size, meta.estimator_expert_num],
                "terrain_probs": [batch_size, meta.num_terrain_types],
                "fall_recovery_prob": [batch_size, 1],
            },
            "actor": {
                "action": [batch_size, meta.num_actions],
            },
        },
        "policy_cfg": dict(bundle["policy_cfg"]),
        "estimator_cfg": dict(bundle["estimator_cfg"]),
        "num_obs": meta.num_obs,
        "num_critic_obs": meta.num_critic_obs,
        "num_actions": meta.num_actions,
        "asset_dir": str(output_dir),
        "note": "Current Go2 Parkour MoE deployment uses estimator.onnx + actor.onnx. No selector ONNX is exported.",
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(onnx_dir: Path) -> dict:
    manifest_path = onnx_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
