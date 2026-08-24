#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "onnxruntime is required for ONNX validation. Install it first, for example:\n"
        "  pip install onnxruntime"
    ) from exc

from parkour_moe_trt_common import DEFAULT_ASSET_DIR, DEFAULT_BUNDLE, load_bundle_models, load_manifest


class ErrorStats:
    def __init__(self):
        self.max_abs = 0.0
        self.sum_mean_abs = 0.0
        self.count = 0

    def update(self, actual, expected):
        diff = np.abs(actual - expected)
        self.max_abs = max(self.max_abs, float(diff.max()))
        self.sum_mean_abs += float(diff.mean())
        self.count += 1

    @property
    def mean_abs(self):
        return self.sum_mean_abs / max(self.count, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate current Go2 Parkour MoE ONNX assets against the PyTorch bundle."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="Path to parkour_moe_inference_bundle.pt.")
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ASSET_DIR, help="Directory containing estimator.onnx and actor.onnx.")
    parser.add_argument("--trials", type=int, default=20, help="Number of random validation rollouts.")
    parser.add_argument("--steps", type=int, default=50, help="Steps per rollout.")
    parser.add_argument("--refresh-every", type=int, default=5, help="Estimator refresh interval in steps.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Torch device for PT reference.")
    parser.add_argument("--atol", type=float, default=1.0e-4, help="Maximum allowed absolute error.")
    return parser.parse_args()


def append_vision_flag(mcp_code, vision_flag):
    if vision_flag.dim() == 1:
        vision_flag = vision_flag.unsqueeze(-1)
    return torch.cat((mcp_code, vision_flag.to(dtype=mcp_code.dtype, device=mcp_code.device)), dim=-1)


def main():
    args = parse_args()
    if args.refresh_every < 1:
        raise ValueError("--refresh-every must be >= 1")

    rng = np.random.default_rng(args.seed)
    bundle_path = args.bundle.expanduser().resolve()
    onnx_dir = args.onnx_dir.expanduser().resolve()
    manifest = load_manifest(onnx_dir)

    estimator_path = onnx_dir / manifest["files"]["estimator"]
    actor_path = onnx_dir / manifest["files"]["actor"]
    if not estimator_path.exists():
        raise FileNotFoundError(f"Missing estimator ONNX: {estimator_path}")
    if not actor_path.exists():
        raise FileNotFoundError(f"Missing actor ONNX: {actor_path}")

    depth_shape = manifest["inputs"]["estimator"]["depth"]
    depth_height = int(depth_shape[2])
    depth_width = int(depth_shape[3])
    device = torch.device(args.device)
    _bundle, actor_critic, estimator, meta = load_bundle_models(
        bundle_path,
        device=device,
        depth_height=depth_height,
        depth_width=depth_width,
    )

    estimator_session = ort.InferenceSession(str(estimator_path), providers=["CPUExecutionProvider"])
    actor_session = ort.InferenceSession(str(actor_path), providers=["CPUExecutionProvider"])

    stats = {
        "mcp_code": ErrorStats(),
        "shared_state": ErrorStats(),
        "m_hat": ErrorStats(),
        "gating_weights": ErrorStats(),
        "action": ErrorStats(),
    }

    with torch.inference_mode():
        for _trial in range(args.trials):
            history = np.zeros((meta.history_length, meta.num_obs), dtype=np.float32)
            shared_state_onnx = np.zeros((1, meta.shared_state_dim), dtype=np.float32)
            latest_mcp_pt = None
            latest_mcp_onnx = None
            estimator.reset()

            for step in range(args.steps):
                obs = rng.normal(0.0, 0.5, size=(meta.num_obs,)).astype(np.float32)
                depth = rng.normal(
                    0.0,
                    0.5,
                    size=(1, meta.depth_frame_count, meta.depth_height, meta.depth_width),
                ).astype(np.float32)
                vision_flag_np = np.array(
                    [[1.0 if (step // args.refresh_every) % 2 == 0 else 0.0]],
                    dtype=np.float32,
                )

                history = np.roll(history, shift=-1, axis=0)
                history[-1] = obs

                if step % args.refresh_every == 0 or latest_mcp_pt is None:
                    proprio_history_np = history.reshape(1, -1).astype(np.float32)
                    obs_np = obs.reshape(1, -1).astype(np.float32)
                    vision_flag_t = torch.from_numpy(vision_flag_np).to(device)

                    out_pt = estimator(
                        torch.from_numpy(proprio_history_np).to(device),
                        torch.from_numpy(depth).to(device),
                        vision_flag_t.reshape(-1) > 0.5,
                        obs_now=torch.from_numpy(obs_np).to(device),
                        update_state=True,
                    )
                    latest_mcp_pt = append_vision_flag(out_pt["mcp_code"], vision_flag_t).cpu().numpy()

                    out_onnx = estimator_session.run(
                        None,
                        {
                            "proprio_history": proprio_history_np,
                            "depth": depth,
                            "vision_flag": vision_flag_np,
                            "obs_now": obs_np,
                            "shared_state": shared_state_onnx,
                        },
                    )
                    latest_mcp_onnx, shared_state_onnx, m_hat_onnx, gating_onnx, _terrain_probs, _fall_prob = out_onnx

                    stats["mcp_code"].update(latest_mcp_onnx, latest_mcp_pt)
                    stats["shared_state"].update(shared_state_onnx, out_pt["shared_state"].cpu().numpy())
                    stats["m_hat"].update(m_hat_onnx, out_pt["m_hat"].cpu().numpy())
                    stats["gating_weights"].update(gating_onnx, out_pt["swav_gating_weights"].cpu().numpy())

                obs_np = obs.reshape(1, -1).astype(np.float32)
                action_pt = actor_critic.act_inference(
                    torch.from_numpy(latest_mcp_pt).to(device),
                    torch.from_numpy(obs_np).to(device),
                ).cpu().numpy()
                action_onnx = actor_session.run(
                    None,
                    {
                        "mcp_code": latest_mcp_onnx.astype(np.float32),
                        "obs": obs_np,
                    },
                )[0]
                stats["action"].update(action_onnx, action_pt)

    print("Validation summary")
    failed = False
    for name, stat in stats.items():
        print(f"{name}: max_abs={stat.max_abs:.8f}, mean_abs={stat.mean_abs:.8f}, samples={stat.count}")
        if stat.max_abs > args.atol:
            failed = True

    if failed:
        raise SystemExit(f"ONNX validation failed: at least one max_abs exceeded atol={args.atol}")
    print(f"ONNX validation passed with atol={args.atol}")


if __name__ == "__main__":
    main()
