#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from parkour_moe_trt_common import (
    ActorExportWrapper,
    DEFAULT_ASSET_DIR,
    DEFAULT_BUNDLE,
    EstimatorExportWrapper,
    disable_torch_export_fastpaths,
    load_bundle_models,
    make_manifest,
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export current Go2 Parkour MoE bundle to TensorRT-friendly ONNX assets."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="Path to parkour_moe_inference_bundle.pt.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ASSET_DIR, help="Directory for ONNX assets.")
    parser.add_argument("--device", default="cpu", help="Torch device used during export.")
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version. Default 16 avoids LayerNormalization op for TensorRT 8.5.x.",
    )
    parser.add_argument("--depth-height", type=int, default=58, help="Policy depth input height.")
    parser.add_argument("--depth-width", type=int, default=87, help="Policy depth input width.")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size for tracing. Use 1 for real robot.")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export dynamic batch axes. Not recommended for TensorRT/GRU deployment.",
    )
    return parser.parse_args()


def export_onnx(model, inputs, output_path, input_names, output_names, opset, dynamic_batch=False):
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: "batch"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        model,
        inputs,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def main():
    args = parse_args()
    bundle_path = args.bundle.expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    disable_torch_export_fastpaths()

    bundle, actor_critic, estimator, meta = load_bundle_models(
        bundle_path,
        device=args.device,
        depth_height=args.depth_height,
        depth_width=args.depth_width,
    )

    batch_size = int(args.batch_size)
    device = torch.device(args.device)
    proprio_history = torch.randn(batch_size, meta.history_length * meta.num_obs, device=device)
    depth = torch.randn(batch_size, meta.depth_frame_count, meta.depth_height, meta.depth_width, device=device)
    vision_flag = torch.ones(batch_size, 1, device=device)
    obs_now = torch.randn(batch_size, meta.num_obs, device=device)
    shared_state = torch.zeros(batch_size, meta.shared_state_dim, device=device)
    mcp_code = torch.randn(batch_size, meta.mcp_dim_with_flag, device=device)
    obs = torch.randn(batch_size, meta.num_obs, device=device)

    estimator_onnx = output_dir / "estimator.onnx"
    actor_onnx = output_dir / "actor.onnx"

    with torch.inference_mode():
        export_onnx(
            EstimatorExportWrapper(estimator),
            (proprio_history, depth, vision_flag, obs_now, shared_state),
            estimator_onnx,
            ["proprio_history", "depth", "vision_flag", "obs_now", "shared_state"],
            ["mcp_code", "shared_state_next", "m_hat", "gating_weights", "terrain_probs", "fall_recovery_prob"],
            args.opset,
            dynamic_batch=args.dynamic_batch,
        )
        export_onnx(
            ActorExportWrapper(actor_critic),
            (mcp_code, obs),
            actor_onnx,
            ["mcp_code", "obs"],
            ["action"],
            args.opset,
            dynamic_batch=args.dynamic_batch,
        )

    manifest = make_manifest(bundle_path, output_dir, bundle, meta, args.opset, batch_size)
    write_manifest(output_dir / "manifest.json", manifest)

    print(f"Bundle: {bundle_path}")
    print(f"Output: {output_dir}")
    print("Exported assets:")
    print(f"  - {estimator_onnx}")
    print(f"  - {actor_onnx}")
    print(f"  - {output_dir / 'manifest.json'}")
    print("No selector asset is exported: current Go2 framework feeds vision_flag directly into estimator.")


if __name__ == "__main__":
    main()
