#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "rsl_rl"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from parkour_moe_trt_common import DEFAULT_BUNDLE, load_bundle_models


warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.*",
    category=UserWarning,
)


@dataclass
class BundleMeta:
    obs_dim: int
    num_actions: int
    critic_obs_dim: int
    history_length: int
    depth_frames: int
    depth_height: int
    depth_width: int
    shared_state_dim: int


@dataclass
class LatencyStats:
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    iters_per_sec: float
    iterations: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU/GPU inference latency for a Parkour MoE bundle."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Path to the exported parkour_moe_inference_bundle.pt.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda"],
        help="Devices to benchmark, for example: cpu cuda or cuda:0.",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations per benchmark.")
    parser.add_argument("--iters", type=int, default=200, help="Measured iterations per benchmark.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic inputs.")
    parser.add_argument(
        "--refresh-every",
        type=int,
        default=5,
        help="Simulate deployment cadence: 1 refresh step plus N-1 cached actor steps.",
    )
    parser.add_argument(
        "--vision",
        choices=["on", "off"],
        default="on",
        help="Whether the synthetic vision flag is enabled.",
    )
    parser.add_argument(
        "--depth-height",
        type=int,
        default=None,
        help="Depth input height. Defaults to 58, matching deployment config.",
    )
    parser.add_argument(
        "--depth-width",
        type=int,
        default=None,
        help="Depth input width. Defaults to 87, matching deployment config.",
    )
    parser.add_argument(
        "--depth-frames",
        type=int,
        default=None,
        help="Depth input channel count. Defaults to the first conv input channels in the estimator.",
    )
    parser.add_argument(
        "--obs-dim",
        "--num-obs",
        dest="obs_dim",
        type=int,
        default=None,
        help="Observation dimension fed into the estimator/actor. Defaults to bundle proprio_dim.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override torch CPU thread count before benchmarking.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for synthetic inputs.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to dump the benchmark report as JSON.",
    )
    return parser.parse_args()


def infer_bundle_meta(bundle: dict, args) -> BundleMeta:
    estimator_cfg = dict(bundle["estimator_cfg"])
    policy_cfg = dict(bundle["policy_cfg"])
    actor_state_dict = bundle["actor_critic_state_dict"]
    num_actions = int(actor_state_dict["std"].numel())
    actor_input_dim = int(actor_state_dict["actor_moe.experts.backbone.network.0.weight"].shape[1])
    mcp_dim = int(policy_cfg.get("mcp_dim", 68))
    num_actor_obs_now = int(policy_cfg.get("num_actor_obs_now", actor_input_dim - mcp_dim - 1))
    if "critic_experts.backbone.network.0.weight" in actor_state_dict:
        critic_obs_dim = int(actor_state_dict["critic_experts.backbone.network.0.weight"].shape[1] - 1)
    else:
        critic_obs_dim = int(actor_state_dict["critic.0.weight"].shape[1] - 1)
    depth_frames = int(args.depth_frames if args.depth_frames is not None else estimator_cfg.get("depth_frame_count", 1))

    return BundleMeta(
        obs_dim=int(args.obs_dim if args.obs_dim is not None else num_actor_obs_now),
        num_actions=num_actions,
        critic_obs_dim=critic_obs_dim,
        history_length=int(estimator_cfg.get("history_length", 10)),
        depth_frames=depth_frames,
        depth_height=int(args.depth_height if args.depth_height is not None else 58),
        depth_width=int(args.depth_width if args.depth_width is not None else 87),
        shared_state_dim=int(estimator_cfg.get("shared_state_dim", 64)),
    )


def resolve_devices(device_names: list[str]) -> list[torch.device]:
    devices = []
    for name in device_names:
        if name.startswith("cuda") and not torch.cuda.is_available():
            print(f"[skip] Requested {name}, but CUDA is not available.")
            continue
        try:
            devices.append(torch.device(name))
        except RuntimeError as exc:
            print(f"[skip] Invalid device '{name}': {exc}")
    if not devices:
        raise RuntimeError("No valid devices available to benchmark.")
    return devices


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_models(bundle_path: Path, meta: BundleMeta, device: torch.device):
    _bundle, actor_critic, estimator, _meta = load_bundle_models(
        bundle_path,
        device=device,
        depth_height=meta.depth_height,
        depth_width=meta.depth_width,
    )
    return actor_critic, estimator


def build_inputs(meta: BundleMeta, batch_size: int, use_vision: bool, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    obs_now = torch.randn(batch_size, meta.obs_dim, generator=generator, dtype=torch.float32)
    proprio_hist = torch.randn(
        batch_size,
        meta.history_length * meta.obs_dim,
        generator=generator,
        dtype=torch.float32,
    )
    depth_seq = torch.rand(
        batch_size,
        meta.depth_frames,
        meta.depth_height,
        meta.depth_width,
        generator=generator,
        dtype=torch.float32,
    ) - 0.5
    mask_vision = torch.full((batch_size,), use_vision, dtype=torch.bool)

    return (
        obs_now.to(device),
        proprio_hist.to(device),
        depth_seq.to(device),
        mask_vision.to(device),
    )


def append_vision_flag(mcp_code: torch.Tensor, vision_flag: torch.Tensor) -> torch.Tensor:
    if vision_flag.dim() == 1:
        vision_flag = vision_flag.unsqueeze(-1)
    vision_flag = vision_flag.to(device=mcp_code.device, dtype=mcp_code.dtype)
    return torch.cat((mcp_code, vision_flag), dim=-1)


def summarize_latencies(latencies_ms: list[float]) -> LatencyStats:
    samples = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(samples.mean())
    return LatencyStats(
        mean_ms=mean_ms,
        std_ms=float(samples.std(ddof=0)),
        min_ms=float(samples.min()),
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        max_ms=float(samples.max()),
        iters_per_sec=float(1000.0 / mean_ms) if mean_ms > 0.0 else float("inf"),
        iterations=int(samples.size),
    )


def benchmark_latency(
    fn: Callable[[], torch.Tensor | dict],
    device: torch.device,
    warmup: int,
    iters: int,
    prepare: Callable[[], None] | None = None,
) -> LatencyStats:
    if prepare is not None:
        prepare()
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync_device(device)

        latencies_ms: list[float] = []
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                start_event.record()
                fn()
                end_event.record()
                torch.cuda.synchronize(device)
                latencies_ms.append(float(start_event.elapsed_time(end_event)))
        else:
            for _ in range(iters):
                start_time = time.perf_counter()
                fn()
                end_time = time.perf_counter()
                latencies_ms.append((end_time - start_time) * 1000.0)
    return summarize_latencies(latencies_ms)


def format_stats(name: str, stats: LatencyStats) -> str:
    return (
        f"{name:<20} mean={stats.mean_ms:8.3f} ms  "
        f"p50={stats.p50_ms:8.3f} ms  "
        f"p95={stats.p95_ms:8.3f} ms  "
        f"max={stats.max_ms:8.3f} ms  "
        f"throughput={stats.iters_per_sec:8.2f} it/s"
    )


def benchmark_device(bundle_path: Path, meta: BundleMeta, args, device: torch.device) -> dict:
    actor_critic, estimator = build_models(bundle_path, meta, device)
    obs_now, proprio_hist, depth_seq, mask_vision = build_inputs(
        meta,
        args.batch_size,
        args.vision == "on",
        args.seed,
        device,
    )

    def estimator_only():
        return estimator(
            proprio_hist,
            depth_seq,
            mask_vision,
            obs_now=obs_now,
        )

    def refresh_step():
        est_out = estimator_only()
        mcp_code = append_vision_flag(est_out["mcp_code"], mask_vision)
        return actor_critic.act_inference(mcp_code, obs_now)

    with torch.inference_mode():
        estimator.reset()
        cached_est_out = estimator_only()
        cached_mcp_code = append_vision_flag(cached_est_out["mcp_code"], mask_vision)
        cached_action = actor_critic.act_inference(cached_mcp_code, obs_now)
        sync_device(device)

    def cached_actor_step():
        return actor_critic.act_inference(cached_mcp_code, obs_now)

    refresh_prepare = estimator.reset
    estimator_prepare = estimator.reset

    estimator_stats = benchmark_latency(
        estimator_only,
        device,
        args.warmup,
        args.iters,
        prepare=estimator_prepare,
    )
    actor_stats = benchmark_latency(
        cached_actor_step,
        device,
        args.warmup,
        args.iters,
    )
    refresh_stats = benchmark_latency(
        refresh_step,
        device,
        args.warmup,
        args.iters,
        prepare=refresh_prepare,
    )

    average_control_step_ms = (
        refresh_stats.mean_ms + max(args.refresh_every - 1, 0) * actor_stats.mean_ms
    ) / max(args.refresh_every, 1)

    device_name = "CPU"
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(cuda_index)

    result = {
        "device": str(device),
        "device_name": device_name,
        "estimator_only": asdict(estimator_stats),
        "cached_actor_step": asdict(actor_stats),
        "refresh_step": asdict(refresh_stats),
        "average_control_step_ms": average_control_step_ms,
        "refresh_every": int(args.refresh_every),
        "vision_enabled": bool(args.vision == "on"),
        "action_shape": list(cached_action.shape),
        "mcp_shape": list(cached_mcp_code.shape),
    }

    print(f"\n[{device}] {device_name}")
    print(format_stats("estimator_only", estimator_stats))
    print(format_stats("cached_actor_step", actor_stats))
    print(format_stats("refresh_step", refresh_stats))
    print(
        f"{'avg_control_step':<20} mean={average_control_step_ms:8.3f} ms  "
        f"(refresh_every={args.refresh_every})"
    )
    return result


def print_speedup_summary(results: dict[str, dict]):
    if "cpu" not in results:
        return

    cpu_result = results["cpu"]
    for device_name, device_result in results.items():
        if device_name == "cpu":
            continue

        print(f"\n[Speedup] {device_name} vs cpu")
        for key in ("estimator_only", "cached_actor_step", "refresh_step"):
            cpu_ms = float(cpu_result[key]["mean_ms"])
            dev_ms = float(device_result[key]["mean_ms"])
            speedup = cpu_ms / dev_ms if dev_ms > 0.0 else float("inf")
            print(f"{key:<20} {speedup:8.3f}x")

        cpu_avg = float(cpu_result["average_control_step_ms"])
        dev_avg = float(device_result["average_control_step_ms"])
        avg_speedup = cpu_avg / dev_avg if dev_avg > 0.0 else float("inf")
        print(f"{'avg_control_step':<20} {avg_speedup:8.3f}x")


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.refresh_every < 1:
        raise ValueError("--refresh-every must be >= 1")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("--warmup must be >= 0 and --iters must be >= 1")

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    bundle_path = args.bundle.resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    bundle = torch.load(bundle_path, map_location="cpu")
    meta = infer_bundle_meta(bundle, args)
    devices = resolve_devices(args.devices)

    print(f"Bundle: {bundle_path}")
    print(
        "Config: "
        f"obs_dim={meta.obs_dim}, num_actions={meta.num_actions}, "
        f"history_length={meta.history_length}, depth_shape=({meta.depth_frames}, {meta.depth_height}, {meta.depth_width}), "
        f"batch_size={args.batch_size}, vision={args.vision}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )
    if args.cpu_threads is not None:
        print(f"CPU threads: {torch.get_num_threads()}")

    results = {}
    for device in devices:
        result = benchmark_device(bundle_path, meta, args, device)
        results[str(device)] = result

    print_speedup_summary(results)

    report = {
        "bundle": str(bundle_path),
        "config": {
            "obs_dim": meta.obs_dim,
            "num_actions": meta.num_actions,
            "critic_obs_dim": meta.critic_obs_dim,
            "history_length": meta.history_length,
            "depth_frames": meta.depth_frames,
            "depth_height": meta.depth_height,
            "depth_width": meta.depth_width,
            "shared_state_dim": meta.shared_state_dim,
            "batch_size": args.batch_size,
            "vision": args.vision,
            "warmup": args.warmup,
            "iterations": args.iters,
            "refresh_every": args.refresh_every,
            "cpu_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "results": results,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, indent=2)
        print(f"\nJSON report written to: {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
