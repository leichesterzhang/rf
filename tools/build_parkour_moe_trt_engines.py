#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from parkour_moe_trt_common import DEFAULT_ASSET_DIR


MODEL_NAMES = ("estimator", "actor")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines for current Go2 Parkour MoE estimator/actor ONNX assets."
    )
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ASSET_DIR, help="Directory containing estimator.onnx and actor.onnx.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated engines. Defaults to --onnx-dir.")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "both"),
        default="both",
        help="Which precision variants to build.",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES), help="Subset of models to build.")
    parser.add_argument("--trtexec", type=Path, default=None, help="Optional explicit path to trtexec.")
    parser.add_argument("--timing-cache", type=Path, default=None, help="Optional timing cache path.")
    parser.add_argument(
        "--builder-opt-level",
        type=int,
        default=None,
        help="Optional TensorRT builder optimization level. Leave unset for TensorRT 8.5 compatibility.",
    )
    parser.add_argument("--run-inference", action="store_true", help="Let trtexec run inference once after building.")
    parser.add_argument("--verbose", action="store_true", help="Enable trtexec verbose output.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def resolve_trtexec(explicit_path: Path | None, allow_missing_explicit: bool = False) -> Path:
    candidates = []
    if explicit_path is not None:
        if allow_missing_explicit:
            return explicit_path
        candidates.append(explicit_path)
    elif allow_missing_explicit:
        return Path("trtexec")

    which_result = shutil.which("trtexec")
    if which_result is not None:
        candidates.append(Path(which_result))

    candidates.extend(
        (
            Path("/usr/src/tensorrt/bin/trtexec"),
            Path("/opt/tensorrt/bin/trtexec"),
            Path("/usr/local/bin/trtexec"),
        )
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    search_hint = "\n".join(f"  - {path}" for path in candidates if path is not None)
    raise FileNotFoundError(
        "Could not find trtexec. Install TensorRT on the target machine or pass --trtexec.\n"
        f"Searched:\n{search_hint}"
    )


def precision_variants(precision: str):
    if precision == "both":
        return ("fp32", "fp16")
    return (precision,)


def build_command(
    trtexec_path: Path,
    onnx_path: Path,
    engine_path: Path,
    timing_cache_path: Path,
    precision: str,
    builder_opt_level: int | None,
    run_inference: bool,
    verbose: bool,
):
    cmd = [
        str(trtexec_path),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--timingCacheFile={timing_cache_path}",
    ]
    if builder_opt_level is not None:
        cmd.append(f"--builderOptimizationLevel={builder_opt_level}")
    if precision == "fp16":
        cmd.append("--fp16")
    if not run_inference:
        cmd.append("--buildOnly")
    if verbose:
        cmd.append("--verbose")
    return cmd


def run_command(cmd, dry_run: bool):
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"trtexec failed with exit code {result.returncode}")


def main():
    args = parse_args()
    onnx_dir = args.onnx_dir.expanduser().resolve()
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    trtexec_path = resolve_trtexec(
        args.trtexec.expanduser().resolve() if args.trtexec is not None else None,
        allow_missing_explicit=args.dry_run,
    )
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else onnx_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_cache_path = (
        args.timing_cache.expanduser().resolve()
        if args.timing_cache is not None
        else output_dir / "parkour_moe.timing.cache"
    )

    print(f"trtexec: {trtexec_path}")
    print(f"ONNX dir: {onnx_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Timing cache: {timing_cache_path}")

    for model_name in args.models:
        onnx_path = onnx_dir / f"{model_name}.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX file: {onnx_path}")

        for precision in precision_variants(args.precision):
            engine_path = output_dir / f"{model_name}.{precision}.engine"
            cmd = build_command(
                trtexec_path=trtexec_path,
                onnx_path=onnx_path,
                engine_path=engine_path,
                timing_cache_path=timing_cache_path,
                precision=precision,
                builder_opt_level=args.builder_opt_level,
                run_inference=args.run_inference,
                verbose=args.verbose,
            )
            run_command(cmd, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
