#!/usr/bin/env python3
"""Build the repeated RM75 ramp as the same quantized height grid used in training."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np


HORIZONTAL_SCALE = 0.05
VERTICAL_SCALE = 0.005
CELL_LENGTH = 12.0
TRACK_WIDTH = 9.0
RAMP_LENGTH = 2.0
PLATFORM_LENGTH = 2.0
SLOPE_DEGREES = 35.0
DEFAULT_REPEATS = 20


def build_profile(repeats: int) -> np.ndarray:
    """Reproduce the int16 assignment/truncation in the Isaac Gym generator."""
    samples_per_cell = int(CELL_LENGTH / HORIZONTAL_SCALE)
    platform_samples = int(PLATFORM_LENGTH / HORIZONTAL_SCALE)
    ramp_samples = int(RAMP_LENGTH / HORIZONTAL_SCALE)
    profile_raw = np.zeros(repeats * samples_per_cell + 1, dtype=np.int16)
    slope_strength = math.tan(math.radians(SLOPE_DEGREES)) * (
        HORIZONTAL_SCALE / VERTICAL_SCALE
    )
    max_height = slope_strength * ramp_samples

    for cell in range(repeats):
        start = cell * samples_per_cell
        uphill = start + platform_samples
        summit = uphill + ramp_samples
        downhill = summit + platform_samples
        finish = downhill + ramp_samples
        profile_raw[uphill:summit] = (
            slope_strength * np.arange(ramp_samples)
        ).astype(np.int16)
        profile_raw[summit:downhill] = np.int16(max_height)
        profile_raw[downhill:finish] = (
            max_height - slope_strength * np.arange(ramp_samples)
        ).astype(np.int16)

    height_m = profile_raw.astype(np.float32) * VERTICAL_SCALE
    return height_m


def write_mujoco_hfield(path: Path, repeats: int) -> tuple[int, int, float]:
    profile = build_profile(repeats)
    rows = int(TRACK_WIDTH / HORIZONTAL_SCALE) + 1
    cols = profile.size
    top = float(profile.max())
    normalized = np.broadcast_to(profile / top, (rows, cols)).astype("<f4", copy=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<ii", rows, cols))
        stream.write(normalized.tobytes(order="C"))
    return rows, cols, top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/robots/RM75/visual_ramp_35deg_repeated.bin"),
    )
    args = parser.parse_args()
    rows, cols, top = write_mujoco_hfield(args.output, args.repeats)
    print(
        f"wrote {args.output}: rows={rows}, cols={cols}, "
        f"length={(cols - 1) * HORIZONTAL_SCALE:.3f} m, top={top:.3f} m"
    )


if __name__ == "__main__":
    main()
