#!/usr/bin/env python3
"""Export publication-friendly PNG plots from RM75 dynamics recordings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


TERRAINS = (
    ("flat", "平地", "flat_model30000.npz"),
    ("uphill_12deg", "12°上坡", "uphill_12deg_model30000.npz"),
    ("downhill_12deg", "12°下坡", "downhill_12deg_model30000.npz"),
)

METRICS = (
    ("joint_velocity", "关节速度", "joint_dof_velocities", "rad/s"),
    ("instantaneous_power", "关节瞬时功率", "joint_instantaneous_power", "W"),
    ("joint_torque", "关节力矩", "joint_torques", "N·m"),
    ("com_position", "整机质心位置", "system_com_position", "m"),
    ("com_velocity", "整机质心速度", "system_com_velocity", "m/s"),
)

LEG_COLORS = {
    "FL": "#0072B2",
    "FR": "#D55E00",
    "RL": "#009E73",
    "RR": "#CC79A7",
}
JOINT_STYLES = {"hip": "-", "thigh": "--", "calf": ":"}
COM_COLORS = ("#0072B2", "#D55E00", "#009E73")
COM_LABELS = ("x", "y", "z")
JOINT_PARTS = (("hip", "髋关节"), ("thigh", "大腿关节"), ("calf", "小腿关节"))


def configure_matplotlib() -> None:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            family = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
        }
    )


def load_recording(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def abbreviated_joint_name(name: str) -> str:
    return name.replace("_joint", "").replace("_", "-")


def plot_metric(
    ax,
    recording: dict[str, np.ndarray],
    metric_key: str,
    joint_part: Optional[str] = None,
) -> None:
    time = recording["time"]
    values = recording[metric_key]
    if metric_key.startswith("system_com"):
        for axis, label, color in zip(range(3), COM_LABELS, COM_COLORS):
            ax.plot(time, values[:, axis], color=color, linewidth=1.5, label=label)
    else:
        names = [str(name) for name in recording["joint_names"]]
        for index, name in enumerate(names):
            leg, joint, *_ = name.split("_")
            if joint_part is not None and joint != joint_part:
                continue
            ax.plot(
                time,
                values[:, index],
                color=LEG_COLORS[leg],
                linestyle="-",
                linewidth=1.35,
                alpha=0.95,
                label=leg,
            )
    ax.axhline(0.0, color="#555555", linewidth=0.7, alpha=0.55)
    ax.set_xlim(float(time[0]), float(time[-1]))


def add_legend(ax, metric_key: str, compact: bool = False) -> None:
    if metric_key.startswith("system_com"):
        ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    elif compact:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=4,
            frameon=False,
            fontsize=6.5,
            handlelength=2.5,
            columnspacing=0.9,
        )
    else:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=4,
            frameon=False,
            fontsize=8,
            handlelength=2.8,
            columnspacing=1.2,
        )


def save_individual_plots(
    output_dir: Path,
    terrain_slug: str,
    terrain_label: str,
    recording: dict[str, np.ndarray],
) -> None:
    terrain_dir = output_dir / terrain_slug
    terrain_dir.mkdir(parents=True, exist_ok=True)
    duration = float(recording["time"][-1] - recording["time"][0])
    index = 1
    for slug, label, key, unit in METRICS[:3]:
        for joint_part, joint_part_label in JOINT_PARTS:
            part_metric_label = joint_part_label + (label[2:] if label.startswith("关节") else label)
            fig, ax = plt.subplots(figsize=(13.0, 6.0), constrained_layout=False)
            plot_metric(ax, recording, key, joint_part=joint_part)
            ax.set_title(
                f"RM75 model_30000｜{terrain_label}｜{part_metric_label}",
                pad=12,
            )
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel(f"{label} ({unit})")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
            fig.text(
                0.99,
                0.012,
                f"四条曲线：FL 左前腿、FR 右前腿、RL 左后腿、RR 右后腿｜vx=1.0 m/s｜记录时长 {duration:.2f} s",
                ha="right",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
            fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.25)
            fig.savefig(terrain_dir / f"{index:02d}_{slug}_{joint_part}.png")
            plt.close(fig)
            index += 1

    for slug, label, key, unit in METRICS[3:]:
        fig, ax = plt.subplots(figsize=(13.0, 6.0), constrained_layout=False)
        plot_metric(ax, recording, key)
        ax.set_title(f"RM75 model_30000｜{terrain_label}｜{label}", pad=12)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel(f"{label} ({unit})")
        add_legend(ax, key)
        fig.text(
            0.99,
            0.012,
            f"速度指令 vx=1.0 m/s｜记录时长 {duration:.2f} s｜功率为有符号机械功率 τ·q̇",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.25)
        fig.savefig(terrain_dir / f"{index:02d}_{slug}.png")
        plt.close(fig)
        index += 1


def save_terrain_overview(
    output_dir: Path,
    terrain_slug: str,
    terrain_label: str,
    recording: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(15.0, 20.0), sharex=True)
    for ax, (_, label, key, unit) in zip(axes, METRICS):
        plot_metric(ax, recording, key)
        ax.set_ylabel(f"{label}\n({unit})")
        add_legend(ax, key, compact=True)
    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle(
        f"RM75 model_30000｜{terrain_label}动力学曲线｜vx=1.0 m/s",
        y=0.998,
        fontsize=16,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.975, bottom=0.04, hspace=0.58)
    fig.savefig(output_dir / f"{terrain_slug}_overview.png")
    plt.close(fig)


def save_comparison(output_dir: Path, recordings: list[tuple[str, dict[str, np.ndarray]]]) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(22.0, 22.0), squeeze=False)
    for row, (_, metric_label, key, unit) in enumerate(METRICS):
        row_min = min(float(np.nanmin(recording[key])) for _, recording in recordings)
        row_max = max(float(np.nanmax(recording[key])) for _, recording in recordings)
        padding = max((row_max - row_min) * 0.06, 1e-6)
        for column, (terrain_label, recording) in enumerate(recordings):
            ax = axes[row, column]
            plot_metric(ax, recording, key)
            ax.set_ylim(row_min - padding, row_max + padding)
            if row == 0:
                ax.set_title(terrain_label, fontsize=14, pad=10)
            if column == 0:
                ax.set_ylabel(f"{metric_label}\n({unit})")
            if row == len(METRICS) - 1:
                ax.set_xlabel("时间 (s)")
            add_legend(ax, key, compact=True)
    fig.suptitle(
        "RM75 model_30000｜平地、12°上坡与12°下坡动力学对比｜vx=1.0 m/s",
        y=0.998,
        fontsize=18,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.972, bottom=0.035, hspace=0.66, wspace=0.20)
    fig.savefig(output_dir / "all_terrain_comparison.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("action_data/model30000/RM75/report"),
        help="Directory containing the three segmented NPZ recordings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("action_data/model30000/RM75/figures_by_joint_group"),
        help="Directory that will receive PNG files.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for terrain_slug, terrain_label, filename in TERRAINS:
        recording = load_recording(args.input_dir / filename)
        save_individual_plots(args.output_dir, terrain_slug, terrain_label, recording)
    print(f"PNG plots saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
