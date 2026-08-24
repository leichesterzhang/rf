import argparse
import sys
from typing import Any
from pathlib import Path

import numpy as np
import numpy.typing as npt

if not hasattr(npt, "NDArray"):
    class _NDArrayCompat:
        def __class_getitem__(cls, _item):
            return Any

    npt.NDArray = _NDArrayCompat

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_gym import LEGGED_GYM_ROOT_DIR


PLOT_GROUPS = [
    ("joint_dof_velocities", "Joint DOF Velocities"),
    ("joint_dof_accelerations", "Joint DOF Accelerations"),
    ("joint_linear_velocities", "Joint Linear Velocities"),
    ("joint_angular_velocities", "Joint Angular Velocities"),
    ("joint_linear_accelerations", "Joint Linear Accelerations"),
    ("joint_angular_accelerations", "Joint Angular Accelerations"),
    ("joint_torques", "Joint Torques"),
    ("base_linear_velocity", "Base Linear Velocity"),
    ("base_linear_acceleration", "Base Linear Acceleration"),
    ("base_angular_velocity", "Base Angular Velocity"),
    ("base_angular_acceleration", "Base Angular Acceleration"),
    ("base_com_position", "Base COM Position"),
]


def resolve_data_path(path_ref):
    path = Path(path_ref)
    if path.is_file():
        return path.resolve()
    candidate = Path(LEGGED_GYM_ROOT_DIR) / "action_data" / path_ref
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find action data file: {path_ref}")


def newest_data_file(robot_name=None):
    root = Path(LEGGED_GYM_ROOT_DIR) / "action_data"
    search_root = root / robot_name if robot_name else root
    files = sorted(search_root.rglob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .npz files found under {search_root}")
    return files[-1]


def to_text(value):
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array)


def component_labels(key, data, joint_names):
    values = data[key]
    if values.ndim == 2 and values.shape[1] == len(joint_names):
        return list(joint_names)
    if values.ndim == 2 and values.shape[1] == 3:
        return ["x", "y", "z"]
    if values.ndim == 3 and values.shape[1] == len(joint_names) and values.shape[2] == 3:
        axes = ["x", "y", "z"]
        return [f"{joint}_{axis}" for joint in joint_names for axis in axes]
    return [f"{key}_{index}" for index in range(int(np.prod(values.shape[1:])))]


def flatten_components(values):
    values = np.asarray(values)
    return values.reshape(values.shape[0], -1)


class ActionDataPlayer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = np.load(self.data_path, allow_pickle=False)
        self.times = np.asarray(self.data["time"], dtype=np.float32)
        self.index = 0
        self.playing = True
        self.joint_names = [str(name) for name in self.data["joint_names"]]

        ncols = 2
        nrows = int(np.ceil(len(PLOT_GROUPS) / ncols))
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(16, 12), sharex=True)
        self.axes = self.axes.reshape(-1)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.15, hspace=0.42)

        robot_name = to_text(self.data["robot_name"])
        terrain_name = to_text(self.data["terrain_name"])
        self.fig.suptitle(f"{robot_name} on {terrain_name}: {self.data_path.name}")

        self.time_markers = []
        for axis, (key, title) in zip(self.axes, PLOT_GROUPS):
            values = flatten_components(self.data[key])
            labels = component_labels(key, self.data, self.joint_names)
            for component_index in range(values.shape[-1]):
                axis.plot(self.times, values[:, component_index], linewidth=0.9, label=labels[component_index])
            marker = axis.axvline(self.times[0], color="black", linewidth=1.4)
            self.time_markers.append(marker)
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
            if len(labels) <= 12:
                axis.legend(loc="upper right", fontsize=7, ncols=2)

        for axis in self.axes[len(PLOT_GROUPS):]:
            axis.set_visible(False)

        self.axes[-1].set_xlabel("time [s]")
        self.slider_axis = self.fig.add_axes([0.12, 0.06, 0.68, 0.03])
        self.time_slider = Slider(
            self.slider_axis,
            "time",
            float(self.times[0]),
            float(self.times[-1]),
            valinit=float(self.times[0]),
        )
        self.time_slider.on_changed(self.on_slider_changed)

        self.button_axis = self.fig.add_axes([0.83, 0.045, 0.10, 0.055])
        self.play_button = Button(self.button_axis, "Pause")
        self.play_button.on_clicked(self.toggle_play)

        interval_ms = int(max(np.median(np.diff(self.times)) * 1000.0, 1.0)) if len(self.times) > 1 else 20
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

    def on_slider_changed(self, value):
        self.index = int(np.searchsorted(self.times, value, side="left"))
        self.index = int(np.clip(self.index, 0, len(self.times) - 1))
        self.set_time_marker(float(self.times[self.index]))

    def toggle_play(self, _event):
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")

    def set_time_marker(self, current_time):
        for marker in self.time_markers:
            marker.set_xdata([current_time, current_time])
        self.fig.canvas.draw_idle()

    def update(self, _frame):
        if not self.playing or len(self.times) == 0:
            return self.time_markers
        self.index = (self.index + 1) % len(self.times)
        current_time = float(self.times[self.index])
        self.time_slider.set_val(current_time)
        self.set_time_marker(current_time)
        return self.time_markers

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="Path to a saved .npz action data file.")
    parser.add_argument("--latest", action="store_true", help="Replay the newest saved action data file.")
    parser.add_argument("--robot", help="Robot folder to search when using --latest, such as b2, go2, or dog_502.")
    args = parser.parse_args()

    if args.latest:
        data_path = newest_data_file(args.robot)
    elif args.file:
        data_path = resolve_data_path(args.file)
    else:
        raise SystemExit("Provide a data file or use --latest.")

    print(f"Replaying action data: {data_path}")
    ActionDataPlayer(data_path).show()


if __name__ == "__main__":
    main()
