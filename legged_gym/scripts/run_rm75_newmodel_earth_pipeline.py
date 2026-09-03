"""Run the isolated V1.1 Earth flat-adaptation and visual-ramp pipeline."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "legged_gym" / "scripts" / "train.py"

VARIANTS = {
    "hip100_knee150": {
        "stage_a_task": "RM75_newmodel_earth_flat_hip100_knee150",
        "stage_a_experiment": "RM75_newmodel_earth_flat_hip100_knee150",
        "stage_b_task": "RM75_newmodel_earth_ramp35_hip100_knee150",
        "stage_b_experiment": "RM75_newmodel_earth_ramp35_hip100_knee150",
    },
    "hip84_knee150": {
        "stage_a_task": "RM75_newmodel_earth_flat_hip84_knee150",
        "stage_a_experiment": "RM75_newmodel_earth_flat_hip84_knee150",
        "stage_b_task": "RM75_newmodel_earth_ramp35_hip84_knee150",
        "stage_b_experiment": "RM75_newmodel_earth_ramp35_hip84_knee150",
    },
}


class Pipeline:
    def __init__(self, variant, physical_gpu):
        self.variant = variant
        self.cfg = VARIANTS[variant]
        self.logs = ROOT / "logs"
        self.control = self.logs / "RM75_newmodel_earth_pipeline" / variant
        self.control.mkdir(parents=True, exist_ok=True)
        self.state_path = self.control / "pipeline_state.json"
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        self.env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'rsl_rl'}"
        self.env["TORCH_EXTENSIONS_DIR"] = (
            f"/tmp/rm75_earth_newmodel_gpu{physical_gpu}_extensions"
        )
        self.env["LD_LIBRARY_PATH"] = (
            "/home/user/miniconda3/envs/unitree-rl/lib:"
            + self.env.get("LD_LIBRARY_PATH", "")
        )

    def update(self, **values):
        state = {}
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state.update(values)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def latest_complete_run(self, experiment, run_suffix, checkpoint):
        experiment_dir = self.logs / experiment
        if not experiment_dir.is_dir():
            return None
        candidates = sorted(
            (
                path
                for path in experiment_dir.iterdir()
                if path.is_dir()
                and path.name.endswith(run_suffix)
                and (path / f"model_{checkpoint}.pt").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
        )
        return candidates[-1] if candidates else None

    def train(self, task, run_name, log_name):
        command = [
            sys.executable,
            str(TRAIN),
            "--task",
            task,
            "--headless",
            "--sim_device",
            "cuda:0",
            "--rl_device",
            "cuda:0",
            "--run_name",
            run_name,
        ]
        log_path = self.control / log_name
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("COMMAND: " + " ".join(command) + "\n")
            log_file.flush()
            subprocess.run(
                command,
                cwd=ROOT,
                env=self.env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def run(self):
        stage_a_suffix = "stage_a_flat_adaptation_v2_zero_calibrated"
        stage_a_run = self.latest_complete_run(
            self.cfg["stage_a_experiment"], stage_a_suffix, 10000
        )
        if stage_a_run is None:
            self.update(status="stage_a_flat_adaptation")
            self.train(
                self.cfg["stage_a_task"],
                stage_a_suffix,
                "stage_a_train.log",
            )
            stage_a_run = self.latest_complete_run(
                self.cfg["stage_a_experiment"], stage_a_suffix, 10000
            )
        if stage_a_run is None:
            raise RuntimeError("Stage A ended without model_10000.pt")

        selected_dir = self.logs / self.cfg["stage_a_experiment"] / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected_model = selected_dir / "model.pt"
        shutil.copy2(stage_a_run / "model_10000.pt", selected_model)
        self.update(
            status="stage_b_visual_ramp",
            stage_a_run=stage_a_run.name,
            stage_a_checkpoint=10000,
            selected_model=str(selected_model),
        )

        stage_b_suffix = "stage_b_visual_ramp_v2_zero_calibrated"
        stage_b_run = self.latest_complete_run(
            self.cfg["stage_b_experiment"], stage_b_suffix, 30000
        )
        if stage_b_run is None:
            self.train(
                self.cfg["stage_b_task"],
                stage_b_suffix,
                "stage_b_train.log",
            )
            stage_b_run = self.latest_complete_run(
                self.cfg["stage_b_experiment"], stage_b_suffix, 30000
            )
        if stage_b_run is None:
            raise RuntimeError("Stage B ended without model_30000.pt")
        self.update(status="complete", stage_b_run=stage_b_run.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    Pipeline(args.variant, args.gpu).run()


if __name__ == "__main__":
    main()
