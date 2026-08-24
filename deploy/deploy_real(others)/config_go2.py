from pathlib import Path
LEGGED_GYM_ROOT_DIR = str(Path(__file__).parents[2])
import numpy as np
import yaml


def resolve_repo_path(path_str):
    return str(path_str).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.joint2motor_idx = config["joint2motor_idx"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = resolve_repo_path(config["policy_path"])
            self.policy_type = config.get("policy_type", "auto")
            self.device = config.get("device", "cpu")
            self.engine_dir = resolve_repo_path(config["engine_dir"]) if config.get("engine_dir", "") else ""
            self.engine_precision = str(config.get("engine_precision", "auto"))
            self.engine_paths = {
                name: resolve_repo_path(path_str)
                for name, path_str in dict(config.get("engine_paths", {})).items()
            }

            self.kps = np.array(config["kps"],dtype=np.float32)
            self.kds = np.array(config["kds"],dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.obs_scales_ang_vel = config["obs_scales_ang_vel"]
            self.obs_scales_dof_pos = config["obs_scales_dof_pos"]
            self.obs_scales_dof_vel = config["obs_scales_dof_vel"]

            self.command_scale = config["command_scale"]
            self.action_scale = config["action_scale"]

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.cmd_init = np.array(config.get("cmd_init", [0.8, 0.0, 0.0]), dtype=np.float32)
            self.camera = dict(config.get("camera", {}))
            self.blind_walk = dict(config.get("blind_walk", {}))
