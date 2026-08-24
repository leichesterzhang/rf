import os
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from legged_gym.utils.helpers import class_to_dict
from rsl_rl.algorithms import PPOParkourMoE
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticParkourMoE, DefaultEstimator


def numpy_representer(dumper, data):
    return dumper.represent_float(float(data))


def numpy_int_representer(dumper, data):
    return dumper.represent_int(int(data))


yaml.add_representer(np.float32, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.float64, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int32, numpy_int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int64, numpy_int_representer, Dumper=yaml.SafeDumper)


class OnPolicyRunnerParkourMoE:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg.get("estimator", {})
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCriticParkourMoE = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        estimator_ori = DefaultEstimator(
            input_dim=self.policy_cfg.get("num_actor_obs_now", self.env.num_obs),
            hidden_dims=self.estimator_cfg.get("hidden_dims", [128, 64]),
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPOParkourMoE = alg_class(
            actor_critic,
            estimator_ori=estimator_ori,
            estimator_cfg=self.estimator_cfg,
            device=self.device,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.obs_now_dim = self.env.num_obs
        self.history_length = self.estimator_cfg.get("history_length", 10)
        self.history_dim = self.obs_now_dim * self.history_length
        self.mcp_dim = self.policy_cfg.get("mcp_dim", 55)
        self.actor_mcp_dim = self.mcp_dim + 1
        self.camera_update_interval = getattr(self.env.cfg.camera, "update_interval", 5)
        self.vision_toggle_interval = self.cfg.get("vision_toggle_interval", 20)
        self.enable_timing = bool(self.cfg.get("enable_timing", False))
        self.timing_sync_cuda = bool(self.cfg.get("timing_sync_cuda", True))
        self.env.enable_timing = self.enable_timing
        self.env.timing_sync_cuda = self.timing_sync_cuda

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            [self.actor_mcp_dim],
        )
        self.history = torch.zeros(
            self.env.num_envs,
            self.history_length,
            self.obs_now_dim,
            device=self.device,
            dtype=torch.float,
        )

        self.mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.easy_idx_tensor = self._build_easy_terrain_indices(self.cfg.get("easy_terrain_names", []))
        self.omni_idx_tensor = self._build_omni_indices()
        self.omni_blind = bool(getattr(self.env.cfg.commands, "omni_blind", True))
        if self.omni_idx_tensor.numel() > 0:
            self.easy_idx_tensor = self._exclude_indices(self.easy_idx_tensor, self.omni_idx_tensor)
            self._apply_omni_vision_mask(self.mask)

        self.swav_window_length = self.estimator_cfg.get("terrain_window_length", 10)
        self.swav_gate_hist = torch.zeros(
            self.env.num_envs,
            self.swav_window_length - 1,
            self.estimator_cfg.get("expert_num", 3),
            device=self.device,
            dtype=torch.float,
        )
        self.swav_map_hist = torch.zeros(
            self.env.num_envs,
            self.swav_window_length - 1,
            self.estimator_cfg.get("terrain_map_dim", 187),
            device=self.device,
            dtype=torch.float,
        )
        self.swav_valid_hist = torch.zeros(
            self.env.num_envs,
            self.swav_window_length - 1,
            device=self.device,
            dtype=torch.bool,
        )

        self.latest_inference_mcp_code = None
        self.latest_inference_m_hat = None
        self.latest_inference_gating_weights = None
        self.latest_inference_token_dropout_mask = None
        self.latest_inference_token_grid_shape = None
        self.latest_inference_active_token_coords = None
        self.latest_inference_active_token_valid_mass = None

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
        if self.log_dir is not None and self.env.cfg.env.test is False:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            all_cfg = {"train_cfg": train_cfg, "env_cfg": class_to_dict(self.env.cfg)}
            yaml.safe_dump(all_cfg, open(os.path.join(self.log_dir, "config.yaml"), "w"))

    def _sync_timing_device(self):
        if not self.enable_timing or not self.timing_sync_cuda:
            return
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _timing_start(self):
        if not self.enable_timing:
            return None
        self._sync_timing_device()
        return time.perf_counter()

    def _timing_stop(self, start_time, timing_store, key):
        if start_time is None:
            return
        self._sync_timing_device()
        timing_store[key] += time.perf_counter() - start_time

    def _merge_env_timing(self, timing_store):
        if not self.enable_timing:
            return
        env_timing = getattr(self.env, "last_step_timing", None)
        if not env_timing:
            return
        for key, value in env_timing.items():
            timing_store[key] += float(value)

    def _build_easy_terrain_indices(self, terrain_names):
        if (
            not terrain_names
            or not hasattr(self.env, "terrain")
            or not hasattr(self.env, "terrain_types")
            or not hasattr(self.env.terrain, "name2cols")
        ):
            return torch.zeros(0, device=self.device, dtype=torch.long)
        env_mask = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        for terrain_name in terrain_names:
            cols = self.env.terrain.name2cols.get(terrain_name, None)
            if cols is None:
                continue
            if isinstance(cols, set):
                cols = torch.tensor(list(cols), device=self.device, dtype=torch.long)
            elif not isinstance(cols, torch.Tensor):
                cols = torch.tensor(cols, device=self.device, dtype=torch.long)
            else:
                cols = cols.to(self.device)
            if cols.numel() == 0:
                continue
            env_mask |= torch.isin(self.env.terrain_types.to(self.device), cols)
        return env_mask.nonzero(as_tuple=False).flatten()

    def _build_omni_indices(self):
        omni_mask = getattr(self.env, "omni_mask", None)
        if omni_mask is None:
            return torch.zeros(0, device=self.device, dtype=torch.long)
        omni_mask = omni_mask.to(self.device).bool()
        return omni_mask.nonzero(as_tuple=False).flatten()

    def _exclude_indices(self, indices, excluded_indices):
        if indices.numel() == 0 or excluded_indices.numel() == 0:
            return indices
        keep_mask = ~torch.isin(indices, excluded_indices)
        return indices[keep_mask]

    def _apply_omni_vision_mask(self, mask):
        if self.omni_blind and self.omni_idx_tensor.numel() > 0:
            mask[self.omni_idx_tensor] = False

    def _to_device_obs(self, obs):
        return obs.to(self.device)

    def _to_device_additional_obs(self, additional_obs):
        return {key: value.to(self.device) for key, value in additional_obs.items()}

    def _extract_obs_and_history(self, obs):
        return obs.clone(), self.history.flatten(1).clone()

    def _append_vision_flag_to_mcp(self, mcp_code, vision_flag):
        if vision_flag.dim() == 1:
            vision_flag = vision_flag.unsqueeze(-1)
        vision_flag = vision_flag.to(device=mcp_code.device, dtype=mcp_code.dtype).detach()
        return torch.cat((mcp_code.detach(), vision_flag), dim=-1)

    def _update_history(self, obs, dones=None):
        if dones is not None:
            done_ids = dones.view(-1).bool()
            if torch.any(done_ids):
                self.history[done_ids] = 0.0
        self.history = torch.cat((self.history[:, 1:], obs.unsqueeze(1)), dim=1)

    def _get_estimator_targets(self, critic_obs):
        if hasattr(self.env, "fall_recovery_privileged_slice"):
            fall_recovery_target = critic_obs[:, self.env.fall_recovery_privileged_slice].clone().detach()
        else:
            fall_recovery_target = -torch.ones(
                critic_obs.shape[0],
                1,
                device=critic_obs.device,
                dtype=critic_obs.dtype,
            )
        return (
            critic_obs[:, self.env.vt_privileged_slice].clone().detach(),
            critic_obs[:, self.env.ht_privileged_slice].clone().detach(),
            critic_obs[:, self.env.mt_privileged_slice].clone().detach(),
            critic_obs[:, self.env.terrain_id_privileged_slice].clone().detach(),
            fall_recovery_target,
        )

    def _build_swav_window(self, gate_weights, terrain_map, dones):
        current_valid = ~dones.view(-1).bool()
        gate_window = torch.cat((self.swav_gate_hist, gate_weights.unsqueeze(1)), dim=1)
        map_window = torch.cat((self.swav_map_hist, terrain_map.unsqueeze(1)), dim=1)
        valid_window = torch.cat((self.swav_valid_hist, current_valid.unsqueeze(1)), dim=1)
        self.swav_gate_hist = gate_window[:, 1:].detach()
        self.swav_map_hist = map_window[:, 1:].detach()
        self.swav_valid_hist = valid_window[:, 1:].detach()
        return gate_window, map_window, valid_window

    def _reset_swav_history(self, dones):
        done_ids = dones.view(-1).bool()
        if torch.any(done_ids):
            self.swav_gate_hist[done_ids] = 0.0
            self.swav_map_hist[done_ids] = 0.0
            self.swav_valid_hist[done_ids] = False

    def _build_random_token_dropout_mask(
        self,
        visual_token_data,
        mask_vision,
        dropout_min=None,
        dropout_max=None,
    ):
        image_height = int(visual_token_data.get("image_height", 0) or 0)
        image_width = int(visual_token_data.get("image_width", 0) or 0)
        tokens_per_frame = image_height * image_width
        frame_count = int(
            visual_token_data.get(
                "visual_frame_count",
                getattr(self.alg.estimator.core, "depth_frame_count", 1),
            )
            or 1
        )
        batch_size = mask_vision.shape[0]
        device = mask_vision.device
        dropout_mask = torch.zeros(
            batch_size,
            frame_count,
            tokens_per_frame,
            dtype=torch.bool,
            device=device,
        )
        if tokens_per_frame <= 0:
            return dropout_mask

        if dropout_min is None:
            dropout_min = getattr(self.alg.estimator.core, "token_dropout_min", 0.0)
        if dropout_max is None:
            dropout_max = getattr(self.alg.estimator.core, "token_dropout_max", 0.0)
        if hasattr(self.alg.estimator.core, "build_token_dropout_mask"):
            return self.alg.estimator.core.build_token_dropout_mask(
                batch_size,
                tokens_per_frame,
                mask_vision,
                device,
                dropout_min=dropout_min,
                dropout_max=dropout_max,
                frame_count=frame_count,
                image_height=image_height,
                image_width=image_width,
            )

        dropout_min = max(float(dropout_min), 0.0)
        dropout_max = min(max(float(dropout_max), dropout_min), 1.0)
        if dropout_max <= 0.0:
            return dropout_mask

        drop_ratio = torch.empty(batch_size, frame_count, device=device).uniform_(dropout_min, dropout_max)
        drop_count = torch.round(drop_ratio * tokens_per_frame).long().clamp(min=0, max=tokens_per_frame)
        random_rank = torch.rand(batch_size, frame_count, tokens_per_frame, device=device).argsort(dim=-1).argsort(dim=-1)
        dropout_mask = random_rank < drop_count.unsqueeze(-1)
        return dropout_mask & mask_vision.view(batch_size, 1, 1)

    def _aggregate_termination_metrics(self, ep_infos):
        if not ep_infos:
            return {}

        weighted_ep_infos = []
        for ep_info in ep_infos:
            reset_count = ep_info.get("term_reset_count", None)
            if reset_count is None:
                continue
            if isinstance(reset_count, torch.Tensor):
                reset_count = float(reset_count.detach().mean().item())
            else:
                reset_count = float(reset_count)
            weighted_ep_infos.append((ep_info, reset_count))

        if not weighted_ep_infos:
            return {}

        total_reset_count = sum(reset_count for _, reset_count in weighted_ep_infos)
        if total_reset_count <= 0.0:
            return {}

        termination_metrics = {
            "reset_count_total": total_reset_count,
            "reset_count_mean": total_reset_count / len(weighted_ep_infos),
        }

        frac_keys = sorted(
            {
                key
                for ep_info in ep_infos
                for key in ep_info.keys()
                if key.startswith("term_") and key.endswith("_frac")
            }
        )

        for key in frac_keys:
            weighted_sum = 0.0
            for ep_info, reset_count in weighted_ep_infos:
                value = ep_info.get(key, None)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    value = float(value.detach().mean().item())
                else:
                    value = float(value)
                weighted_sum += value * reset_count
            termination_metrics[key[5:]] = weighted_sum / total_reset_count

        return termination_metrics

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self._to_device_obs(self.env.get_observations())
        privileged_obs = self.env.get_privileged_observations()
        if privileged_obs is not None:
            privileged_obs = privileged_obs.to(self.device)
        additional_obs = self._to_device_additional_obs(self.env.get_additional_observations())
        self.history.zero_()
        self._update_history(obs)
        self.alg.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            if self.easy_idx_tensor.numel() > 0 and it % self.vision_toggle_interval == 0:
                self.mask[self.easy_idx_tensor] = ~self.mask[self.easy_idx_tensor]
            self._apply_omni_vision_mask(self.mask)

            estimator_metric_sums = {
                "ht_loss": 0.0,
                "mt_loss": 0.0,
                "vt_loss": 0.0,
                "reconstruction_loss": 0.0,
                "terrain_id_loss": 0.0,
                "terrain_id_acc": 0.0,
                "fall_recovery_loss": 0.0,
                "fall_recovery_acc": 0.0,
                "fall_recovery_prob_mean": 0.0,
                "local_patch_recon_loss": 0.0,
                "token_self_recon_loss": 0.0,
                "local_patch_recon_weighted_loss": 0.0,
                "z_kl_loss": 0.0,
                "z_kl_weight": 0.0,
                "z_kl_weighted_loss": 0.0,
                "z_mu_abs_mean": 0.0,
                "z_mu_std": 0.0,
                "z_logvar_abs_mean": 0.0,
                "active_latent_dims": 0.0,
                "active_latent_fraction": 0.0,
                "load_balance_loss": 0.0,
                "terrain_swav_loss": 0.0,
                "estimator_total_loss": 0.0,
                "terrain_swav_weight": 0.0,
            }
            estimator_metric_count = 0
            mcp_code = None
            collection_breakdown = defaultdict(float)
            env_step_breakdown = defaultdict(float)

            start = time.time()
            for _ in range(self.num_steps_per_env):
                critic_obs = privileged_obs if privileged_obs is not None else obs
                refresh_estimator = (
                    self.env.common_step_counter % self.camera_update_interval == 0 or mcp_code is None
                )

                if refresh_estimator:
                    timed_start = self._timing_start()
                    obs_now, proprio_hist = self._extract_obs_and_history(obs)
                    (
                        gt_vt_step,
                        gt_ht_step,
                        gt_mt_step,
                        gt_terrain_id_step,
                        gt_fall_recovery_step,
                    ) = self._get_estimator_targets(critic_obs)
                    camera_depth = additional_obs["depth_camera"].detach().clone()
                    mask_snapshot = self.mask.clone()
                    self._timing_stop(
                        timed_start,
                        collection_breakdown,
                        "runner_estimator_prepare",
                    )

                    timed_start = self._timing_start()
                    est_out = self.alg.estimator(
                        proprio_hist,
                        camera_depth,
                        mask_snapshot,
                        gt_mt_step=gt_mt_step,
                        obs_now=obs_now,
                    )
                    mcp_code = self._append_vision_flag_to_mcp(est_out["mcp_code"], mask_snapshot)
                    self._timing_stop(
                        timed_start,
                        collection_breakdown,
                        "runner_estimator_forward",
                    )

                timed_start = self._timing_start()
                with torch.inference_mode():
                    actions = self.alg.act(mcp_code, obs, critic_obs)
                self._timing_stop(
                    timed_start,
                    collection_breakdown,
                    "runner_policy_act",
                )

                timed_start = self._timing_start()
                obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                self._timing_stop(
                    timed_start,
                    collection_breakdown,
                    "runner_env_step",
                )
                self._merge_env_timing(env_step_breakdown)

                timed_start = self._timing_start()
                additional_obs = self._to_device_additional_obs(self.env.get_additional_observations())
                obs = self._to_device_obs(obs)
                if privileged_obs is not None:
                    privileged_obs = privileged_obs.to(self.device)
                critic_obs = privileged_obs if privileged_obs is not None else obs
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                self._timing_stop(
                    timed_start,
                    collection_breakdown,
                    "runner_obs_transfer",
                )

                timed_start = self._timing_start()
                self._update_history(obs, dones)
                self._timing_stop(
                    timed_start,
                    collection_breakdown,
                    "runner_history_update",
                )

                if refresh_estimator:
                    gt_next_step = obs.clone()
                    timed_start = self._timing_start()
                    swav_gate_window, swav_map_window, swav_valid_mask = self._build_swav_window(
                        est_out["swav_gating_weights"],
                        gt_mt_step,
                        dones,
                    )
                    self._timing_stop(
                        timed_start,
                        collection_breakdown,
                        "runner_swav_window",
                    )

                    timed_start = self._timing_start()
                    estimator_metrics = self.alg.update_estimator(
                        est_out,
                        gt_ht_step,
                        gt_mt_step,
                        gt_vt_step,
                        gt_terrain_id_step,
                        gt_fall_recovery_step,
                        gt_next_step,
                        swav_gate_window,
                        swav_map_window,
                        swav_valid_mask,
                    )
                    self._timing_stop(
                        timed_start,
                        collection_breakdown,
                        "runner_estimator_update",
                    )

                    for key, value in estimator_metrics.items():
                        estimator_metric_sums.setdefault(key, 0.0)
                        estimator_metric_sums[key] += float(value)
                    estimator_metric_count += 1
                    self.alg.estimator.detach_hidden_state()

                timed_start = self._timing_start()
                self.alg.process_env_step(rewards, dones, infos)
                self.alg.estimator.reset(dones)
                self._reset_swav_history(dones)
                self._timing_stop(
                    timed_start,
                    collection_breakdown,
                    "runner_process_env_step",
                )

                if self.log_dir is not None:
                    timed_start = self._timing_start()
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                    self._timing_stop(
                        timed_start,
                        collection_breakdown,
                        "runner_logging",
                    )

            stop = time.time()
            collection_time = stop - start
            start = stop
            self.alg.compute_returns(critic_obs, mcp_code, obs)
            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_entropy_loss,
                mean_actor_load_balance_loss,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            estimator_metrics = {}
            if estimator_metric_count > 0:
                estimator_metrics = {
                    key: value / estimator_metric_count
                    for key, value in estimator_metric_sums.items()
                }

            if self.log_dir is not None:
                self.log(
                    {
                        "it": it,
                        "num_learning_iterations": num_learning_iterations,
                        "ep_infos": ep_infos,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "collection_time": collection_time,
                        "learn_time": learn_time,
                        "mean_value_loss": mean_value_loss,
                        "mean_surrogate_loss": mean_surrogate_loss,
                        "mean_entropy_loss": mean_entropy_loss,
                        "mean_actor_load_balance_loss": mean_actor_load_balance_loss,
                        "estimator_metrics": estimator_metrics,
                        "collection_breakdown": dict(collection_breakdown),
                        "env_step_breakdown": dict(env_step_breakdown),
                    }
                )

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), it, False)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
            self.current_learning_iteration,
            True,
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_time += iteration_time

        ep_string = ""
        termination_metrics = self._aggregate_termination_metrics(locs["ep_infos"])
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                if key.startswith("term_"):
                    continue
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/entropy", locs["mean_entropy_loss"], locs["it"])
        self.writer.add_scalar("Loss/actor_load_balance", locs["mean_actor_load_balance_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        for key, value in locs.get("collection_breakdown", {}).items():
            self.writer.add_scalar(f"PerfBreakdown/collection/{key}", value, locs["it"])
            self.writer.add_scalar(
                f"PerfBreakdown/collection_share/{key}",
                value / max(locs["collection_time"], 1.0e-9),
                locs["it"],
            )
        env_step_total = locs.get("collection_breakdown", {}).get("runner_env_step", 0.0)
        for key, value in locs.get("env_step_breakdown", {}).items():
            self.writer.add_scalar(f"PerfBreakdown/env_step/{key}", value, locs["it"])
            if env_step_total > 0.0:
                self.writer.add_scalar(
                    f"PerfBreakdown/env_step_share/{key}",
                    value / env_step_total,
                    locs["it"],
                )

        for key, value in locs["estimator_metrics"].items():
            self.writer.add_scalar(f"Estimator/{key}", value, locs["it"])

        for key, value in termination_metrics.items():
            self.writer.add_scalar(f"Termination/{key}", value, locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )

        str_ = (
            f" \033[1m Learning iteration {locs['it']}/"
            f"{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        )
        log_string = (
            f"{'#' * width}\n"
            f"{str_.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
            f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
            f"{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"
            f"{'Actor load balance loss:':>{pad}} {locs['mean_actor_load_balance_loss']:.4f}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        for key, value in termination_metrics.items():
            log_string += f"{('term ' + key + ':'):>{pad}} {value:.4f}\n"
        for key, value in locs["estimator_metrics"].items():
            log_string += f"{key + ':':>{pad}} {value:.4f}\n"
        collection_breakdown = locs.get("collection_breakdown", {})
        if collection_breakdown:
            log_string += f"{'Collection breakdown:':>{pad}} \n"
            for key, value in sorted(collection_breakdown.items(), key=lambda item: item[1], reverse=True):
                ratio = 100.0 * value / max(locs["collection_time"], 1.0e-9)
                log_string += f"{key + ':':>{pad}} {value:.4f}s ({ratio:.1f}%)\n"
        env_step_breakdown = locs.get("env_step_breakdown", {})
        env_step_total = collection_breakdown.get("runner_env_step", 0.0)
        if env_step_breakdown and env_step_total > 0.0:
            log_string += f"{'Env step breakdown:':>{pad}} \n"
            for key, value in sorted(env_step_breakdown.items(), key=lambda item: item[1], reverse=True):
                ratio = 100.0 * value / env_step_total
                log_string += f"{key + ':':>{pad}} {value:.4f}s ({ratio:.1f}% env.step)\n"
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
        )
        print(log_string)

    def save(self, path, it, last_model, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "estimator_state_dict": self.alg.estimator.state_dict(),
                "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
                # Periodic saves happen before current_learning_iteration is
                # updated at the end of learn(), so persist the explicit loop
                # iteration passed by the caller.  Otherwise every checkpoint
                # in a long run incorrectly records iteration zero.
                "iter": it,
                "estimator_update_counter": self.alg.estimator_update_counter,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        estimator_load_result = self.alg.estimator.load_state_dict(
            loaded_dict["estimator_state_dict"],
            strict=False,
        )
        missing_estimator_keys = list(estimator_load_result.missing_keys)
        unexpected_estimator_keys = list(estimator_load_result.unexpected_keys)
        allowed_missing_prefixes = ("core.active_token_sampler.",)
        allowed_unexpected_prefixes = (
            "core.context_patch_decoder.",
            "core.active_token_sampler.",
        )
        disallowed_missing_keys = [
            key for key in missing_estimator_keys if not key.startswith(allowed_missing_prefixes)
        ]
        disallowed_unexpected_keys = [
            key for key in unexpected_estimator_keys if not key.startswith(allowed_unexpected_prefixes)
        ]
        if disallowed_missing_keys or disallowed_unexpected_keys:
            raise RuntimeError(
                "Failed to load estimator state_dict cleanly. "
                f"missing={missing_estimator_keys}, unexpected={unexpected_estimator_keys}"
            )
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if missing_estimator_keys or unexpected_estimator_keys:
                print(
                    "Estimator checkpoint differs from the current estimator module set; "
                    "skipping estimator optimizer state load for fine-tuning."
                )
            else:
                self.alg.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        self.alg.estimator_update_counter = loaded_dict.get("estimator_update_counter", 0)
        return loaded_dict["infos"]

    def get_inference_policy(
        self,
        device=None,
        random_token_dropout=False,
        token_dropout_min=None,
        token_dropout_max=None,
    ):
        self.alg.test_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
            self.alg.estimator.to(device)

        def policy(observations):
            obs = observations.to(self.device)
            reset_dones = self.env.reset_buf.to(self.device)
            if torch.any(reset_dones > 0):
                self.alg.estimator.reset(reset_dones)
                self._reset_swav_history(reset_dones)
                self.latest_inference_mcp_code = None
                self.latest_inference_m_hat = None
                self.latest_inference_gating_weights = None
                self.latest_inference_token_dropout_mask = None
                self.latest_inference_token_grid_shape = None
                self.latest_inference_active_token_coords = None
                self.latest_inference_active_token_valid_mass = None
            self._update_history(obs, reset_dones)

            additional_obs = self._to_device_additional_obs(self.env.get_additional_observations())
            refresh_estimator = (
                self.env.common_step_counter % self.camera_update_interval == 0
                or self.latest_inference_mcp_code is None
            )
            if refresh_estimator:
                obs_now, proprio_hist = self._extract_obs_and_history(obs)
                depth_camera = additional_obs["depth_camera"]
                #depth_camera = torch.rand_like(depth_camera)
                mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
                self._apply_omni_vision_mask(mask)
                with torch.no_grad():
                    visual_token_data = self.alg.estimator.encode_visual_tokens(depth_camera)
                    token_dropout_mask = None
                    if random_token_dropout:
                        token_dropout_mask = self._build_random_token_dropout_mask(
                            visual_token_data,
                            mask,
                            dropout_min=token_dropout_min,
                            dropout_max=token_dropout_max,
                        )
                    est_out = self.alg.estimator.forward_from_visual_tokens(
                        proprio_hist,
                        visual_token_data,
                        mask,
                        token_padding_mask=token_dropout_mask,
                        obs_now=obs_now,
                        apply_training_dropout=False,
                    )
                    self.latest_inference_mcp_code = self._append_vision_flag_to_mcp(est_out["mcp_code"], mask)
                    self.latest_inference_m_hat = est_out["m_hat"].detach()
                    self.latest_inference_gating_weights = est_out["swav_gating_weights"].detach()
                    self.latest_inference_token_dropout_mask = (
                        token_dropout_mask.detach() if token_dropout_mask is not None else None
                    )
                    self.latest_inference_token_grid_shape = (
                        int(est_out.get("image_height", 0) or 0),
                        int(est_out.get("image_width", 0) or 0),
                    )
                    active_token_coords = est_out.get("active_token_coords", None)
                    active_token_valid_mass = est_out.get("active_token_valid_mass", None)
                    self.latest_inference_active_token_coords = (
                        active_token_coords.detach() if torch.is_tensor(active_token_coords) else None
                    )
                    self.latest_inference_active_token_valid_mass = (
                        active_token_valid_mass.detach() if torch.is_tensor(active_token_valid_mass) else None
                    )
                # print(est_out["v_t"])
                # if(est_out["v_t"][0,0] < 0.1):
                #     print("Low velocity detected")
                #     self.alg.estimator.reset()
            return self.alg.actor_critic.act_inference(self.latest_inference_mcp_code, obs)

        return policy

    def export_policy(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "actor_critic_state_dict": self.alg.actor_critic.state_dict(),
                "estimator_state_dict": self.alg.estimator.state_dict(),
                "policy_cfg": self.policy_cfg,
                "estimator_cfg": self.estimator_cfg,
            },
            os.path.join(path, "parkour_moe_inference_bundle.pt"),
        )
