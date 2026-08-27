import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules import ParkourEstimator
from rsl_rl.storage import RolloutStorageParkourMoE


class PPOParkourMoE:
    def __init__(
        self,
        actor_critic,
        estimator_ori=None,
        estimator_cfg=None,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        estimator_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        z_kl_weight_max=1.0e-2,
        z_kl_warmup_start=0,
        z_kl_warmup_steps=5000,
        terrain_swav_weight_max=0.02,
        terrain_swav_warmup_start=2000,
        terrain_swav_warmup_steps=8000,
        terrain_id_loss_coef=1.0,
        fall_recovery_loss_coef=1.0,
        local_patch_recon_loss_coef=0.05,
        load_balance_coef=0.01,
        gate_smooth_coef=0.02,
        cycle_consistency_coef=0.01,
        vision_consistency_coef=0.02,
        device="cpu",
    ):
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.estimator_cfg = estimator_cfg or {}

        self.estimator = ParkourEstimator(**self.estimator_cfg).to(device)

        self.storage = None
        if hasattr(self.actor_critic, "optimizer_parameter_groups"):
            actor_critic_parameters = self.actor_critic.optimizer_parameter_groups(learning_rate)
        else:
            actor_critic_parameters = self.actor_critic.parameters()
        self.optimizer = optim.Adam(actor_critic_parameters, lr=learning_rate)
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_learning_rate)
        self.transition = RolloutStorageParkourMoE.Transition()

        self.desired_kl = desired_kl
        self.z_kl_weight_max = z_kl_weight_max
        self.z_kl_warmup_start = z_kl_warmup_start
        self.z_kl_warmup_steps = z_kl_warmup_steps
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.terrain_swav_weight_max = terrain_swav_weight_max
        self.terrain_swav_warmup_start = terrain_swav_warmup_start
        self.terrain_swav_warmup_steps = terrain_swav_warmup_steps
        self.terrain_id_loss_coef = terrain_id_loss_coef
        self.fall_recovery_loss_coef = fall_recovery_loss_coef
        self.local_patch_recon_loss_coef = local_patch_recon_loss_coef
        self.load_balance_coef = load_balance_coef
        self.estimator_update_counter = 0

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        mcp_shape,
    ):
        self.storage = RolloutStorageParkourMoE(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            mcp_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()
        self.estimator.eval()

    def train_mode(self):
        self.actor_critic.train()
        self.estimator.train()

    def act(self, mcp_code, obs, critic_obs):
        self.transition.actions = self.actor_critic.act(mcp_code, obs).detach()
        self.transition.values = self.actor_critic.evaluate(
            critic_obs,
            mcp_code=mcp_code,
            observations=obs,
        )[0].detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.mcp_code = mcp_code
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_mcp_code, last_obs):
        last_values = self.actor_critic.evaluate(
            last_critic_obs,
            mcp_code=last_mcp_code,
            observations=last_obs,
        )[0].detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy_loss = 0.0
        mean_actor_load_balance_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            mcp_code_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            self.actor_critic.act(mcp_code_batch, obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch, ac_weights = self.actor_critic.evaluate(
                critic_obs_batch,
                mcp_code=mcp_code_batch,
                observations=obs_batch,
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            mean_usage = torch.mean(ac_weights, dim=0)
            target_usage = torch.full_like(mean_usage, 1.0 / ac_weights.shape[1])
            actor_load_balance_loss = torch.mean((mean_usage - target_usage).pow(2))

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.load_balance_coef * actor_load_balance_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()
            mean_actor_load_balance_loss += actor_load_balance_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_actor_load_balance_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_actor_load_balance_loss

    def gaussian_kl(self, mu, logvar):
        return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)

    def get_terrain_swav_weight(self):
        step = self.estimator_update_counter
        if step < self.terrain_swav_warmup_start:
            return 0.0
        if step >= self.terrain_swav_warmup_start + self.terrain_swav_warmup_steps:
            return self.terrain_swav_weight_max
        return self.terrain_swav_weight_max * (
            (step - self.terrain_swav_warmup_start) / max(self.terrain_swav_warmup_steps, 1)
        )

    def get_z_kl_weight(self):
        step = self.estimator_update_counter
        if step < self.z_kl_warmup_start:
            return 0.0
        if step >= self.z_kl_warmup_start + self.z_kl_warmup_steps:
            return self.z_kl_weight_max
        return self.z_kl_weight_max * (
            (step - self.z_kl_warmup_start) / max(self.z_kl_warmup_steps, 1)
        )

    def update_estimator(
        self,
        est_out,
        gt_ht_buffer,
        gt_mt_buffer,
        gt_vt_buffer,
        gt_terrain_id_buffer,
        gt_fall_recovery_buffer,
        gt_next_obs_buffer,
        swav_gate_window=None,
        swav_map_window=None,
        swav_valid_mask=None,
        consistency_batch=None,
    ):
        vt_pred = est_out["v_t"]
        ht_pred = est_out["h_tf"]
        mt_pred = est_out["m_hat"]
        next_obs_pred = est_out["o_hat"]
        terrain_next_obs_pred = est_out.get("terrain_o_hat", next_obs_pred)
        z_mu = est_out["z_mu"]
        z_logvar = est_out["z_logvar"]
        terrain_logits = est_out["terrain_logits"]
        fall_recovery_logits = est_out["fall_recovery_logits"]
        gating_weights = est_out["swav_gating_weights"]
        local_patch_recon_loss = est_out["token_recon_loss"].mean()
        token_self_recon_loss = est_out.get("token_self_recon_loss", est_out["token_recon_loss"]).mean()

        ht_loss = F.mse_loss(ht_pred, gt_ht_buffer)
        mt_loss = F.mse_loss(mt_pred, gt_mt_buffer)
        vt_loss = F.mse_loss(vt_pred, gt_vt_buffer)
        motion_reconstruction_loss = F.mse_loss(next_obs_pred, gt_next_obs_buffer)
        terrain_reconstruction_loss = F.mse_loss(terrain_next_obs_pred, gt_next_obs_buffer)
        reconstruction_loss = 0.5 * (motion_reconstruction_loss + terrain_reconstruction_loss)
        terrain_targets = gt_terrain_id_buffer.view(-1).long()
        valid_terrain_mask = terrain_targets >= 0
        if torch.any(valid_terrain_mask):
            terrain_id_loss = F.cross_entropy(
                terrain_logits[valid_terrain_mask],
                terrain_targets[valid_terrain_mask],
            )
        else:
            terrain_id_loss = torch.zeros((), device=terrain_logits.device, dtype=terrain_logits.dtype)
        fall_recovery_targets = gt_fall_recovery_buffer.to(
            device=fall_recovery_logits.device,
            dtype=fall_recovery_logits.dtype,
        ).reshape_as(fall_recovery_logits)
        valid_fall_recovery_mask = fall_recovery_targets >= 0.0
        if torch.any(valid_fall_recovery_mask):
            fall_recovery_loss = F.binary_cross_entropy_with_logits(
                fall_recovery_logits[valid_fall_recovery_mask],
                fall_recovery_targets[valid_fall_recovery_mask],
            )
        else:
            fall_recovery_loss = torch.zeros(
                (),
                device=fall_recovery_logits.device,
                dtype=fall_recovery_logits.dtype,
            )
        z_kl_loss = self.gaussian_kl(z_mu, z_logvar).mean()
        z_kl_weight = self.get_z_kl_weight()
        z_kl_weighted_loss = z_kl_weight * z_kl_loss

        with torch.no_grad():
            z_mu_abs_mean = z_mu.abs().mean()
            z_mu_std = z_mu.std(unbiased=False)
            z_logvar_abs_mean = z_logvar.abs().mean()
            active_latent_mask = z_mu.std(dim=0, unbiased=False) > 1.0e-2
            active_latent_dims = active_latent_mask.float().sum()
            active_latent_fraction = active_latent_mask.float().mean()
            if torch.any(valid_terrain_mask):
                terrain_id_acc = (
                    terrain_logits.argmax(dim=-1)[valid_terrain_mask]
                    == terrain_targets[valid_terrain_mask]
                ).float().mean()
            else:
                terrain_id_acc = torch.zeros((), device=terrain_logits.device, dtype=terrain_logits.dtype)
            if torch.any(valid_fall_recovery_mask):
                fall_recovery_pred = (torch.sigmoid(fall_recovery_logits) >= 0.5).to(fall_recovery_targets.dtype)
                fall_recovery_acc = (
                    fall_recovery_pred[valid_fall_recovery_mask]
                    == fall_recovery_targets[valid_fall_recovery_mask]
                ).float().mean()
                fall_recovery_prob_mean = torch.sigmoid(
                    fall_recovery_logits[valid_fall_recovery_mask]
                ).mean()
            else:
                fall_recovery_acc = torch.zeros(
                    (),
                    device=fall_recovery_logits.device,
                    dtype=fall_recovery_logits.dtype,
                )
                fall_recovery_prob_mean = torch.zeros(
                    (),
                    device=fall_recovery_logits.device,
                    dtype=fall_recovery_logits.dtype,
                )

        mean_usage = torch.mean(gating_weights, dim=0)
        target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
        load_balance_loss = torch.mean((mean_usage - target_usage).pow(2))

        terrain_swav_loss = self.estimator.cross_view_swav(
            swav_gate_window,
            swav_map_window,
            swav_valid_mask,
        )
        terrain_swav_weight = self.get_terrain_swav_weight()

        total_loss = (
            ht_loss
            + mt_loss
            + vt_loss
            + reconstruction_loss
            + self.terrain_id_loss_coef * terrain_id_loss
            + self.fall_recovery_loss_coef * fall_recovery_loss
            + self.local_patch_recon_loss_coef * local_patch_recon_loss
            + z_kl_weighted_loss
            + terrain_swav_weight * terrain_swav_loss
        )

        self.estimator_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
        self.estimator_optimizer.step()
        self.estimator_update_counter += 1

        return {
            "ht_loss": ht_loss.item(),
            "mt_loss": mt_loss.item(),
            "vt_loss": vt_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "motion_reconstruction_loss": motion_reconstruction_loss.item(),
            "terrain_reconstruction_loss": terrain_reconstruction_loss.item(),
            "terrain_id_loss": terrain_id_loss.item(),
            "terrain_id_acc": terrain_id_acc.item(),
            "fall_recovery_loss": fall_recovery_loss.item(),
            "fall_recovery_acc": fall_recovery_acc.item(),
            "fall_recovery_prob_mean": fall_recovery_prob_mean.item(),
            "local_patch_recon_loss": local_patch_recon_loss.item(),
            "token_self_recon_loss": token_self_recon_loss.item(),
            "local_patch_recon_weighted_loss": (
                self.local_patch_recon_loss_coef * local_patch_recon_loss
            ).item(),
            "z_kl_loss": z_kl_loss.item(),
            "z_kl_weight": z_kl_weight,
            "z_kl_weighted_loss": z_kl_weighted_loss.item(),
            "z_mu_abs_mean": z_mu_abs_mean.item(),
            "z_mu_std": z_mu_std.item(),
            "z_logvar_abs_mean": z_logvar_abs_mean.item(),
            "active_latent_dims": active_latent_dims.item(),
            "active_latent_fraction": active_latent_fraction.item(),
            "load_balance_loss": load_balance_loss.item(),
            "terrain_swav_loss": terrain_swav_loss.item() if torch.is_tensor(terrain_swav_loss) else float(terrain_swav_loss),
            "terrain_swav_weight": terrain_swav_weight,
            "estimator_total_loss": total_loss.item(),
        }
