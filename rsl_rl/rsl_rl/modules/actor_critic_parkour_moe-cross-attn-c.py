import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.modules.utils import get_activation


def build_sincos_1d_encoding(length, dim, device, dtype, positions=None):
    if dim <= 0:
        return torch.zeros(length, 0, device=device, dtype=dtype)

    if positions is None:
        positions = torch.arange(length, device=device, dtype=torch.float32)
    else:
        positions = positions.to(device=device, dtype=torch.float32)

    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros(length, dim, device=device, dtype=dtype)

    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=device, dtype=torch.float32)
        / max(half_dim - 1, 1)
    )
    phase = positions.unsqueeze(-1) * frequencies.unsqueeze(0)
    encoding = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if dim % 2 == 1:
        encoding = F.pad(encoding, (0, 1))
    return encoding[:, :dim].to(dtype=dtype)


def build_sincos_2d_encoding(height, width, dim, device, dtype):
    row_dim = dim // 2
    col_dim = dim - row_dim
    row_encoding = build_sincos_1d_encoding(height, row_dim, device, dtype)
    col_encoding = build_sincos_1d_encoding(width, col_dim, device, dtype)
    row_encoding = row_encoding.unsqueeze(1).expand(height, width, row_dim)
    col_encoding = col_encoding.unsqueeze(0).expand(height, width, col_dim)
    return torch.cat((row_encoding, col_encoding), dim=-1).reshape(height * width, dim)


class DefaultEstimator(nn.Module):
    def __init__(self, input_dim=45, hidden_dims=None, output_dim=9):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ParkourEstimatorCore(nn.Module):
    def __init__(self, proprio_dim=45, history_length=10, token_dim=64, cross_attn_nhead=4):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.history_length = history_length
        self.token_dim = token_dim

        self.image_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.image_token_norm = nn.LayerNorm(token_dim)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, token_dim),
        )
        self.proprio_token_norm = nn.LayerNorm(token_dim)
        self.terrain_token_norm = nn.LayerNorm(token_dim)
        self.proprio_to_visual_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=cross_attn_nhead,
            batch_first=True,
        )

    def build_token_encodings(self, image_height, image_width, device, dtype):
        modality_positions = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=torch.float32)
        modality_encoding = build_sincos_1d_encoding(
            length=modality_positions.numel(),
            dim=self.token_dim,
            device=device,
            dtype=dtype,
            positions=modality_positions,
        )

        image_type_encoding = modality_encoding[0].view(1, 1, -1)
        proprio_type_encoding = modality_encoding[1].view(1, 1, -1)
        flag_type_encoding = modality_encoding[2].view(1, 1, -1)
        terrain_type_encoding = modality_encoding[3].view(1, 1, -1)

        image_position_encoding = build_sincos_2d_encoding(
            image_height,
            image_width,
            self.token_dim,
            device,
            dtype,
        ).unsqueeze(0)
        proprio_position_encoding = build_sincos_1d_encoding(
            self.history_length,
            self.token_dim,
            device,
            dtype,
        ).unsqueeze(0)
        aux_position_encoding = build_sincos_1d_encoding(
            2,
            self.token_dim,
            device,
            dtype,
        ).unsqueeze(0)
        return (
            image_type_encoding,
            image_position_encoding,
            proprio_type_encoding,
            proprio_position_encoding,
            flag_type_encoding,
            terrain_type_encoding,
            aux_position_encoding[:, 0:1, :],
            aux_position_encoding[:, 1:2, :],
        )

    def encode_modalities(self, proprio_seq, depth_seq, mask_vision):
        batch_size = proprio_seq.shape[0]
        mask_vision = mask_vision.bool()

        image_features = self.image_encoder(depth_seq)
        image_height, image_width = image_features.shape[-2:]
        image_tokens = image_features.flatten(2).transpose(1, 2)
        image_tokens = self.image_token_norm(image_tokens)

        proprio_tokens = self.proprio_encoder(
            proprio_seq.reshape(batch_size, self.history_length, self.proprio_dim)
        )
        proprio_tokens = self.proprio_token_norm(proprio_tokens)
        vision_token = mask_vision.float().view(batch_size, 1, 1)
        return (
            image_tokens,
            proprio_tokens,
            vision_token,
            mask_vision,
            image_height,
            image_width,
        )

    def prepare_tokens(
        self,
        image_tokens,
        proprio_tokens,
        vision_token,
        mask_vision,
        image_height,
        image_width,
    ):
        batch_size = image_tokens.shape[0]
        (
            image_type_encoding,
            image_position_encoding,
            proprio_type_encoding,
            proprio_position_encoding,
            flag_type_encoding,
            terrain_type_encoding,
            flag_position_encoding,
            terrain_position_encoding,
        ) = self.build_token_encodings(
            image_height,
            image_width,
            image_tokens.device,
            image_tokens.dtype,
        )

        image_tokens = (
            image_tokens + image_type_encoding + image_position_encoding
        ) * mask_vision.float().view(batch_size, 1, 1)
        proprio_tokens = proprio_tokens + proprio_type_encoding + proprio_position_encoding
        # Keep the flag token zero when vision is invalid so it only conditions
        # the proprio queries when depth should participate.
        vision_token = mask_vision.float().view(batch_size, 1, 1) * (
            vision_token + flag_type_encoding + flag_position_encoding
        )
        image_padding_mask = (~mask_vision).unsqueeze(1).expand(batch_size, image_tokens.shape[1])
        return image_tokens, proprio_tokens, vision_token, image_padding_mask

    def cross_attend_proprio_to_visual(
        self,
        image_tokens,
        proprio_tokens,
        vision_token,
        image_padding_mask,
        mask_vision,
    ):
        batch_size = proprio_tokens.shape[0]
        query_tokens = proprio_tokens + vision_token.expand(-1, proprio_tokens.shape[1], -1)
        attended_tokens = torch.zeros_like(proprio_tokens)
        attn_weights = torch.zeros(
            batch_size,
            proprio_tokens.shape[1],
            image_tokens.shape[1],
            device=proprio_tokens.device,
            dtype=proprio_tokens.dtype,
        )
        valid_mask = mask_vision.bool()
        if torch.any(valid_mask):
            valid_attended, valid_weights = self.proprio_to_visual_attn(
                query=query_tokens[valid_mask],
                key=image_tokens[valid_mask],
                value=image_tokens[valid_mask],
                key_padding_mask=image_padding_mask[valid_mask],
                need_weights=True,
            )
            attended_tokens[valid_mask] = valid_attended
            attn_weights[valid_mask] = valid_weights

        fused_tokens = proprio_tokens + mask_vision.float().view(batch_size, 1, 1) * attended_tokens
        return fused_tokens, attn_weights

    def forward(self, proprio_seq, depth_seq, mask_vision, terrain_token=None):
        del terrain_token
        prepared_tokens = self.prepare_tokens(
            *self.encode_modalities(proprio_seq, depth_seq, mask_vision),
        )
        return self.cross_attend_proprio_to_visual(
            *prepared_tokens,
            mask_vision=mask_vision.bool(),
        )


class ReadWriteExpert(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, shared_state_dim=64):
        super().__init__()
        self.read = nn.Sequential(
            nn.Linear(shared_state_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.write = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, shared_state_dim),
        )

    def forward(self, expert_feature, shared_state):
        hidden_init = self.read(shared_state)
        hidden_state = self.gru_cell(expert_feature, hidden_init)
        shared_delta = self.write(hidden_state)
        return hidden_state, shared_delta


class ParkourEstimator(nn.Module):
    def __init__(
        self,
        expert_num=4,
        proprio_dim=45,
        history_length=10,
        vision_token_dim=64,
        shared_state_dim=None,
        expert_hidden_dim=None,
        latent_dim=16,
        terrain_latent_dim=32,
        terrain_map_dim=187,
        feet_height_dim=4,
        num_terrain_types=12,
        transformer_num_layers=1,
        transformer_nhead=4,
        cross_attn_nhead=None,
        transformer_ff_dim=256,
        terrain_window_length=10,
        shared_state_alpha=0.5,
        swav_temperature=0.1,
        swav_epsilon=0.05,
        swav_sinkhorn_iters=3,
        swav_num_prototypes=16,
        **kwargs,
    ):
        super().__init__()
        self.expert_num = expert_num
        self.num_terrain_types = num_terrain_types
        self.terrain_window_length = terrain_window_length
        self.shared_state_dim = vision_token_dim if shared_state_dim is None else shared_state_dim
        self.expert_hidden_dim = vision_token_dim if expert_hidden_dim is None else expert_hidden_dim
        self.shared_state_alpha = shared_state_alpha
        self.swav_temperature = swav_temperature
        self.swav_epsilon = swav_epsilon
        self.swav_sinkhorn_iters = swav_sinkhorn_iters
        self.swav_num_prototypes = swav_num_prototypes
        if cross_attn_nhead is None:
            cross_attn_nhead = transformer_nhead

        self.core = ParkourEstimatorCore(
            proprio_dim=proprio_dim,
            history_length=history_length,
            token_dim=vision_token_dim,
            cross_attn_nhead=cross_attn_nhead,
        )
        self.shared_pool_query = nn.Parameter(torch.empty(1, vision_token_dim))
        self.expert_pool_queries = nn.Parameter(torch.empty(expert_num, vision_token_dim))
        nn.init.normal_(self.shared_pool_query, std=vision_token_dim**-0.5)
        nn.init.normal_(self.expert_pool_queries, std=vision_token_dim**-0.5)
        self.shared_head = nn.Sequential(
            nn.Linear(vision_token_dim, vision_token_dim),
            nn.ELU(),
        )
        self.selector_head = nn.Sequential(
            nn.Linear(vision_token_dim, vision_token_dim),
            nn.ELU(),
            nn.Linear(vision_token_dim, num_terrain_types),
        )
        self.fall_recovery_head = nn.Sequential(
            nn.Linear(vision_token_dim, vision_token_dim),
            nn.ELU(),
            nn.Linear(vision_token_dim, 1),
        )
        self.terrain_token_encoder = nn.Sequential(
            nn.Linear(num_terrain_types, vision_token_dim),
            nn.ELU(),
            nn.Linear(vision_token_dim, vision_token_dim),
        )
        self.expert_fall_recovery_encoder = nn.Sequential(
            nn.Linear(1, vision_token_dim),
            nn.ELU(),
            nn.Linear(vision_token_dim, vision_token_dim),
        )
        self.router_terrain_encoder = nn.Sequential(
            nn.Linear(num_terrain_types, 32),
            nn.ELU(),
            nn.Linear(32, 32),
        )
        self.router_fall_recovery_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ELU(),
            nn.Linear(16, 16),
        )
        self.router = nn.Sequential(
            nn.Linear(proprio_dim + 1 + vision_token_dim + 32 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, expert_num),
            nn.Softmax(dim=-1),
        )
        self.experts = nn.ModuleList(
            [
                ReadWriteExpert(
                    input_size=vision_token_dim,
                    hidden_size=self.expert_hidden_dim,
                    shared_state_dim=self.shared_state_dim,
                )
                for _ in range(expert_num)
            ]
        )
        self.shared_state_updater = nn.GRUCell(
            input_size=self.shared_state_dim,
            hidden_size=self.shared_state_dim,
        )
        self.shared_state_norm = nn.LayerNorm(self.shared_state_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.shared_state_dim, vision_token_dim),
            nn.ELU(),
        )

        self.fc_v_t = nn.Linear(vision_token_dim, 3)
        self.fc_h_tf = nn.Linear(vision_token_dim, feet_height_dim)
        self.fc_z_mu = nn.Linear(vision_token_dim, latent_dim)
        self.fc_z_logvar = nn.Linear(vision_token_dim, latent_dim)
        self.fc_z_tm = nn.Linear(vision_token_dim, terrain_latent_dim)

        self.decoder_obs = nn.Sequential(
            nn.Linear(3 + feet_height_dim + latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, proprio_dim),
        )
        self.decoder_map = nn.Sequential(
            nn.Linear(terrain_latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, terrain_map_dim),
        )

        self.gate_terrain_projector = nn.Sequential(
            nn.Linear(expert_num, 64),
            nn.ELU(),
            nn.Linear(64, 32),
        )
        self.elevation_terrain_projector = nn.Sequential(
            nn.Linear(terrain_map_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
        )
        self.terrain_prototypes = nn.Linear(32, swav_num_prototypes, bias=False)

        self.fc_zt = nn.Linear(latent_dim, latent_dim)
        self.map_unet = nn.Identity()
        self.fc_zm_fine = nn.Linear(terrain_latent_dim, terrain_latent_dim)
        self.register_buffer("shared_state", torch.zeros(0, self.shared_state_dim), persistent=False)

    def attention_pool(self, tokens, padding_mask, queries):
        batch_size = tokens.shape[0]
        if queries.dim() == 2:
            queries = queries.unsqueeze(0).expand(batch_size, -1, -1)

        scores = torch.matmul(queries, tokens.transpose(1, 2)) / math.sqrt(tokens.shape[-1])
        scores = scores.masked_fill(
            padding_mask.unsqueeze(1),
            torch.finfo(scores.dtype).min,
        )
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(1), 0.0)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.matmul(attn_weights, tokens)
        return pooled, attn_weights

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _ensure_runtime_state(self, batch_size, device, dtype):
        if (
            self.shared_state.shape != (batch_size, self.shared_state_dim)
            or self.shared_state.device != device
            or self.shared_state.dtype != dtype
        ):
            self.shared_state = torch.zeros(
                batch_size,
                self.shared_state_dim,
                device=device,
                dtype=dtype,
            )

    def reset(self, dones=None):
        if self.shared_state.numel() == 0:
            return
        new_shared_state = torch.zeros(
            self.shared_state.shape,
            device=self.shared_state.device,
            dtype=self.shared_state.dtype,
        )
        if dones is None:
            self.shared_state = new_shared_state
            return
        dones = dones.view(-1).bool()
        if dones.shape[0] == self.shared_state.shape[0]:
            keep_mask = ~dones
            if torch.any(keep_mask):
                new_shared_state[keep_mask] = self.shared_state[keep_mask]
            self.shared_state = new_shared_state

    def detach_hidden_state(self):
        if self.shared_state.numel() > 0:
            self.shared_state = self.shared_state.detach()

    def sinkhorn(self, scores):
        scores = scores / self.swav_epsilon
        scores = scores - scores.max()
        q = torch.exp(scores).t()
        q = q / q.sum().clamp_min(1e-6)
        k, batch_size = q.shape
        for _ in range(self.swav_sinkhorn_iters):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)
            q = q / k
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-6)
            q = q / batch_size
        return (q * batch_size).t()

    def cross_view_swav(self, gate_window, map_window, valid_mask):
        if gate_window is None or map_window is None or valid_mask is None:
            return torch.tensor(0.0, device=self.terrain_prototypes.weight.device)

        valid_mask = valid_mask.bool()
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=gate_window.device)

        gate_tokens = gate_window[valid_mask]
        map_tokens = map_window[valid_mask]

        gate_features = F.normalize(self.gate_terrain_projector(gate_tokens), dim=-1)
        map_features = F.normalize(self.elevation_terrain_projector(map_tokens), dim=-1)

        gate_logits = self.terrain_prototypes(gate_features)
        map_logits = self.terrain_prototypes(map_features)

        with torch.no_grad():
            gate_assign = self.sinkhorn(gate_logits.detach())
            map_assign = self.sinkhorn(map_logits.detach())

        gate_probs = F.softmax(gate_logits / self.swav_temperature, dim=-1)
        map_probs = F.softmax(map_logits / self.swav_temperature, dim=-1)

        loss_gate = torch.sum(gate_assign * torch.log(map_probs.clamp_min(1e-6)), dim=-1).mean()
        loss_map = torch.sum(map_assign * torch.log(gate_probs.clamp_min(1e-6)), dim=-1).mean()
        return -0.5 * (loss_gate + loss_map)

    def forward(
        self,
        proprio_seq,
        depth_seq,
        mask_vision,
        gt_mt_step=None,
        obs_now=None,
        shared_state_override=None,
        update_state=True,
    ):
        if obs_now is None:
            obs_now = torch.zeros(
                proprio_seq.shape[0],
                self.core.proprio_dim,
                device=proprio_seq.device,
                dtype=proprio_seq.dtype,
            )
        self._ensure_runtime_state(proprio_seq.shape[0], proprio_seq.device, proprio_seq.dtype)
        if shared_state_override is None:
            shared_state_prev = self.shared_state
        else:
            shared_state_prev = shared_state_override.to(device=proprio_seq.device, dtype=proprio_seq.dtype)

        fused_tokens, cross_attn_weights = self.core(
            proprio_seq,
            depth_seq,
            mask_vision,
        )
        token_padding_mask = torch.zeros(
            fused_tokens.shape[0],
            fused_tokens.shape[1],
            dtype=torch.bool,
            device=fused_tokens.device,
        )
        selector_pooled_feature, selector_pool_weights = self.attention_pool(
            fused_tokens,
            token_padding_mask,
            self.shared_pool_query,
        )
        selector_pooled_feature = selector_pooled_feature.squeeze(1)
        terrain_logits = self.selector_head(selector_pooled_feature)
        terrain_probs = F.softmax(terrain_logits, dim=-1)
        fall_recovery_logits = self.fall_recovery_head(selector_pooled_feature)
        fall_recovery_prob = torch.sigmoid(fall_recovery_logits)
        shared_pooled_feature = selector_pooled_feature
        shared_pool_weights = selector_pool_weights
        context_feature = self.shared_head(shared_pooled_feature)
        expert_pooled_features, expert_pool_weights = self.attention_pool(
            fused_tokens,
            token_padding_mask,
            self.expert_pool_queries,
        )
        expert_fall_recovery_feature = self.expert_fall_recovery_encoder(fall_recovery_prob)
        expert_pooled_features = expert_pooled_features + expert_fall_recovery_feature.unsqueeze(1)

        vision_flag = mask_vision.float().unsqueeze(-1)
        terrain_router_feature = self.router_terrain_encoder(terrain_probs)
        fall_recovery_router_feature = self.router_fall_recovery_encoder(fall_recovery_prob)
        router_input = torch.cat(
            (
                obs_now,
                vision_flag,
                context_feature,
                terrain_router_feature,
                fall_recovery_router_feature,
            ),
            dim=-1,
        )
        gating_weights = self.router(router_input)

        expert_hidden_states = []
        expert_deltas = []
        for expert_idx, expert in enumerate(self.experts):
            hidden_state, shared_delta = expert(
                expert_pooled_features[:, expert_idx, :],
                shared_state_prev,
            )
            expert_hidden_states.append(hidden_state)
            expert_deltas.append(shared_delta)
        expert_hidden_states = torch.stack(expert_hidden_states, dim=1)
        expert_deltas = torch.stack(expert_deltas, dim=1)

        mixed_delta = torch.sum(gating_weights.unsqueeze(-1) * expert_deltas, dim=1)
        shared_state = self.shared_state_norm(
            self.shared_state_updater(mixed_delta, shared_state_prev)
        )
        predictor_input = self.prediction_head(shared_state)

        v_t = self.fc_v_t(predictor_input)
        h_tf = self.fc_h_tf(predictor_input)
        z_mu = self.fc_z_mu(predictor_input)
        z_logvar = self.fc_z_logvar(predictor_input)
        z_tm = self.fc_z_tm(predictor_input)

        z_t = self.reparameterize(z_mu, z_logvar)
        o_hat = self.decoder_obs(torch.cat((v_t, h_tf, z_t), dim=-1))
        m_hat = self.decoder_map(z_tm)
        terrain_pred = torch.argmax(terrain_logits, dim=-1)
        terrain_pred_onehot = F.one_hot(
            terrain_pred,
            num_classes=self.num_terrain_types,
        ).to(proprio_seq.dtype)
        fall_recovery_pred = (fall_recovery_prob >= 0.5).to(proprio_seq.dtype)

        mcp_code = torch.cat(
            (v_t, h_tf, z_mu, z_tm, terrain_pred_onehot, fall_recovery_pred),
            dim=-1,
        )
        mcp_code_additional = torch.cat((z_mu, z_logvar, m_hat, o_hat), dim=-1)

        if update_state:
            self.shared_state = shared_state.detach()

        return {
            "v_t": v_t,
            "h_tf": h_tf,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z_tm": z_tm,
            "o_hat": o_hat,
            "m_hat": m_hat,
            "mcp_code": mcp_code,
            "mcp_code_additional": mcp_code_additional,
            "shared_state_prev": shared_state_prev,
            "shared_state": shared_state,
            "expert_hidden_states": expert_hidden_states,
            "expert_deltas": expert_deltas,
            "terrain_logits": terrain_logits,
            "terrain_probs": terrain_probs,
            "terrain_pred": terrain_pred,
            "terrain_pred_onehot": terrain_pred_onehot,
            "fall_recovery_logits": fall_recovery_logits,
            "fall_recovery_prob": fall_recovery_prob,
            "fall_recovery_pred": fall_recovery_pred,
            "swav_gating_weights": gating_weights,
            "cross_attn_weights": cross_attn_weights,
            "selector_pool_weights": selector_pool_weights.squeeze(1),
            "shared_pool_weights": shared_pool_weights.squeeze(1),
            "expert_pool_weights": expert_pool_weights,
        }


class DepthAutoEncoder(nn.Module):
    def __init__(self, image_recon_latent_dim=128, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.fc_enc = nn.Sequential(
            nn.Linear(64 * 8 * 11, image_recon_latent_dim),
            nn.ELU(),
        )
        self.fc_dec = nn.Linear(image_recon_latent_dim, 64 * 8 * 11)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.fc_enc(encoded.flatten(1))
        decoded = self.fc_dec(latent).view(-1, 64, 8, 11)
        recon = self.decoder(decoded)
        recon = F.interpolate(recon, size=(58, 87), mode="bilinear", align_corners=False)
        loss = F.mse_loss(recon, x)
        return recon, loss


class ActorCriticParkourMoE(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        num_actor_obs_now=45,
        num_critic_obs_fixed=None,
        mcp_dim=68,
        **kwargs,
    ):
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        if kwargs:
            print(
                "ActorCriticParkourMoE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_actor_obs_now = num_actor_obs_now
        self.num_critic_obs_fixed = num_critic_obs if num_critic_obs_fixed is None else num_critic_obs_fixed
        self.mcp_dim = mcp_dim
        self.vision_flag_dim = 1
        activation_layer = get_activation(activation)
        critic_input_dim = self.num_critic_obs_fixed + self.vision_flag_dim

        actor_layers = [
            nn.Linear(num_actor_obs_now + mcp_dim + self.vision_flag_dim, actor_hidden_dims[0]),
            activation_layer,
        ]
        for idx in range(len(actor_hidden_dims)):
            if idx == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[idx], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[idx], actor_hidden_dims[idx + 1]))
                actor_layers.append(get_activation(activation))
        self.actor = nn.Sequential(*actor_layers)
        nn.init.zeros_(self.actor[0].weight[:, self.mcp_dim : self.mcp_dim + self.vision_flag_dim])

        critic_layers = [nn.Linear(critic_input_dim, critic_hidden_dims[0]), get_activation(activation)]
        for idx in range(len(critic_hidden_dims)):
            if idx == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[idx], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[idx], critic_hidden_dims[idx + 1]))
                critic_layers.append(get_activation(activation))
        self.critic = nn.Sequential(*critic_layers)
        nn.init.zeros_(self.critic[0].weight[:, -self.vision_flag_dim :])

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _prepare_vision_flag(self, vision_flag, reference):
        if vision_flag is None:
            return torch.ones(reference.shape[0], 1, device=reference.device, dtype=reference.dtype)
        if vision_flag.dim() == 1:
            vision_flag = vision_flag.unsqueeze(-1)
        vision_flag = vision_flag.to(device=reference.device, dtype=reference.dtype).reshape(reference.shape[0], -1)
        if vision_flag.shape[1] != self.vision_flag_dim:
            raise ValueError(
                f"Expected vision_flag to have shape [batch, {self.vision_flag_dim}], got {tuple(vision_flag.shape)}"
            )
        return vision_flag.detach()

    def build_actor_input(self, mcp_code, observations):
        obs_now = observations[:, : self.num_actor_obs_now]
        return torch.cat((mcp_code, obs_now), dim=-1)

    def build_critic_input(self, critic_observations, vision_flag=None, mcp_code=None):
        critic_obs = critic_observations[:, : self.num_critic_obs_fixed]
        if vision_flag is None and mcp_code is not None:
            vision_flag = mcp_code[:, -self.vision_flag_dim :]
        vision_flag = self._prepare_vision_flag(vision_flag, critic_obs)
        return torch.cat((critic_obs, vision_flag), dim=-1)

    def update_distribution(self, mcp_code, observations):
        actor_input = self.build_actor_input(mcp_code, observations)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, mcp_code, observations, **kwargs):
        self.update_distribution(mcp_code, observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, mcp_code, observations):
        return self.actor(self.build_actor_input(mcp_code, observations))

    def evaluate(self, critic_observations, vision_flag=None, mcp_code=None, **kwargs):
        return self.critic(self.build_critic_input(critic_observations, vision_flag=vision_flag, mcp_code=mcp_code))
