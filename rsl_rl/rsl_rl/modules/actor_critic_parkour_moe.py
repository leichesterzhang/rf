import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.modules.utils import Experts, MoE


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


class ActiveTokenSampler(nn.Module):
    def __init__(
        self,
        token_dim=64,
        num_tokens=32,
        hidden_dim=None,
        noise_std=0.02,
        min_valid_mass=0.05,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = max(int(num_tokens), 1)
        self.noise_std = max(float(noise_std), 0.0)
        self.min_valid_mass = max(float(min_valid_mass), 0.0)
        if hidden_dim is None:
            hidden_dim = token_dim

        self.coord_predictor = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_tokens * 2),
            nn.Sigmoid(),
        )
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, token_map, token_valid_mask):
        batch_size, token_dim, image_height, image_width = token_map.shape
        if token_dim != self.token_dim:
            raise ValueError(
                f"Expected token_map channel dim {self.token_dim}, got {token_dim}"
            )

        valid_weights = token_valid_mask.to(dtype=token_map.dtype).unsqueeze(1)
        masked_token_map = token_map * valid_weights
        valid_count = valid_weights.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = masked_token_map.sum(dim=(2, 3)) / valid_count

        coords_01 = self.coord_predictor(pooled).view(
            batch_size,
            self.num_tokens,
            2,
        )
        coords = coords_01 * 2.0 - 1.0
        if self.training and self.noise_std > 0.0:
            coords = coords + torch.randn_like(coords) * self.noise_std
        coords = coords.clamp(-1.0, 1.0)

        sample_grid = coords.view(batch_size, self.num_tokens, 1, 2)
        sampled_tokens = F.grid_sample(
            masked_token_map,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(-1).transpose(1, 2)
        sampled_tokens = sampled_tokens + self.coord_encoder(coords)

        sampled_valid_mass = F.grid_sample(
            valid_weights,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(1).squeeze(-1)
        sampled_padding_mask = sampled_valid_mass <= self.min_valid_mass
        return sampled_tokens, sampled_padding_mask, coords, sampled_valid_mass


class ParkourEstimatorCore(nn.Module):
    def __init__(
        self,
        proprio_dim=45,
        history_length=10,
        token_dim=64,
        depth_frame_count=2,
        local_patch_size=15,
        local_patch_stride=8,
        token_dropout_min=0.05,
        token_dropout_max=0.15,
        token_dropout_mode="spatial_block",
        token_dropout_block_min_size=(1, 2),
        token_dropout_block_max_size=(3, 5),
        token_dropout_block_min_count=1,
        token_dropout_block_max_count=4,
        token_dropout_uniform_weight=0.15,
        token_dropout_ground_weight=3.0,
        token_dropout_far_weight=2.0,
        token_dropout_center_weight=0.5,
        token_dropout_center_sigma=0.35,
        token_dropout_ground_row_start=0.55,
        token_dropout_far_row_end=0.35,
        token_confidence_temperature=1.0,
        use_active_token_sampler=False,
        active_token_count=32,
        active_token_hidden_dim=None,
        active_token_noise_std=0.02,
        active_token_min_valid_mass=0.05,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.history_length = history_length
        self.token_dim = token_dim
        self.depth_frame_count = depth_frame_count
        self.local_patch_size = local_patch_size
        self.local_patch_stride = local_patch_stride
        self.local_patch_padding = local_patch_size // 2
        self.local_patch_dim = local_patch_size * local_patch_size
        self.token_dropout_min = token_dropout_min
        self.token_dropout_max = token_dropout_max
        self.token_dropout_mode = str(token_dropout_mode)
        self.token_dropout_block_min_size = self._coerce_pair(token_dropout_block_min_size, (1, 2))
        self.token_dropout_block_max_size = self._coerce_pair(token_dropout_block_max_size, (3, 5))
        self.token_dropout_block_min_count = max(int(token_dropout_block_min_count), 1)
        self.token_dropout_block_max_count = max(
            int(token_dropout_block_max_count),
            self.token_dropout_block_min_count,
        )
        self.token_dropout_uniform_weight = max(float(token_dropout_uniform_weight), 0.0)
        self.token_dropout_ground_weight = max(float(token_dropout_ground_weight), 0.0)
        self.token_dropout_far_weight = max(float(token_dropout_far_weight), 0.0)
        self.token_dropout_center_weight = max(float(token_dropout_center_weight), 0.0)
        self.token_dropout_center_sigma = max(float(token_dropout_center_sigma), 1.0e-3)
        self.token_dropout_ground_row_start = min(max(float(token_dropout_ground_row_start), 0.0), 0.99)
        self.token_dropout_far_row_end = min(max(float(token_dropout_far_row_end), 0.01), 1.0)
        self.token_confidence_temperature = max(float(token_confidence_temperature), 1.0e-6)
        self.use_active_token_sampler = bool(use_active_token_sampler)
        self.active_token_count = max(int(active_token_count), 1)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.image_token_norm = nn.LayerNorm(token_dim)
        self.local_patch_decoder = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.local_patch_dim),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, token_dim),
        )
        self.proprio_token_norm = nn.LayerNorm(token_dim)
        self.terrain_token_norm = nn.LayerNorm(token_dim)
        if self.use_active_token_sampler:
            self.active_token_sampler = ActiveTokenSampler(
                token_dim=token_dim,
                num_tokens=self.active_token_count,
                hidden_dim=active_token_hidden_dim,
                noise_std=active_token_noise_std,
                min_valid_mass=active_token_min_valid_mass,
            )
        else:
            self.active_token_sampler = None

    @staticmethod
    def _coerce_pair(value, default):
        if value is None:
            return (int(default[0]), int(default[1]))
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return (int(default[0]), int(default[1]))
            if len(value) == 1:
                scalar = int(value[0])
                return (scalar, scalar)
            return (int(value[0]), int(value[1]))
        scalar = int(value)
        return (scalar, scalar)

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
        frame_position_encoding = build_sincos_1d_encoding(
            self.depth_frame_count,
            self.token_dim,
            device,
            dtype,
        ).view(1, self.depth_frame_count, 1, self.token_dim)
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
            frame_position_encoding,
            proprio_type_encoding,
            proprio_position_encoding,
            flag_type_encoding,
            terrain_type_encoding,
            aux_position_encoding[:, 0:1, :],
            aux_position_encoding[:, 1:2, :],
        )

    def extract_local_patches(self, depth_frames, image_height, image_width):
        patches = F.unfold(
            depth_frames,
            kernel_size=self.local_patch_size,
            stride=self.local_patch_stride,
            padding=self.local_patch_padding,
        )
        expected_tokens = image_height * image_width
        if patches.shape[-1] != expected_tokens:
            raise RuntimeError(
                "Local patch grid does not match image token grid: "
                f"got {patches.shape[-1]} patches, expected {expected_tokens} "
                f"({image_height}x{image_width})."
            )
        return patches.transpose(1, 2)

    def _sanitize_token_dropout_ratio(self, dropout_min=None, dropout_max=None):
        if dropout_min is None:
            dropout_min = self.token_dropout_min
        if dropout_max is None:
            dropout_max = self.token_dropout_max
        dropout_min = max(float(dropout_min), 0.0)
        dropout_max = min(max(float(dropout_max), dropout_min), 1.0)
        return dropout_min, dropout_max

    def _build_independent_token_dropout_mask(
        self,
        batch_size,
        frame_count,
        tokens_per_frame,
        mask_vision,
        device,
        dropout_min,
        dropout_max,
    ):
        dropout_mask = torch.zeros(
            batch_size,
            frame_count,
            tokens_per_frame,
            dtype=torch.bool,
            device=device,
        )
        if tokens_per_frame <= 0 or dropout_max <= 0.0:
            return dropout_mask

        drop_ratio = torch.empty(
            batch_size,
            frame_count,
            device=device,
        ).uniform_(dropout_min, dropout_max)
        drop_count = torch.round(drop_ratio * tokens_per_frame).long()
        drop_count = drop_count.clamp(min=0, max=tokens_per_frame)
        random_rank = torch.rand(
            batch_size,
            frame_count,
            tokens_per_frame,
            device=device,
        ).argsort(dim=-1).argsort(dim=-1)
        dropout_mask = random_rank < drop_count.unsqueeze(-1)
        return dropout_mask & mask_vision.view(batch_size, 1, 1)

    def build_spatial_token_dropout_weights(self, image_height, image_width, device):
        rows = torch.linspace(0.0, 1.0, image_height, device=device).view(image_height, 1)
        cols = torch.linspace(-1.0, 1.0, image_width, device=device).view(1, image_width)
        ground = ((rows - self.token_dropout_ground_row_start).clamp(min=0.0) /
                  max(1.0 - self.token_dropout_ground_row_start, 1.0e-6)).pow(2)
        far = ((self.token_dropout_far_row_end - rows).clamp(min=0.0) /
               max(self.token_dropout_far_row_end, 1.0e-6)).pow(2)
        center_row = rows - 0.5
        center = torch.exp(
            -0.5
            * (center_row.pow(2) + cols.pow(2))
            / (self.token_dropout_center_sigma ** 2)
        )
        weights = (
            self.token_dropout_uniform_weight
            + self.token_dropout_ground_weight * ground
            + self.token_dropout_far_weight * far
            + self.token_dropout_center_weight * center
        )
        weights = weights.reshape(-1).clamp(min=0.0)
        if not torch.isfinite(weights).all() or weights.sum() <= 0.0:
            weights = torch.ones(image_height * image_width, device=device)
        return weights / weights.sum().clamp_min(1.0e-6)

    def _build_spatial_block_token_dropout_mask(
        self,
        batch_size,
        frame_count,
        image_height,
        image_width,
        mask_vision,
        device,
        dropout_min,
        dropout_max,
    ):
        tokens_per_frame = image_height * image_width
        dropout_mask = torch.zeros(
            batch_size,
            frame_count,
            tokens_per_frame,
            dtype=torch.bool,
            device=device,
        )
        if tokens_per_frame <= 0 or dropout_max <= 0.0:
            return dropout_mask

        drop_ratio = torch.empty(batch_size, frame_count, device=device).uniform_(
            dropout_min,
            dropout_max,
        )
        target_count = torch.round(drop_ratio * tokens_per_frame).long().clamp(
            min=0,
            max=tokens_per_frame,
        )
        flat_target_count = target_count.reshape(-1)
        if torch.all(flat_target_count <= 0):
            return dropout_mask

        min_block_h = max(int(self.token_dropout_block_min_size[0]), 1)
        min_block_w = max(int(self.token_dropout_block_min_size[1]), 1)
        max_block_h = max(int(self.token_dropout_block_max_size[0]), min_block_h)
        max_block_w = max(int(self.token_dropout_block_max_size[1]), min_block_w)
        min_block_h = min(min_block_h, image_height)
        min_block_w = min(min_block_w, image_width)
        max_block_h = min(max_block_h, image_height)
        max_block_w = min(max_block_w, image_width)
        max_blocks = max(int(self.token_dropout_block_max_count), 1)
        min_blocks = min(max(int(self.token_dropout_block_min_count), 1), max_blocks)

        flat_size = batch_size * frame_count
        weights = self.build_spatial_token_dropout_weights(image_height, image_width, device)
        center_samples = torch.multinomial(
            weights.view(1, -1).repeat(flat_size, 1),
            num_samples=max_blocks,
            replacement=max_blocks > tokens_per_frame,
        )
        center_rows = center_samples // image_width
        center_cols = center_samples % image_width

        block_h = torch.randint(
            min_block_h,
            max_block_h + 1,
            (flat_size, max_blocks),
            device=device,
        )
        block_w = torch.randint(
            min_block_w,
            max_block_w + 1,
            (flat_size, max_blocks),
            device=device,
        )
        avg_block_area = max(
            1.0,
            0.25
            * float(min_block_h + max_block_h)
            * float(min_block_w + max_block_w),
        )
        base_block_count = torch.ceil(flat_target_count.float() / avg_block_area).long()
        if max_blocks > min_blocks:
            block_jitter = torch.randint(-1, 2, (flat_size,), device=device)
        else:
            block_jitter = torch.zeros(flat_size, dtype=torch.long, device=device)
        block_count = (base_block_count + block_jitter).clamp(min=min_blocks, max=max_blocks)
        block_count = torch.where(
            flat_target_count > 0,
            block_count,
            torch.zeros_like(block_count),
        )
        block_indices = torch.arange(max_blocks, device=device).view(1, max_blocks)
        active_blocks = block_indices < block_count.view(flat_size, 1)

        top = (center_rows - block_h // 2).clamp(min=0)
        left = (center_cols - block_w // 2).clamp(min=0)
        top = torch.minimum(top, image_height - block_h)
        left = torch.minimum(left, image_width - block_w)
        bottom = top + block_h
        right = left + block_w

        row_ids = torch.arange(image_height, device=device).view(1, 1, image_height, 1)
        col_ids = torch.arange(image_width, device=device).view(1, 1, 1, image_width)
        block_mask = (
            active_blocks.view(flat_size, max_blocks, 1, 1)
            & (row_ids >= top.view(flat_size, max_blocks, 1, 1))
            & (row_ids < bottom.view(flat_size, max_blocks, 1, 1))
            & (col_ids >= left.view(flat_size, max_blocks, 1, 1))
            & (col_ids < right.view(flat_size, max_blocks, 1, 1))
        )
        dropout_mask = block_mask.any(dim=1).reshape(
            batch_size,
            frame_count,
            tokens_per_frame,
        )
        return dropout_mask & mask_vision.view(batch_size, 1, 1)

    def build_token_dropout_mask(
        self,
        batch_size,
        tokens_per_frame,
        mask_vision,
        device,
        dropout_min=None,
        dropout_max=None,
        frame_count=None,
        image_height=None,
        image_width=None,
        mode=None,
    ):
        frame_count = self.depth_frame_count if frame_count is None else int(frame_count)
        dropout_min, dropout_max = self._sanitize_token_dropout_ratio(dropout_min, dropout_max)
        if mode is None:
            mode = self.token_dropout_mode
        mode = str(mode).lower()

        if dropout_max <= 0.0 or mode in ("none", "off", "disabled", "disable"):
            return torch.zeros(
                batch_size,
                frame_count,
                tokens_per_frame,
                dtype=torch.bool,
                device=device,
            )

        if (
            mode in ("spatial", "spatial_block", "block", "blocks")
            and image_height is not None
            and image_width is not None
            and int(image_height) * int(image_width) == tokens_per_frame
        ):
            return self._build_spatial_block_token_dropout_mask(
                batch_size,
                frame_count,
                int(image_height),
                int(image_width),
                mask_vision,
                device,
                dropout_min,
                dropout_max,
            )

        return self._build_independent_token_dropout_mask(
            batch_size,
            frame_count,
            tokens_per_frame,
            mask_vision,
            device,
            dropout_min,
            dropout_max,
        )

    def build_training_token_dropout_mask(
        self,
        batch_size,
        tokens_per_frame,
        mask_vision,
        device,
        image_height=None,
        image_width=None,
    ):
        if not self.training:
            return torch.zeros(
                batch_size,
                self.depth_frame_count,
                tokens_per_frame,
                dtype=torch.bool,
                device=device,
            )
        return self.build_token_dropout_mask(
            batch_size,
            tokens_per_frame,
            mask_vision,
            device,
            frame_count=self.depth_frame_count,
            image_height=image_height,
            image_width=image_width,
        )

    def set_token_dropout_ratio(self, dropout_min=None, dropout_max=None):
        if dropout_min is not None:
            self.token_dropout_min = float(dropout_min)
        if dropout_max is not None:
            self.token_dropout_max = float(dropout_max)
        self.token_dropout_min = max(self.token_dropout_min, 0.0)
        self.token_dropout_max = max(self.token_dropout_max, self.token_dropout_min)
        self.token_dropout_max = min(self.token_dropout_max, 1.0)

    def encode_visual_tokens(self, depth_seq):
        batch_size = depth_seq.shape[0]
        if depth_seq.dim() != 4 or depth_seq.shape[1] != self.depth_frame_count:
            raise ValueError(
                f"Expected depth_seq to have shape [batch, {self.depth_frame_count}, H, W], "
                f"got {tuple(depth_seq.shape)}"
            )

        depth_frames = depth_seq.reshape(
            batch_size * self.depth_frame_count,
            1,
            depth_seq.shape[-2],
            depth_seq.shape[-1],
        )
        image_features = self.image_encoder(depth_frames)
        image_height, image_width = image_features.shape[-2:]
        tokens_per_frame = image_height * image_width
        image_tokens = image_features.flatten(2).transpose(1, 2)
        image_tokens = image_tokens.reshape(
            batch_size,
            self.depth_frame_count,
            tokens_per_frame,
            self.token_dim,
        )
        image_tokens = self.image_token_norm(image_tokens)
        if self.training:
            recon_frame_start = self.depth_frame_count - 1
            recon_frame_count = 1
        else:
            recon_frame_start = 0
            recon_frame_count = self.depth_frame_count
        recon_frame_end = recon_frame_start + recon_frame_count
        recon_depth_frames = depth_seq[:, recon_frame_start:recon_frame_end].reshape(
            batch_size * recon_frame_count,
            1,
            depth_seq.shape[-2],
            depth_seq.shape[-1],
        )
        local_patch_targets = self.extract_local_patches(
            recon_depth_frames,
            image_height,
            image_width,
        ).reshape(
            batch_size,
            recon_frame_count,
            tokens_per_frame,
            self.local_patch_dim,
        )
        local_patch_recon = self.local_patch_decoder(
            image_tokens[:, recon_frame_start:recon_frame_end]
        )
        token_self_recon_loss = F.mse_loss(
            local_patch_recon,
            local_patch_targets,
            reduction="none",
        ).mean(dim=-1)
        token_recon_loss = token_self_recon_loss
        token_confidence = torch.exp(-token_recon_loss / self.token_confidence_temperature)

        return {
            "image_tokens": image_tokens,
            "local_patch_recon": local_patch_recon,
            "local_patch_targets": local_patch_targets,
            "token_self_recon_loss": token_self_recon_loss,
            "token_recon_loss": token_recon_loss,
            "token_confidence": token_confidence,
            "visual_tokens_per_frame": tokens_per_frame,
            "visual_frame_count": self.depth_frame_count,
            "image_height": image_height,
            "image_width": image_width,
            "recon_frame_start": recon_frame_start,
            "recon_frame_count": recon_frame_count,
        }

    def encode_depth_tokens(self, depth_seq):
        return self.encode_visual_tokens(depth_seq)

    def encode_proprio_tokens(self, proprio_seq):
        batch_size = proprio_seq.shape[0]
        proprio_tokens = self.proprio_encoder(
            proprio_seq.reshape(batch_size, self.history_length, self.proprio_dim)
        )
        return self.proprio_token_norm(proprio_tokens)

    def normalize_external_token_mask(self, token_padding_mask, batch_size, tokens_per_frame, device):
        if token_padding_mask is None:
            return torch.zeros(
                batch_size,
                self.depth_frame_count,
                tokens_per_frame,
                dtype=torch.bool,
                device=device,
            )

        token_padding_mask = token_padding_mask.to(device=device, dtype=torch.bool)
        expected_total_tokens = self.depth_frame_count * tokens_per_frame
        if token_padding_mask.dim() == 2:
            expected_shape = (batch_size, expected_total_tokens)
            if tuple(token_padding_mask.shape) != expected_shape:
                raise ValueError(
                    "Expected flattened token_padding_mask to have shape "
                    f"{expected_shape}, got {tuple(token_padding_mask.shape)}"
                )
            return token_padding_mask.reshape(batch_size, self.depth_frame_count, tokens_per_frame)

        if token_padding_mask.dim() == 3:
            expected_shape = (batch_size, self.depth_frame_count, tokens_per_frame)
            if tuple(token_padding_mask.shape) == expected_shape:
                return token_padding_mask
            latest_frame_shape = (batch_size, 1, tokens_per_frame)
            if tuple(token_padding_mask.shape) == latest_frame_shape:
                full_mask = torch.zeros(
                    batch_size,
                    self.depth_frame_count,
                    tokens_per_frame,
                    dtype=torch.bool,
                    device=device,
                )
                full_mask[:, -1:, :] = token_padding_mask
                return full_mask
            raise ValueError(
                "Expected token_padding_mask to have shape "
                f"{expected_shape} or {latest_frame_shape}, got {tuple(token_padding_mask.shape)}"
            )

        raise ValueError(
            "Expected token_padding_mask to have rank 2 [batch, frame*tokens] "
            f"or rank 3 [batch, frame, tokens], got rank {token_padding_mask.dim()}"
        )

    def build_transformer_input_from_tokens(
        self,
        proprio_seq,
        visual_token_data,
        mask_vision,
        token_padding_mask=None,
        apply_training_dropout=True,
    ):
        batch_size = proprio_seq.shape[0]
        mask_vision = mask_vision.bool()
        image_tokens = visual_token_data["image_tokens"]
        image_height = int(visual_token_data["image_height"])
        image_width = int(visual_token_data["image_width"])
        tokens_per_frame = image_height * image_width

        expected_image_shape = (
            batch_size,
            self.depth_frame_count,
            tokens_per_frame,
            self.token_dim,
        )
        if tuple(image_tokens.shape) != expected_image_shape:
            raise ValueError(
                "Expected image_tokens to have shape "
                f"{expected_image_shape}, got {tuple(image_tokens.shape)}"
            )

        proprio_tokens = self.encode_proprio_tokens(proprio_seq)
        training_token_dropout_mask = self.build_training_token_dropout_mask(
            batch_size,
            tokens_per_frame,
            mask_vision,
            image_tokens.device,
            image_height=image_height,
            image_width=image_width,
        )
        if not apply_training_dropout:
            training_token_dropout_mask = torch.zeros_like(training_token_dropout_mask)
        external_token_padding_mask = self.normalize_external_token_mask(
            token_padding_mask,
            batch_size,
            tokens_per_frame,
            image_tokens.device,
        )
        vision_token = mask_vision.float().view(batch_size, 1, 1)
        (
            image_type_encoding,
            image_position_encoding,
            frame_position_encoding,
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

        dense_image_padding_mask = (
            (~mask_vision).view(batch_size, 1, 1).expand_as(training_token_dropout_mask)
            | training_token_dropout_mask
            | external_token_padding_mask
        )
        use_active_sampler = (
            self.active_token_sampler is not None
            and self.active_token_count < tokens_per_frame
        )
        active_token_coords = None
        active_token_valid_mass = None
        dense_visual_token_padding_mask = dense_image_padding_mask.reshape(batch_size, -1)

        if use_active_sampler:
            sampler_token_map = image_tokens.reshape(
                batch_size * self.depth_frame_count,
                tokens_per_frame,
                self.token_dim,
            ).transpose(1, 2).reshape(
                batch_size * self.depth_frame_count,
                self.token_dim,
                image_height,
                image_width,
            )
            sampler_valid_mask = (~dense_image_padding_mask).reshape(
                batch_size * self.depth_frame_count,
                image_height,
                image_width,
            )
            sampled_image_tokens, sampled_padding_mask, active_token_coords, active_token_valid_mass = (
                self.active_token_sampler(
                    sampler_token_map,
                    sampler_valid_mask,
                )
            )
            sampled_image_tokens = sampled_image_tokens.reshape(
                batch_size,
                self.depth_frame_count,
                self.active_token_count,
                self.token_dim,
            )
            sampled_padding_mask = sampled_padding_mask.reshape(
                batch_size,
                self.depth_frame_count,
                self.active_token_count,
            )
            active_token_coords = active_token_coords.reshape(
                batch_size,
                self.depth_frame_count,
                self.active_token_count,
                2,
            )
            active_token_valid_mass = active_token_valid_mass.reshape(
                batch_size,
                self.depth_frame_count,
                self.active_token_count,
            )
            image_tokens = (
                sampled_image_tokens
                + image_type_encoding.view(1, 1, 1, -1)
                + frame_position_encoding
            )
            image_padding_mask = sampled_padding_mask
            visual_tokens_per_frame = self.active_token_count
        else:
            image_tokens = (
                image_tokens
                + image_type_encoding.view(1, 1, 1, -1)
                + image_position_encoding.view(1, 1, image_height * image_width, -1)
                + frame_position_encoding
            )
            image_padding_mask = dense_image_padding_mask
            visual_tokens_per_frame = tokens_per_frame

        image_tokens = image_tokens.reshape(batch_size, -1, self.token_dim)
        image_padding_mask = image_padding_mask.reshape(batch_size, -1)
        image_tokens = image_tokens * (~image_padding_mask).to(image_tokens.dtype).unsqueeze(-1)
        proprio_tokens = proprio_tokens + proprio_type_encoding + proprio_position_encoding
        vision_token = vision_token + flag_type_encoding + flag_position_encoding

        tokens = [image_tokens, proprio_tokens, vision_token]
        tail_token_count = proprio_tokens.shape[1] + 1
        proprio_token_start = image_tokens.shape[1]
        proprio_token_end = proprio_token_start + proprio_tokens.shape[1]

        fused_input = torch.cat(tokens, dim=1)
        token_padding_mask = torch.cat(
            (
                image_padding_mask,
                torch.zeros(
                    batch_size,
                    tail_token_count,
                    dtype=torch.bool,
                    device=image_tokens.device,
                ),
            ),
            dim=1,
        )
        aux = dict(visual_token_data)
        aux.update({
            "training_token_dropout_mask": training_token_dropout_mask,
            "external_token_padding_mask": external_token_padding_mask,
            "visual_token_padding_mask": image_padding_mask,
            "dense_visual_token_padding_mask": dense_visual_token_padding_mask,
            "visual_tokens_per_frame": visual_tokens_per_frame,
            "dense_visual_tokens_per_frame": tokens_per_frame,
            "visual_frame_count": self.depth_frame_count,
            "image_height": image_height,
            "image_width": image_width,
            "active_token_enabled": use_active_sampler,
            "active_token_count": visual_tokens_per_frame if use_active_sampler else 0,
            "active_token_coords": active_token_coords,
            "active_token_valid_mass": active_token_valid_mass,
        })
        return fused_input, token_padding_mask, proprio_token_start, proprio_token_end, aux

    def forward_from_visual_tokens(
        self,
        proprio_seq,
        visual_token_data,
        mask_vision,
        token_padding_mask=None,
        apply_training_dropout=True,
    ):
        return self.build_transformer_input_from_tokens(
            proprio_seq,
            visual_token_data,
            mask_vision,
            token_padding_mask=token_padding_mask,
            apply_training_dropout=apply_training_dropout,
        )

    def forward_from_tokens(self, proprio_seq, visual_token_data, mask_vision, token_padding_mask=None):
        return self.forward_from_visual_tokens(
            proprio_seq,
            visual_token_data,
            mask_vision,
            token_padding_mask=token_padding_mask,
        )

    def forward(self, proprio_seq, depth_seq, mask_vision, terrain_token=None):
        del terrain_token
        visual_token_data = self.encode_visual_tokens(depth_seq)
        return self.forward_from_visual_tokens(proprio_seq, visual_token_data, mask_vision)


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
        expert_num=3,
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
        depth_frame_count=2,
        local_patch_size=15,
        local_patch_stride=8,
        token_dropout_min=0.05,
        token_dropout_max=0.15,
        token_dropout_mode="spatial_block",
        token_dropout_block_min_size=(1, 2),
        token_dropout_block_max_size=(3, 5),
        token_dropout_block_min_count=1,
        token_dropout_block_max_count=4,
        token_dropout_uniform_weight=0.15,
        token_dropout_ground_weight=3.0,
        token_dropout_far_weight=2.0,
        token_dropout_center_weight=0.5,
        token_dropout_center_sigma=0.35,
        token_dropout_ground_row_start=0.55,
        token_dropout_far_row_end=0.35,
        token_confidence_temperature=1.0,
        use_active_token_sampler=False,
        active_token_count=32,
        active_token_hidden_dim=None,
        active_token_noise_std=0.02,
        active_token_min_valid_mass=0.05,
        swav_temperature=0.1,
        swav_epsilon=0.05,
        swav_sinkhorn_iters=3,
        swav_num_prototypes=16,
        **kwargs,
    ):
        super().__init__()
        if expert_num < 1:
            raise ValueError("ParkourEstimator needs at least 1 terrain expert.")
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
        if cross_attn_nhead is not None:
            transformer_nhead = cross_attn_nhead

        self.core = ParkourEstimatorCore(
            proprio_dim=proprio_dim,
            history_length=history_length,
            token_dim=vision_token_dim,
            depth_frame_count=depth_frame_count,
            local_patch_size=local_patch_size,
            local_patch_stride=local_patch_stride,
            token_dropout_min=token_dropout_min,
            token_dropout_max=token_dropout_max,
            token_dropout_mode=token_dropout_mode,
            token_dropout_block_min_size=token_dropout_block_min_size,
            token_dropout_block_max_size=token_dropout_block_max_size,
            token_dropout_block_min_count=token_dropout_block_min_count,
            token_dropout_block_max_count=token_dropout_block_max_count,
            token_dropout_uniform_weight=token_dropout_uniform_weight,
            token_dropout_ground_weight=token_dropout_ground_weight,
            token_dropout_far_weight=token_dropout_far_weight,
            token_dropout_center_weight=token_dropout_center_weight,
            token_dropout_center_sigma=token_dropout_center_sigma,
            token_dropout_ground_row_start=token_dropout_ground_row_start,
            token_dropout_far_row_end=token_dropout_far_row_end,
            token_confidence_temperature=token_confidence_temperature,
            use_active_token_sampler=use_active_token_sampler,
            active_token_count=active_token_count,
            active_token_hidden_dim=active_token_hidden_dim,
            active_token_noise_std=active_token_noise_std,
            active_token_min_valid_mass=active_token_min_valid_mass,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_token_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
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
        self.motion_gru = nn.GRU(
            input_size=vision_token_dim,
            hidden_size=self.expert_hidden_dim,
            batch_first=True,
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
        self.shared_state_candidate = nn.Sequential(
            nn.Linear(self.shared_state_dim * 2, self.shared_state_dim),
            nn.ELU(),
            nn.Linear(self.shared_state_dim, self.shared_state_dim),
        )
        self.shared_state_gate = nn.Sequential(
            nn.Linear(self.shared_state_dim * 2, self.shared_state_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.shared_state_gate[0].bias, -2.0)
        self.shared_state_norm = nn.LayerNorm(self.shared_state_dim)
        self.motion_prediction_head = nn.Sequential(
            nn.Linear(self.expert_hidden_dim, vision_token_dim),
            nn.ELU(),
        )
        self.terrain_prediction_head = nn.Sequential(
            nn.Linear(self.expert_hidden_dim + self.shared_state_dim, vision_token_dim),
            nn.ELU(),
        )

        self.fc_v_t = nn.Linear(vision_token_dim, 3)
        self.fc_h_tf = nn.Linear(vision_token_dim, feet_height_dim)
        self.fc_motion_z_mu = nn.Linear(vision_token_dim, latent_dim)
        self.fc_motion_z_logvar = nn.Linear(vision_token_dim, latent_dim)
        self.fc_z_mu = nn.Linear(vision_token_dim, latent_dim)
        self.fc_z_logvar = nn.Linear(vision_token_dim, latent_dim)
        self.fc_z_tm = nn.Linear(vision_token_dim, terrain_latent_dim)

        self.decoder_obs = nn.Sequential(
            nn.Linear(3 + feet_height_dim + latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, proprio_dim),
        )
        self.terrain_decoder_obs = nn.Sequential(
            nn.Linear(latent_dim, 64),
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

    def set_token_dropout_ratio(self, dropout_min=None, dropout_max=None):
        self.core.set_token_dropout_ratio(dropout_min, dropout_max)

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

    def encode_visual_tokens(self, depth_seq):
        return self.core.encode_visual_tokens(depth_seq)

    def encode_depth_tokens(self, depth_seq):
        return self.encode_visual_tokens(depth_seq)

    def forward_from_visual_tokens(
        self,
        proprio_seq,
        visual_token_data,
        mask_vision,
        token_padding_mask=None,
        gt_mt_step=None,
        obs_now=None,
        shared_state_override=None,
        update_state=True,
        apply_training_dropout=True,
    ):
        del gt_mt_step
        core_outputs = self.core.forward_from_visual_tokens(
            proprio_seq,
            visual_token_data,
            mask_vision,
            token_padding_mask=token_padding_mask,
            apply_training_dropout=apply_training_dropout,
        )
        return self.forward_from_transformer_inputs(
            proprio_seq,
            mask_vision,
            *core_outputs,
            obs_now=obs_now,
            shared_state_override=shared_state_override,
            update_state=update_state,
        )

    def forward_from_tokens(self, proprio_seq, visual_token_data, mask_vision, token_padding_mask=None, **kwargs):
        return self.forward_from_visual_tokens(
            proprio_seq,
            visual_token_data,
            mask_vision,
            token_padding_mask=token_padding_mask,
            **kwargs,
        )

    def forward_from_transformer_inputs(
        self,
        proprio_seq,
        mask_vision,
        transformer_input,
        transformer_padding_mask,
        proprio_token_start,
        proprio_token_end,
        core_aux,
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

        encoded_tokens = self.transformer(
            transformer_input,
            src_key_padding_mask=transformer_padding_mask,
        )
        fused_tokens = encoded_tokens[:, proprio_token_start:proprio_token_end, :]
        cross_attn_weights = torch.zeros(
            fused_tokens.shape[0],
            fused_tokens.shape[1],
            proprio_token_start,
            dtype=fused_tokens.dtype,
            device=fused_tokens.device,
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
        expert_terrain_feature = self.terrain_token_encoder(terrain_probs)
        expert_fall_recovery_feature = self.expert_fall_recovery_encoder(fall_recovery_prob)
        expert_pooled_features = (
            expert_pooled_features
            + expert_terrain_feature.unsqueeze(1)
            + expert_fall_recovery_feature.unsqueeze(1)
        )

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
        terrain_gating_weights = self.router(router_input)
        gating_weights = terrain_gating_weights

        _, motion_gru_hidden = self.motion_gru(fused_tokens)
        motion_hidden = motion_gru_hidden[-1]

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

        terrain_hidden_states = expert_hidden_states
        terrain_deltas = expert_deltas
        mixed_terrain_hidden = torch.sum(
            terrain_gating_weights.unsqueeze(-1) * terrain_hidden_states,
            dim=1,
        )
        mixed_terrain_delta = torch.sum(
            terrain_gating_weights.unsqueeze(-1) * terrain_deltas,
            dim=1,
        )
        state_update_input = torch.cat((shared_state_prev, mixed_terrain_delta), dim=-1)
        state_candidate = self.shared_state_candidate(state_update_input)
        state_gate = self.shared_state_gate(state_update_input)
        shared_state = self.shared_state_norm(
            state_gate * state_candidate + (1.0 - state_gate) * shared_state_prev
        )

        motion_predictor_input = self.motion_prediction_head(motion_hidden)
        terrain_predictor_input = self.terrain_prediction_head(
            torch.cat((mixed_terrain_hidden, shared_state), dim=-1)
        )

        v_t = self.fc_v_t(motion_predictor_input)
        h_tf = self.fc_h_tf(motion_predictor_input)
        motion_z_mu = self.fc_motion_z_mu(motion_predictor_input)
        motion_z_logvar = self.fc_motion_z_logvar(motion_predictor_input)
        z_mu = self.fc_z_mu(terrain_predictor_input)
        z_logvar = self.fc_z_logvar(terrain_predictor_input)
        z_tm = self.fc_z_tm(terrain_predictor_input)

        motion_z_t = self.reparameterize(motion_z_mu, motion_z_logvar)
        z_t = self.reparameterize(z_mu, z_logvar)
        o_hat = self.decoder_obs(torch.cat((v_t, h_tf, motion_z_t), dim=-1))
        terrain_o_hat = self.terrain_decoder_obs(z_t)
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
            "motion_z_mu": motion_z_mu,
            "motion_z_logvar": motion_z_logvar,
            "motion_z_t": motion_z_t,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "terrain_z_mu": z_mu,
            "terrain_z_logvar": z_logvar,
            "terrain_z_t": z_t,
            "z_tm": z_tm,
            "o_hat": o_hat,
            "motion_o_hat": o_hat,
            "terrain_o_hat": terrain_o_hat,
            "m_hat": m_hat,
            "mcp_code": mcp_code,
            "mcp_code_additional": mcp_code_additional,
            "shared_state_prev": shared_state_prev,
            "shared_state": shared_state,
            "expert_hidden_states": expert_hidden_states,
            "expert_deltas": expert_deltas,
            "motion_hidden_state": motion_hidden,
            "terrain_logits": terrain_logits,
            "terrain_probs": terrain_probs,
            "terrain_pred": terrain_pred,
            "terrain_pred_onehot": terrain_pred_onehot,
            "fall_recovery_logits": fall_recovery_logits,
            "fall_recovery_prob": fall_recovery_prob,
            "fall_recovery_pred": fall_recovery_pred,
            "swav_gating_weights": gating_weights,
            "terrain_gating_weights": terrain_gating_weights,
            "cross_attn_weights": cross_attn_weights,
            "selector_pool_weights": selector_pool_weights.squeeze(1),
            "shared_pool_weights": shared_pool_weights.squeeze(1),
            "expert_pool_weights": expert_pool_weights,
            "local_patch_recon": core_aux["local_patch_recon"],
            "local_patch_targets": core_aux["local_patch_targets"],
            "token_self_recon_loss": core_aux["token_self_recon_loss"],
            "token_recon_loss": core_aux["token_recon_loss"],
            "token_confidence": core_aux["token_confidence"],
            "training_token_dropout_mask": core_aux["training_token_dropout_mask"],
            "external_token_padding_mask": core_aux["external_token_padding_mask"],
            "visual_token_padding_mask": core_aux["visual_token_padding_mask"],
            "dense_visual_token_padding_mask": core_aux["dense_visual_token_padding_mask"],
            "visual_tokens_per_frame": core_aux["visual_tokens_per_frame"],
            "dense_visual_tokens_per_frame": core_aux["dense_visual_tokens_per_frame"],
            "visual_frame_count": core_aux["visual_frame_count"],
            "image_height": core_aux["image_height"],
            "image_width": core_aux["image_width"],
            "active_token_enabled": core_aux.get("active_token_enabled", False),
            "active_token_count": core_aux.get("active_token_count", 0),
            "active_token_coords": core_aux.get("active_token_coords", None),
            "active_token_valid_mass": core_aux.get("active_token_valid_mass", None),
        }

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
        del gt_mt_step
        core_outputs = self.core(
            proprio_seq,
            depth_seq,
            mask_vision,
        )
        return self.forward_from_transformer_inputs(
            proprio_seq,
            mask_vision,
            *core_outputs,
            obs_now=obs_now,
            shared_state_override=shared_state_override,
            update_state=update_state,
        )


class DepthAutoEncoder(nn.Module):
    def __init__(self, image_recon_latent_dim=128, depth_frame_count=2, **kwargs):
        super().__init__()
        self.depth_frame_count = depth_frame_count
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
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
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() != 4 or x.shape[1] != self.depth_frame_count:
            raise ValueError(
                f"Expected x to have shape [batch, {self.depth_frame_count}, H, W], got {tuple(x.shape)}"
            )
        depth_frames = x.reshape(batch_size * self.depth_frame_count, 1, x.shape[-2], x.shape[-1])
        encoded = self.encoder(depth_frames)
        latent = self.fc_enc(encoded.flatten(1))
        decoded = self.fc_dec(latent).view(-1, 64, 8, 11)
        recon = self.decoder(decoded)
        recon = F.interpolate(recon, size=(58, 87), mode="bilinear", align_corners=False)
        recon = recon.reshape(batch_size, self.depth_frame_count, 58, 87)
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
        expert_num=8,
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
        self.expert_num = expert_num
        actor_input_dim = num_actor_obs_now + mcp_dim + self.vision_flag_dim
        critic_input_dim = self.num_critic_obs_fixed + self.vision_flag_dim

        self.actor_moe = MoE(
            expert_num=expert_num,
            input_dim=actor_input_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation,
        )

        self.critic_experts = Experts(
            expert_num=expert_num,
            input_dim=critic_input_dim,
            backbone_hidden_dims=critic_hidden_dims[:-1],
            expert_hidden_dim=critic_hidden_dims[-1],
            output_dim=1,
            activation=activation,
        )

        print(f"Actor MoE: {self.actor_moe}")
        print(f"Critic Experts: {self.critic_experts}")

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
        mean, _ = self.actor_moe(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, mcp_code, observations, **kwargs):
        self.update_distribution(mcp_code, observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, mcp_code, observations):
        mean, _ = self.actor_moe(self.build_actor_input(mcp_code, observations))
        return mean

    def evaluate(self, critic_observations, vision_flag=None, mcp_code=None, observations=None, **kwargs):
        if observations is None:
            observations = kwargs.get("obs", None)
        if mcp_code is None or observations is None:
            raise ValueError("ActorCriticParkourMoE.evaluate requires both mcp_code and observations.")
        x_actor = self.build_actor_input(mcp_code, observations)
        weights = self.actor_moe.gating_network(x_actor)
        x_critic = self.build_critic_input(critic_observations, vision_flag=vision_flag, mcp_code=mcp_code)
        experts_value = self.critic_experts(x_critic)
        value = torch.sum(weights.unsqueeze(-1) * experts_value, dim=1)
        return value, weights
