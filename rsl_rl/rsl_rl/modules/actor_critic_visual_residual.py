"""Visual residual actor-critic used to extend a pretrained locomotion policy."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


def _build_mlp(input_dim, hidden_dims, output_dim, activation):
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(get_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCriticVisualResidual(nn.Module):
    """One policy composed of a pretrained proprioceptive actor and a visual residual.

    The residual output layer starts at zero, so a newly created policy behaves exactly
    like the loaded flat-ground actor until visual fine-tuning starts changing it.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        residual_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        num_actor_obs_now=45,
        mcp_dim=69,
        residual_scale=1.0,
        visual_residual_freeze_iterations=250,
        base_actor_freeze_iterations=1000,
        base_actor_lr_scale=0.25,
        num_critic_obs_fixed=None,
        expert_num=None,
        **kwargs,
    ):
        del num_actor_obs, num_critic_obs_fixed, expert_num
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        if residual_hidden_dims is None:
            residual_hidden_dims = [256, 128]
        if kwargs:
            print(
                "ActorCriticVisualResidual.__init__ got unexpected arguments, which will be ignored: "
                + str(sorted(kwargs.keys()))
            )

        super().__init__()
        self.num_actor_obs_now = int(num_actor_obs_now)
        self.mcp_dim = int(mcp_dim)
        self.vision_flag_dim = 1
        self.residual_scale = float(residual_scale)
        self.visual_residual_freeze_iterations = int(visual_residual_freeze_iterations)
        self.base_actor_freeze_iterations = int(base_actor_freeze_iterations)
        self.base_actor_lr_scale = float(base_actor_lr_scale)
        self._base_actor_frozen = False
        self._visual_residual_frozen = False

        visual_input_dim = self.num_actor_obs_now + self.mcp_dim + self.vision_flag_dim
        critic_input_dim = num_critic_obs + self.vision_flag_dim

        self.base_actor = _build_mlp(
            self.num_actor_obs_now,
            actor_hidden_dims,
            num_actions,
            activation,
        )
        self.visual_residual = _build_mlp(
            visual_input_dim,
            residual_hidden_dims,
            num_actions,
            activation,
        )
        self.critic = _build_mlp(
            critic_input_dim,
            critic_hidden_dims,
            1,
            activation,
        )

        residual_output = self.visual_residual[-1]
        nn.init.zeros_(residual_output.weight)
        nn.init.zeros_(residual_output.bias)

        self.std = nn.Parameter(float(init_noise_std) * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        self.set_training_iteration(0)
        print(f"Base proprioceptive actor: {self.base_actor}")
        print(f"Visual residual actor: {self.visual_residual}")
        print(f"Visual asymmetric critic: {self.critic}")

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        del dones

    def _prepare_vision_flag(self, vision_flag, reference):
        if vision_flag is None:
            return torch.ones(
                reference.shape[0],
                self.vision_flag_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
        if vision_flag.dim() == 1:
            vision_flag = vision_flag.unsqueeze(-1)
        return vision_flag.to(device=reference.device, dtype=reference.dtype).reshape(
            reference.shape[0], self.vision_flag_dim
        )

    def build_actor_input(self, mcp_code, observations):
        obs_now = observations[:, : self.num_actor_obs_now]
        return torch.cat((mcp_code, obs_now), dim=-1)

    def build_critic_input(self, critic_observations, vision_flag=None, mcp_code=None):
        if vision_flag is None and mcp_code is not None:
            vision_flag = mcp_code[:, -self.vision_flag_dim :]
        vision_flag = self._prepare_vision_flag(vision_flag, critic_observations)
        return torch.cat((critic_observations, vision_flag), dim=-1)

    def action_mean_from_inputs(self, mcp_code, observations):
        obs_now = observations[:, : self.num_actor_obs_now]
        base_mean = self.base_actor(obs_now)
        residual = self.visual_residual(self.build_actor_input(mcp_code, observations))
        return base_mean + self.residual_scale * residual

    def update_distribution(self, mcp_code, observations):
        mean = self.action_mean_from_inputs(mcp_code, observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, mcp_code, observations, **kwargs):
        del kwargs
        self.update_distribution(mcp_code, observations)
        return self.distribution.sample()

    def act_inference(self, mcp_code, observations):
        return self.action_mean_from_inputs(mcp_code, observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, vision_flag=None, mcp_code=None, observations=None, **kwargs):
        del observations, kwargs
        value = self.critic(
            self.build_critic_input(
                critic_observations,
                vision_flag=vision_flag,
                mcp_code=mcp_code,
            )
        )
        # PPOParkourMoE expects gating weights for its load-balance statistic.
        # A single unified policy has one constant gate and therefore zero loss.
        weights = torch.ones(value.shape[0], 1, device=value.device, dtype=value.dtype)
        return value, weights

    def set_training_iteration(self, iteration):
        iteration = int(iteration)
        should_freeze_base = iteration < self.base_actor_freeze_iterations
        if should_freeze_base != self._base_actor_frozen:
            for parameter in self.base_actor.parameters():
                parameter.requires_grad_(not should_freeze_base)
            self._base_actor_frozen = should_freeze_base
            state = "frozen" if should_freeze_base else "trainable"
            print(f"Base proprioceptive actor is now {state} at iteration {iteration}")

        should_freeze_residual = iteration < self.visual_residual_freeze_iterations
        if should_freeze_residual != self._visual_residual_frozen:
            for parameter in self.visual_residual.parameters():
                parameter.requires_grad_(not should_freeze_residual)
            self._visual_residual_frozen = should_freeze_residual
            state = "frozen" if should_freeze_residual else "trainable"
            print(f"Visual residual actor is now {state} at iteration {iteration}")

    def optimizer_parameter_groups(self, learning_rate):
        return [
            {
                "params": self.base_actor.parameters(),
                "lr": float(learning_rate) * self.base_actor_lr_scale,
            },
            {
                "params": self.visual_residual.parameters(),
                "lr": float(learning_rate),
            },
            {
                "params": self.critic.parameters(),
                "lr": float(learning_rate),
            },
            {
                "params": [self.std],
                "lr": float(learning_rate),
            },
        ]

    def warm_start_from_flat_checkpoint(
        self,
        checkpoint_path,
        max_action_std=0.45,
        critic_common_dim=76,
    ):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(str(checkpoint_path), map_location=self.std.device)
        state_dict = checkpoint["model_state_dict"]

        base_state = self.base_actor.state_dict()
        for key in list(base_state.keys()):
            source_key = f"actor.{key}"
            if source_key not in state_dict or state_dict[source_key].shape != base_state[key].shape:
                raise RuntimeError(
                    f"Cannot warm-start base actor parameter {key}: "
                    f"source={source_key}, expected={tuple(base_state[key].shape)}"
                )
            base_state[key] = state_dict[source_key].to(base_state[key].device)
        self.base_actor.load_state_dict(base_state)

        critic_state = self.critic.state_dict()
        old_critic_weight = state_dict["critic.0.weight"]
        new_critic_weight = critic_state["0.weight"]
        common_dim = min(
            int(critic_common_dim),
            old_critic_weight.shape[1],
            new_critic_weight.shape[1],
        )
        new_critic_weight.zero_()
        new_critic_weight[:, :common_dim] = old_critic_weight[:, :common_dim].to(
            new_critic_weight.device
        )
        critic_state["0.bias"] = state_dict["critic.0.bias"].to(critic_state["0.bias"].device)
        for layer_index in (2, 4, 6):
            for suffix in ("weight", "bias"):
                target_key = f"{layer_index}.{suffix}"
                source_key = f"critic.{target_key}"
                if state_dict[source_key].shape != critic_state[target_key].shape:
                    raise RuntimeError(
                        f"Cannot warm-start critic parameter {target_key}: "
                        f"source={tuple(state_dict[source_key].shape)}, "
                        f"target={tuple(critic_state[target_key].shape)}"
                    )
                critic_state[target_key] = state_dict[source_key].to(
                    critic_state[target_key].device
                )
        self.critic.load_state_dict(critic_state)

        loaded_std = state_dict["std"].to(self.std.device)
        self.std.data.copy_(torch.clamp(loaded_std, max=float(max_action_std)))
        print(
            f"Warm-started visual residual policy from {checkpoint_path}; "
            f"critic_common_dim={common_dim}, action_std_max={float(self.std.max()):.4f}"
        )
        return checkpoint
