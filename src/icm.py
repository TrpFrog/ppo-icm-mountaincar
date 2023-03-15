"""
Curiosity-driven Exploration by Self-supervised Prediction
Pathak et al., 2017
"""
import torch
from torch import Tensor, nn


class Curiosity(nn.Module):
    def loss(self, policy_loss: Tensor, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        raise NotImplementedError

    def reward(self, accum_reward: Tensor, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        raise NotImplementedError


class NoCuriosity(Curiosity):
    def loss(self, policy_loss: Tensor, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        return policy_loss

    def reward(self, accum_reward: Tensor, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        return accum_reward.detach()


class ICMFeatureModel(nn.Sequential):
    def __init__(self,
                 state_dim: int,
                 latent_state_dim: int,
                 hidden_size: int = 256):
        super().__init__(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_state_dim),
        )


class DiscreteICMInverseModel(nn.Module):
    def __init__(self,
                 latent_state_dim: int,
                 num_actions: int,
                 hidden_size: int = 256):
        super().__init__()

        self.inverse = nn.Sequential(
            nn.Linear(latent_state_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, state: Tensor, next_state: Tensor) -> Tensor:
        x = torch.cat([state, next_state], dim=-1)
        return self.inverse(x)


class DiscreteICMForwardModel(nn.Module):
    def __init__(self,
                 latent_state_dim: int,
                 latent_action_dim: int,
                 num_actions: int,
                 hidden_size: int = 256):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, latent_action_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(latent_state_dim + latent_action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_state_dim),
        )

    def forward(self, latent_state: Tensor, action: Tensor) -> Tensor:
        action_embd = self.action_embedding(action.flatten())
        x = torch.cat([latent_state, action_embd], dim=-1)
        return self.forward_model(x)


class DiscreteICM(Curiosity):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 reward_scale: float = 1,           # In the paper: eta
                 policy_loss_weight: float = 0.1,   # In the paper: lambda
                 loss_scale: float = 0.2,           # In the paper: beta
                 hidden_size: int = 256):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.latent_state_dim = hidden_size
        self.latent_action_dim = hidden_size

        self.reward_scale = reward_scale
        assert reward_scale > 0.0
        self.policy_loss_weight = policy_loss_weight
        assert 0 <= policy_loss_weight
        self.loss_scalar = loss_scale
        assert 0.0 <= loss_scale <= 1.0

        self.inverse_model = DiscreteICMInverseModel(
            latent_state_dim=self.latent_state_dim,
            num_actions=self.action_dim,
            hidden_size=hidden_size
        )
        self.feature_model = ICMFeatureModel(
            state_dim=self.state_dim,
            latent_state_dim=self.latent_state_dim,
            hidden_size=hidden_size
        )
        self.forward_model = DiscreteICMForwardModel(
            latent_state_dim=self.latent_state_dim,
            latent_action_dim=self.latent_action_dim,
            num_actions=self.action_dim,
            hidden_size=hidden_size
        )

        self.inverse_criterion = nn.CrossEntropyLoss()
        self.forward_criterion = nn.MSELoss()
        self.reward_mse = nn.MSELoss(reduction='none')

        self.info_inverse_loss = 0
        self.info_forward_loss = 0
        self.info_loss = 0
        self.info_reward = 0

    @property
    def latest_info(self) -> dict:
        return {
            'inverse_loss': self.info_inverse_loss,
            'forward_loss': self.info_forward_loss,
            'loss': self.info_loss,
            'reward': self.info_reward,
        }

    def loss(self, policy_loss: Tensor, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        action = action.flatten()

        latent_state = self.feature_model(state)
        latent_next_state = self.feature_model(next_state)

        predicted_action = self.inverse_model(latent_state, latent_next_state)
        predicted_next_state = self.forward_model(latent_state, action)

        inverse_loss = self.inverse_criterion(predicted_action, action)
        forward_loss = 0.5 * self.forward_criterion(predicted_next_state, latent_next_state)

        beta = self.loss_scalar
        loss = (1 - beta) * inverse_loss + beta * forward_loss

        self.info_inverse_loss = inverse_loss.item()
        self.info_forward_loss = forward_loss.item()
        self.info_loss = loss.item()

        return self.policy_loss_weight * policy_loss + loss

    def reward(self,
               accum_reward: Tensor,
               state: Tensor,
               action: Tensor,
               next_state: Tensor) -> Tensor:

        latent_state = self.feature_model(state)
        latent_next_state = self.feature_model(next_state)
        predicted_next_state = self.forward_model(
            latent_state=latent_state,
            action=action.flatten()
        )

        intrinsic_reward = 0.5 * self.reward_mse(predicted_next_state, latent_next_state)
        intrinsic_reward = torch.sqrt(intrinsic_reward.sum(dim=-1))

        assert accum_reward.shape == intrinsic_reward.shape
        self.info_reward = self.reward_scale * intrinsic_reward.mean().item()

        return (accum_reward + self.reward_scale * intrinsic_reward).detach()
