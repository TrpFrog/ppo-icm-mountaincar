"""
Actor Critic

References
  - PyTorch Implementation of Actor Critic
    https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
"""

from typing import Optional
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from src import utils


device = utils.get_device()


@dataclass(frozen=True)
class EvaluationResult:
    action_log_probs: Tensor
    state_values: Tensor
    dist_entropy: Tensor


@dataclass(frozen=True)
class SelectedAction:
    action: Tensor
    log_prob: Tensor


# =========================================================== #


class PPOActorCritic(nn.Module, metaclass=ABCMeta):

    @dataclass(frozen=True)
    class Params:
        state_dim: int
        action_dim: int
        action_std_init: float

    params: Params

    # input: Tensor (state_dim,)
    # output: Tensor (action_dim,)
    actor: nn.Module

    # input: Tensor (state_dim,)
    # output: Tensor (1,)
    critic: nn.Module

    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def act(self, state: Tensor, greedy: bool = False) -> SelectedAction:
        """
        Get action from policy
        :param state: current state
        :param greedy: if True, get greedy action
        :return: Tuple of action and log probability of action
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, state: Tensor, action: Tensor) -> EvaluationResult | list[EvaluationResult]:
        """
        Get log probability and state value of action
        :param state: current state
        :param action: action to evaluate
        :return: Tuple of (log prob. of action, state value, dist. entropy)
        """
        raise NotImplementedError


# =========================================================== #


class DiscretePPOActorCritic(PPOActorCritic):
    """
    Actor-Critic for discrete action space
    """

    def __init__(self,
                 params: PPOActorCritic.Params,
                 actor: Optional[nn.Module] = None,
                 critic: Optional[nn.Module] = None):
        hidden = 64
        super().__init__()
        self.params = params

        # Actor Network
        if actor is None:
            actor = nn.Sequential(
                nn.Linear(params.state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, params.action_dim),
            )

        if critic is None:
            # Critic network
            critic = nn.Sequential(
                nn.Linear(params.state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.actor = actor
        self.critic = critic

        self._network_assertion()

    def _network_assertion(self):
        input_size = self.params.state_dim
        batch_size = 100
        x = torch.rand(batch_size, input_size)
        assert self.actor(x).shape == (batch_size, self.params.action_dim)
        assert self.critic(x).shape == (batch_size, 1)

    # select action
    def act(self, state: Tensor, greedy=False) -> SelectedAction:
        # Select action from Categorical distribution
        action_logits = self.actor(state)

        if greedy:
            action = torch.argmax(action_logits, dim=1)
            action_log_prob = torch.zeros_like(action)
            return SelectedAction(
                action=action.detach(),
                log_prob=action_log_prob.detach()
            )

        # print(action_probs)
        dist = torch.distributions.Categorical(logits=action_logits)

        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return SelectedAction(
            action=action.detach(),
            log_prob=action_log_prob.detach()
        )

    def evaluate(self, state: Tensor, action: Tensor) -> EvaluationResult:
        self.train()
        action_logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=action_logits)

        action_log_probs = dist.log_prob(action.flatten())
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return EvaluationResult(
            action_log_probs=action_log_probs,
            state_values=state_values,
            dist_entropy=dist_entropy
        )

# =========================================================== #


class ContinuousPPOActorCritic(PPOActorCritic):
    """
    Actor-Critic for continuous action space
    """

    action_var: Tensor  # Variance of actions

    def __init__(self, params: PPOActorCritic.Params):
        hidden = 64

        # Actor Network
        actor = nn.Sequential(
            nn.Linear(params.state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, params.action_dim),
        )

        # Critic Network
        critic = nn.Sequential(
            nn.Linear(params.state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.action_var = torch.full(
            (params.action_dim,),
            params.action_std_init ** 2
        )

        super().__init__(params, actor, critic)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full(
            (self.params.action_dim,),
            new_action_std ** 2
        ).to(device)

    def act(self, state: Tensor, greedy: bool = False) -> SelectedAction:
        action_mean = self.actor(state).softmax(dim=-1)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        if greedy:
            action = action_mean
            action_log_prob = torch.zeros_like(action)
            return SelectedAction(
                action=action.detach(),
                log_prob=action_log_prob.detach()
            )

        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return SelectedAction(
            action=action.detach(),
            log_prob=action_log_prob.detach()
        )

    def evaluate(self, state: Tensor, action: Tensor) -> EvaluationResult:
        action_mean = self.actor(state).softmax(dim=-1)
        assert action_mean.size(-1) == self.params.action_dim

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
        if self.params.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return EvaluationResult(
            action_log_probs=action_log_probs,
            state_values=state_values,
            dist_entropy=dist_entropy
        )


# Debug
if __name__ == "__main__":
    dac = DiscretePPOActorCritic(
        params=PPOActorCritic.Params(
            state_dim=4,
            action_dim=2,
            action_std_init=0.6
        )
    )
    x = torch.rand(dac.params.state_dim)
    print(x)
    act = dac.act(x)
    print(f'{act.action = }')
    print(f'{act.log_prob = }')
