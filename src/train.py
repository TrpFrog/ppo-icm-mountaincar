import dataclasses

import gym
import numpy as np
import typing
import wandb
import torch
from torch import Tensor
from tqdm import trange

from src.config import Config
from src.icm import DiscreteICM
from src.ppo import DiscretePPO
from src import utils

device = utils.get_device()


def train(config: Config):
    render_mode = 'rgb_array' if config.render else None
    with gym.make(config.env, render_mode=render_mode) as env:

        episodes = config.max_episodes
        max_every_step = env.spec.max_episode_steps

        agent_params = DiscretePPO.Params(
            gamma_discount=config.gamma,
            k_epochs=config.k_epochs,
            eps_clip=config.eps_clip,
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            evaluation_batch_size=config.batch_size,
        )

        curiosity = DiscreteICM(
            state_dim=agent_params.state_dim,
            action_dim=agent_params.action_dim,
            reward_scale=config.icm_eta,
            policy_loss_weight=config.icm_lambda,
            loss_scale=config.icm_beta,
            hidden_size=64,
        )

        if config.disable_curiosity:
            curiosity = None
            print('Curiosity disabled')

        agent = DiscretePPO(params=agent_params, curiosity=curiosity).to(device)
        if hasattr(torch, 'compile'):
            agent = torch.compile(agent)
            agent = typing.cast(DiscretePPO, agent)

        wandb.init(
            project='ppo-test-cartpole',
            name=config.name or None,
            config=config.to_dict()
        )

        try:
            for i in range(episodes):
                obs, info = env.reset()
                obs: Tensor = torch.from_numpy(obs).float().to(device)

                for t in trange(max_every_step, desc=f'Playing episode {i}'):
                    if config.render:
                        env.render()
                    # convert obs to tensor
                    action = agent.select_action(obs)

                    obs, reward, terminated, truncated, info = env.step(action)
                    obs = torch.from_numpy(obs).float().to(device)

                    if truncated:
                        # If the episode is truncated (reach the max step),
                        # it adds a negative reward to the last step.
                        reward += config.truncated_negative_reward

                    done = terminated or truncated

                    # In record_step(),
                    # the curiosity reward is automatically calc and add to the reward.
                    agent.record_step(
                        reward=reward,
                        next_state=obs,
                        is_terminal=done,
                    )

                    if done:
                        break

                reward_sum = np.sum(agent.buffer.rewards[-(t + 1):])
                cur_reward_sum = np.sum(agent.buffer.rewards_with_curiosity[-(t + 1):])

                if config.env == 'MountainCar-v0':
                    max_x, max_v = torch.stack(agent.buffer.states[-(t + 1):], dim=0).max(dim=0).values
                    max_v, max_x = max_v.item(), max_x.item()
                    print(f'{max_v = :.2f} {max_x = :.2f} (goal: x = 0.5)')

                if len(agent.buffer.rewards) > config.buffer_update_size:
                    info = agent.update()

                info.update({'total_rewards': cur_reward_sum})
                info.update({'steps': t + 1})

                print(info)

                wandb.log(info)
                print(f'Episode {i} finished after {t + 1} time steps '
                      f'with reward {reward_sum:.2f} (with curiosity {cur_reward_sum:.2f}))')
        finally:
            # save parameters
            torch.save(agent.state_dict(), f'{config.name}.pth')

