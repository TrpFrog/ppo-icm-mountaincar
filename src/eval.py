import gym
import torch
import tqdm
import typing

from src.config import Config
from src.ppo import DiscretePPO
from src import utils

device = utils.get_device()


@torch.inference_mode()
def evaluate(config: Config):
    render_mode = 'human'
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

        agent = DiscretePPO(params=agent_params).to(device)
        agent = torch.compile(agent)
        agent = typing.cast(DiscretePPO, agent)

        agent.load_state_dict(
            torch.load(config.path, map_location=device),
            strict=False
        )

        for i in range(episodes):
            obs, info = env.reset()

            for t in tqdm.trange(max_every_step):
                if config.render:
                    env.render()
                # convert obs to tensor
                obs = torch.from_numpy(obs).float().to(device)
                action = agent.select_action(obs, greedy=True)
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            print(f'Episode {i} finished after {t + 1} time steps')

