from typing import Optional

import classopt


@classopt.classopt(default_long=True)
class Config:
    # global
    train: bool = False
    test: bool = False
    render: bool = False
    seed: int = 42
    name: str = ''

    env: str = 'MountainCar-v0'
    disable_curiosity: bool = False
    max_episodes: int = 10000
    max_episode_length: int = 200

    # hyperparameters
    gamma: float = 0.99
    k_epochs: int = 20
    eps_clip: float = 0.2
    lr_actor: float = 0.001
    lr_critic: float = 0.003
    lr_curiosity: float = 0.001

    batch_size: int = 32
    buffer_update_size: int = 2048

    icm_eta: float = 0.01
    icm_lambda: float = 0.1
    icm_beta: float = 0.2
