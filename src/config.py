from classopt import classopt, config


@classopt(default_long=True)
class Config:
    # global
    train: bool = False
    test: bool = False
    render: bool = config(default=False, help='render environment in train mode')
    seed: int = config(default=42, help='random seed')
    name: str = config(default='', help='name of the run, used for wandb and model saving')
    env: str = 'MountainCar-v0'

    # test
    path: str = config(default='', help='path to model to test')

    # train
    disable_curiosity: bool = config(default=False, help='stop using curiosity module')
    max_episodes: int = config(default=10000, help='max episodes to train')
    truncated_negative_reward: float = config(default=-100, help='negative reward for truncated episode')

    # hyperparameters
    gamma: float = config(default=0.99, help='discount factor')
    k_epochs: int = config(default=10, help='number of epochs on each PPO update')
    eps_clip: float = config(default=0.2, help='clip parameter for PPO')
    lr_actor: float = config(default=0.001, help='learning rate for actor')
    lr_critic: float = config(default=0.003, help='learning rate for critic')
    lr_curiosity: float = config(default=0.001, help='learning rate for ICM')

    batch_size: int = config(default=32, help='batch size for PPO')
    buffer_update_size: int = config(default=2048, help='steps to update buffer')

    icm_eta: float = config(default=0.01, help='reward scale for ICM')
    icm_lambda: float = config(default=0.1, help='policy loss weight for ICM')
    icm_beta: float = config(default=0.2, help='balance parameter for ICM loss '
                                               'between forward and inverse model')
