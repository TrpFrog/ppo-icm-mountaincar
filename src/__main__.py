import torch
import transformers

from src.config import Config
from src.eval import evaluate
from src.train import train
from src import utils


if __name__ == '__main__':
    config = Config.from_args()
    transformers.set_seed(config.seed)

    # Set device globally
    device = 'cpu' if config.cpu else utils.get_device()
    torch.set_default_device(device)
    print(f'Using device: {device}')

    if config.train:
        train(config)
    elif config.test:
        evaluate(config)
