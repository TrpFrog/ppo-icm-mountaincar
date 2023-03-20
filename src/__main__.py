from src.config import Config
from src.eval import evaluate
from src.train import train
from src import utils


if __name__ == '__main__':
    config = Config.from_args()
    utils.set_seed(config.seed)

    print(f'Using device: {utils.get_device()}')

    if config.train:
        train(config)
    elif config.test:
        evaluate(config)
