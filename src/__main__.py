from src.config import Config
from src.eval import evaluate
from src.train import train
from src import utils


if __name__ == '__main__':
    args = Config.from_args()
    if args.cpu:
        utils.get_device = lambda: 'cpu'
    print(f'Using device: {utils.get_device()}')

    if args.train:
        train(args)
    elif args.test:
        evaluate(args)
