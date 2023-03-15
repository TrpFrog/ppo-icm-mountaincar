from src.config import Config
from src.train import train


if __name__ == '__main__':
    args = Config.from_args()
    if args.train:
        train(args)
    elif args.test:
        print('test')
