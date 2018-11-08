import argparse

from lib.model.abstract import AbstractModel

NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 5


def main(batch_size, num_epochs, num_classes):
    model = AbstractModel(batch_size, num_epochs, num_classes)
    with model:
        model.train()
        model.evaluate()


def parse_arguments():
    parser = argparse.ArgumentParser('Model training script.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    args_ = parser.parse_args()
    return vars(args_)


if __name__ == '__main__':
    args = parse_arguments()
    main(**args)
