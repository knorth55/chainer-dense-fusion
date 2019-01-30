import argparse
import random

from chainer_dense_fusion.datasets.ycb import YCBVideoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()

    dataset = YCBVideoDataset(split='val')
    ids = list(range(len(dataset)))
    if args.random:
        random.shuffle(ids)

    for i in ids:
        dataset.visualize(i)
        dataset.visualize_3d(i)


if __name__ == '__main__':
    main()
