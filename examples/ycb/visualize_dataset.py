import argparse
import random

from chainer_dense_fusion.datasets.ycb import YCBVideoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--cad', action='store_true')
    args = parser.parse_args()

    dataset = YCBVideoDataset(split='val')
    ids = list(range(len(dataset)))
    if args.random:
        random.shuffle(ids)

    for i in ids:
        if args.cad:
            dataset.visualize_3d(i)
        else:
            dataset.visualize(i)


if __name__ == '__main__':
    main()
