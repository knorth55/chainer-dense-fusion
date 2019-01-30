import argparse
import random

import chainer
from chainer.backends import cuda

from chainer_dense_fusion.datasets.ycb import YCBVideoDatasetPoseCNNSegmented
from chainer_dense_fusion.links.model import PoseNet
from chainer_dense_fusion.visualizations import vis_6d_pose_estimation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--pretrained-model')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # dataset
    dataset = YCBVideoDatasetPoseCNNSegmented(split='val')
    ids = list(range(len(dataset)))
    if args.random:
        random.shuffle(ids)

    # model
    model = PoseNet(
        n_fg_class=len(dataset.label_names),
        n_point=1000
    )
    chainer.serializers.load_npz(args.pretrained_model, model)
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    for i in ids:
        img, depth, lbl_img, _, _, intrinsic = dataset[i]
        poses, labels, _ = model.predict(
            [img], [depth], [lbl_img], [intrinsic])
        pose = poses[0]
        label = labels[0]

        camera_pcd = dataset.get_camera_pcd(i)
        object_pcds = []
        for lbl in label:
            object_pcds.append(dataset.get_object_pcd(lbl))
        vis_6d_pose_estimation(
            camera_pcd, object_pcds, pose)


if __name__ == '__main__':
    main()
