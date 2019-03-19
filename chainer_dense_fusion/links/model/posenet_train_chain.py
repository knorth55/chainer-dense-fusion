from __future__ import division

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv import utils
import numpy as np

from chainer_dense_fusion.utils import generate_organized_pcd
from chainer_dense_fusion.utils import generate_pose


class PoseNetTrainChain(chainer.Chain):

    def __init__(self, posenet):
        super(PoseNetTrainChain, self).__init__()
        self.posenet = posenet

    def __call__(
        self, imgs, depths, lbl_imgs, bboxes, bbox_labels,
        poses, pose_labels, intrinsics
    ):
        prepared_imgs = []
        for img in imgs:
            img = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)

        for img, depth, lbl_img, bbox, bbox_label, \
            pose, pose_label, intrinsic in zip(
                imgs, depths, lbl_imgs, bboxes, bbox_labels,
                poses, pose_labels, intrinsics):

            # generete organized pcd
            organized_pcd = generate_organized_pcd(depth, intrinsic)
            H, W = img.shape[1:]

            for bb, bb_lbl in zip(bbox, bbox_label):
                pcd_indice = self.get_pcd_indice(
                    depth, lbl_img, bb, bb_lbl)
                if pcd_indice is None:
                    continue

                ymin, xmin, ymax, xmax = bb
                pcd = organized_pcd[:, ymin:ymax, xmin:xmax].reshape((3, -1))
                masked_img = img[:, ymin:ymax, xmin:xmax]
                masked_pcd = pcd[:, pcd_indice]
                masked_img_var = chainer.Variable(
                    self.xp.array(masked_img[None]))
                masked_pcd_var = chainer.Variable(
                    self.xp.array(masked_pcd[None]))
                pcd_indice_var = chainer.Variable(
                    self.xp.array(pcd_indice[None]))
                cls_rot, cls_trans, cls_conf, h_img = self.posenet(
                    masked_img_var, masked_pcd_var, pcd_indice_var)

                rot = cls_rot[0, bb_lbl]
                rot = rot.transpose((1, 0))
                rot = rot / np.linalg.norm(rot, axis=1)[:, None]
                trans = cls_trans[0, bb_lbl]
                trans = trans.transpose((1, 0))
                conf = cls_conf[0, bb_lbl]
