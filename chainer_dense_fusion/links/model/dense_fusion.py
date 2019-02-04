from __future__ import division

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv import utils
import numpy as np

from chainer_dense_fusion.links.model.posenet import PoseNet
from chainer_dense_fusion.utils import generate_organized_pcd
from chainer_dense_fusion.utils import generate_pose


class DenseFusion(chainer.Chain):

    _models = {
        'ycb_converted': {
            'param': {'n_fg_class': 21},
            'url': 'https://github.com/knorth55/'
            'chainer-dense-fusion/releases/download/v0.0.1/'
            'dense_fusion_ycb_converted_2019_02_03.npz',
            'cv2': True
        },
    }

    def __init__(
            self, pretrained_model=None,
            n_fg_class=21, n_point=1000, n_iter=2,
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ):
        super(DenseFusion, self).__init__()
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)
        self.n_fg_class = param['n_fg_class']
        self.n_point = n_point
        self.n_iter = n_iter

        with self.init_scope():
            self.posenet = PoseNet(
                None, n_fg_class, n_point, mean, std)
            self.refiner = PoseRefineNet(
                n_fg_class, n_point)

        if pretrained_model is not None:
            chainer.serializers.load_npz(path, self)

    def predict(self, imgs, depths, lbl_imgs, bboxes, bbox_labels, intrinsics):
        prepared_imgs = []
        for img in imgs:
            img = self.posenet.prepare(img.astype(np.float32))
            prepared_imgs.append(img)

        poses = []
        labels = []
        scores = []
        for img, depth, lbl_img, bbox, bbox_label, intrinsic in zip(
                prepared_imgs, depths, lbl_imgs,
                bboxes, bbox_labels, intrinsics):
            # generete organized pcd
            organized_pcd = generate_organized_pcd(depth, intrinsic)
            H, W = img.shape[1:]

            label = []
            pose = []
            score = []
            for bb, bb_lbl in zip(bbox, bbox_label):
                if bb_lbl < 0:
                    continue
                prepared_bb = self.posenet.prepare_bb(bb, H, W)
                if prepared_bb[0] is None:
                    continue
                pcd_indice = self.posenet.get_pcd_indice(
                    depth, lbl_img, prepared_bb, bb_lbl)
                if pcd_indice is None:
                    continue

                ymin, xmin, ymax, xmax = prepared_bb
                pcd = organized_pcd[:, ymin:ymax, xmin:xmax].reshape((3, -1))
                masked_img = img[:, ymin:ymax, xmin:xmax]
                masked_pcd = pcd[:, pcd_indice]
                rot, trans, conf, h_img_var = self.posenet.predict_each(
                    masked_img, masked_pcd, pcd_indice, bb_lbl)
                pse = generate_pose(rot, trans)

                # refiner
                for _ in range(self.n_iter):
                    # translate: (Tx, Ty, Tz).T, rotation: R.T
                    # (x', y', z').T = R.T.T * ((x, y, z).T - (Tx, Ty, Tz).T)
                    refine_pcd = masked_pcd - pse[3, :3, None]
                    refine_pcd = np.dot(pse[:3, :3], refine_pcd)
                    with chainer.using_config('train', False), \
                            chainer.function.no_backprop_mode():
                        refine_pcd_var = chainer.Variable(
                            self.xp.array(refine_pcd[None]))
                        refine_cls_rot, refine_cls_trans = \
                            self.refiner.__call__(h_img_var, refine_pcd_var)

                    # variable -> cpu array
                    refine_rot = cuda.to_cpu(refine_cls_rot.array)[0, bb_lbl]
                    refine_trans = cuda.to_cpu(
                        refine_cls_trans.array)[0, bb_lbl]
                    refine_rot = refine_rot / np.linalg.norm(refine_rot)
                    refine_pse = generate_pose(refine_rot, refine_trans)
                    pse = np.dot(refine_pse, pse)

                pose.append(pse[None])
                label.append(bb_lbl)
                score.append(conf)

            pose = np.concatenate(pose, axis=0)
            label = np.array(label)
            score = np.array(score)
            poses.append(pose)
            labels.append(label)
            scores.append(score)
        return poses, labels, scores


class PoseRefineNet(chainer.Chain):

    def __init__(self, n_fg_class, n_point):
        super(PoseRefineNet, self).__init__()
        self.n_fg_class = n_fg_class
        self.n_point = n_point

        with self.init_scope():
            self.extractor = PoseRefineNetExtractor(self.n_point)
            self.conv1_rot = L.Linear(1024, 512)
            self.conv1_trans = L.Linear(1024, 512)
            self.conv2_rot = L.Linear(512, 128)
            self.conv2_trans = L.Linear(512, 128)
            self.conv3_rot = L.Linear(128, self.n_fg_class * 4)
            self.conv3_trans = L.Linear(128, self.n_fg_class * 3)

    def __call__(self, h_img, pcd):
        B = pcd.shape[0]
        h = self.extractor(h_img, pcd)
        # conv1
        h_rot = F.relu(self.conv1_rot(h))
        h_trans = F.relu(self.conv1_trans(h))
        # conv2
        h_rot = F.relu(self.conv2_rot(h_rot))
        h_trans = F.relu(self.conv2_trans(h_trans))
        # conv3
        h_rot = self.conv3_rot(h_rot)
        h_trans = self.conv3_trans(h_trans)
        cls_rot = h_rot.reshape((B, self.n_fg_class, 4))
        cls_trans = h_trans.reshape((B, self.n_fg_class, 3))
        return cls_rot, cls_trans


class PoseRefineNetExtractor(chainer.Chain):

    def __init__(self, n_point):
        super(PoseRefineNetExtractor, self).__init__()
        self.n_point = n_point

        with self.init_scope():
            # conv1
            self.conv1_img = L.Convolution1D(32, 64, 1)
            self.conv1_pcd = L.Convolution1D(3, 64, 1)
            # conv2
            self.conv2_img = L.Convolution1D(64, 128, 1)
            self.conv2_pcd = L.Convolution1D(64, 128, 1)
            # conv3, conv4
            self.conv3 = L.Convolution1D(384, 512, 1)
            self.conv4 = L.Convolution1D(512, 1024, 1)

    def __call__(self, h_img, pcd):
        B = h_img.shape[0]
        # conv1
        h_img = F.relu(self.conv1_img(h_img))
        h_pcd = F.relu(self.conv1_pcd(pcd))
        feat1 = F.concat((h_pcd, h_img), axis=1)
        # conv2
        h_img = F.relu(self.conv2_img(h_img))
        h_pcd = F.relu(self.conv2_pcd(h_pcd))
        feat2 = F.concat((h_pcd, h_img), axis=1)
        # conv3, conv4
        feat3 = F.concat((feat1, feat2), axis=1)
        h = F.relu(self.conv3(feat3))
        h = F.relu(self.conv4(h))
        h = F.average_pooling_1d(h, self.n_point)
        h = h.reshape((B, 1024))
        return h
