from __future__ import division

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv import utils
import numpy as np

from chainer_dense_fusion.links.model.pspnet import PSPNetExtractor
from chainer_dense_fusion.links.model.resnet import ResNet18Extractor
from chainer_dense_fusion.utils import generate_organized_pcd
from chainer_dense_fusion.utils import generate_pose


class PoseNet(chainer.Chain):
    _models = {
        'ycb_converted': {
            'param': {'n_fg_class': 21},
            'url': 'https://github.com/knorth55/'
            'chainer-dense-fusion/releases/download/v0.0.0/'
            'posenet_ycb_converted_2019_02_01.npz',
            'cv2': True
        },
    }

    def __init__(
            self, pretrained_model=None,
            n_fg_class=21, n_point=1000,
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ):
        super(PoseNet, self).__init__()
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)
        self.n_fg_class = param['n_fg_class']
        self.n_point = n_point
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

        with self.init_scope():
            # extractor
            self.resnet_extractor = ResNet18Extractor()
            self.pspnet_extractor = PSPNetExtractor()
            self.posenet_extractor = PoseNetExtractor(self.n_point)
            # conv1
            self.conv1_rot = L.Convolution1D(1408, 640, 1)
            self.conv1_trans = L.Convolution1D(1408, 640, 1)
            self.conv1_conf = L.Convolution1D(1408, 640, 1)
            # conv2
            self.conv2_rot = L.Convolution1D(640, 256, 1)
            self.conv2_trans = L.Convolution1D(640, 256, 1)
            self.conv2_conf = L.Convolution1D(640, 256, 1)
            # conv3
            self.conv3_rot = L.Convolution1D(256, 128, 1)
            self.conv3_trans = L.Convolution1D(256, 128, 1)
            self.conv3_conf = L.Convolution1D(256, 128, 1)
            # conv4
            self.conv4_rot = L.Convolution1D(128, self.n_fg_class * 4, 1)
            self.conv4_trans = L.Convolution1D(128, self.n_fg_class * 3, 1)
            self.conv4_conf = L.Convolution1D(128, self.n_fg_class, 1)

        if pretrained_model is not None:
            chainer.serializers.load_npz(path, self)

    def __call__(self, img, pcd, pcd_indice):
        assert img.shape[0] == 1
        pcd_indice = pcd_indice.array[0]

        # resnet extractor
        h_img = self.resnet_extractor(img)
        # pspnet extractor
        h_img = self.pspnet_extractor(h_img)
        # posenet extractor
        B, C = h_img.shape[:2]
        h_img = h_img.reshape((B, C, -1))
        h_img = h_img[:, :, pcd_indice]

        h = self.posenet_extractor(h_img, pcd)

        # conv1
        h_rot = F.relu(self.conv1_rot(h))
        h_trans = F.relu(self.conv1_trans(h))
        h_conf = F.relu(self.conv1_conf(h))

        # conv2
        h_rot = F.relu(self.conv2_rot(h_rot))
        h_trans = F.relu(self.conv2_trans(h_trans))
        h_conf = F.relu(self.conv2_conf(h_conf))
        # conv3
        h_rot = F.relu(self.conv3_rot(h_rot))
        h_trans = F.relu(self.conv3_trans(h_trans))
        h_conf = F.relu(self.conv3_conf(h_conf))
        # conv4
        cls_rot = self.conv4_rot(h_rot)
        cls_trans = self.conv4_trans(h_trans)
        cls_conf = F.sigmoid(self.conv4_conf(h_conf))

        cls_rot = cls_rot.reshape((B, self.n_fg_class, 4, self.n_point))
        cls_trans = cls_trans.reshape((B, self.n_fg_class, 3, self.n_point))
        cls_conf = cls_conf.reshape((B, self.n_fg_class, self.n_point))
        return cls_rot, cls_trans, cls_conf, h_img

    def prepare(self, img):
        img = ((img - self.mean) / self.std).astype(np.float32, copy=False)
        return img

    def prepare_bb(self, bb, H, W):
        if bb[2] - bb[0] < 40 or bb[3] - bb[1] < 40:
            return None, None, None, None
        bb = bb.copy()
        bb[:2] = bb[:2].astype(np.int32) + 1
        bb[2:] = bb[2:].astype(np.int32) - 1
        bb = bb.astype(np.int32)
        bb_h = ((bb[2] - bb[0]) // 40 + 1) * 40
        bb_w = ((bb[3] - bb[1]) // 40 + 1) * 40
        bb_yc = ((bb[2] + bb[0]) / 2).astype(np.int32)
        bb_xc = ((bb[3] + bb[1]) / 2).astype(np.int32)
        ymin = bb_yc - (bb_h / 2).astype(np.int32)
        ymax = bb_yc + (bb_h / 2).astype(np.int32)
        xmin = bb_xc - (bb_w / 2).astype(np.int32)
        xmax = bb_xc + (bb_w / 2).astype(np.int32)
        if ymin < 0:
            ymax = ymax - ymin
            ymin = 0
        if ymax > H:
            ymin = ymin - ymax + H
            ymax = H
        if xmin < 0:
            xmax = xmax - xmin
            xmin = 0
        if xmax > W:
            xmin = xmin - xmax + W
            xmax = W
        return ymin, xmin, ymax, xmax

    def get_pcd_indice(self, depth, lbl_img, bb, bb_lbl):
        ymin, xmin, ymax, xmax = bb
        msk = np.logical_and(lbl_img == bb_lbl, depth != 0)
        pcd_indice = np.where(msk[ymin:ymax, xmin:xmax].flatten())[0]

        if len(pcd_indice) == 0:
            return None
        if len(pcd_indice) > self.n_point:
            pcd_indice_msk = np.zeros(len(pcd_indice), dtype=bool)
            pcd_indice_msk[:self.n_point] = True
            pcd_indice = pcd_indice[
                np.random.permutation(pcd_indice_msk)]
        else:
            pcd_indice = np.pad(
                pcd_indice,
                (0, self.n_point - len(pcd_indice)), 'wrap')
        return pcd_indice

    def predict_each(self, masked_img, masked_pcd, pcd_indice, bb_lbl):
        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            masked_img_var = chainer.Variable(
                self.xp.array(masked_img[None]))
            masked_pcd_var = chainer.Variable(
                self.xp.array(masked_pcd[None]))
            pcd_indice_var = chainer.Variable(
                    self.xp.array(pcd_indice[None]))
            cls_rot, cls_trans, cls_conf, h_img = \
                self.__call__(
                    masked_img_var, masked_pcd_var, pcd_indice_var)

        # variable -> cpu array
        rot = cuda.to_cpu(cls_rot.array)[0, bb_lbl]
        trans = cuda.to_cpu(cls_trans.array)[0, bb_lbl]
        conf = cuda.to_cpu(cls_conf.array)[0, bb_lbl]

        # (B, C, 4, N) -> (N, 4)
        rot = rot.transpose((1, 0))
        rot = rot / np.linalg.norm(rot, axis=1)[:, None]
        # (B, C, 3, N) -> (N, 3)
        trans = trans.transpose((1, 0))
        trans = trans + masked_pcd.transpose((1, 0))

        # get max conf value
        maxid = np.argmax(conf)
        max_rot = rot[maxid]
        max_trans = trans[maxid]
        max_conf = conf[maxid]
        return max_rot, max_trans, max_conf, h_img

    def predict(self, imgs, depths, lbl_imgs, bboxes, bbox_labels, intrinsics):
        prepared_imgs = []
        for img in imgs:
            img = self.prepare(img.astype(np.float32))
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
                prepared_bb = self.prepare_bb(bb, H, W)
                if prepared_bb[0] is None:
                    continue
                pcd_indice = self.get_pcd_indice(
                    depth, lbl_img, prepared_bb, bb_lbl)
                if pcd_indice is None:
                    continue

                ymin, xmin, ymax, xmax = prepared_bb
                pcd = organized_pcd[:, ymin:ymax, xmin:xmax].reshape((3, -1))
                masked_img = img[:, ymin:ymax, xmin:xmax]
                masked_pcd = pcd[:, pcd_indice]
                rot, trans, conf, _ = self.predict_each(
                    masked_img, masked_pcd, pcd_indice, bb_lbl)
                pse = generate_pose(rot, trans)

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


class PoseNetExtractor(chainer.Chain):

    def __init__(self, n_point):
        super(PoseNetExtractor, self).__init__()
        with self.init_scope():
            # conv1
            self.conv1_img = L.Convolution1D(32, 64, 1)
            self.conv1_pcd = L.Convolution1D(3, 64, 1)
            # conv2
            self.conv2_img = L.Convolution1D(64, 128, 1)
            self.conv2_pcd = L.Convolution1D(64, 128, 1)
            # conv3, conv4
            self.conv3 = L.Convolution1D(256, 512, 1)
            self.conv4 = L.Convolution1D(512, 1024, 1)

        self.n_point = n_point

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
        h = F.relu(self.conv3(feat2))
        h = F.relu(self.conv4(h))
        h = F.average_pooling_1d(h, self.n_point)
        h = h.reshape((B, 1024, 1))
        feat3 = F.repeat(h, self.n_point, axis=2)
        feat = F.concat((feat1, feat2, feat3), axis=1)
        return feat
