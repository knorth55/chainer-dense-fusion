import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainercv.utils.mask import mask_to_bbox

from chainer_dense_fusion.links.model.pspnet import PSPNetExtractor
from chainer_dense_fusion.links.model.resnet import ResNet18Extractor


def quaternion_to_rotation_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=q.dtype)
    return rot_matrix


class PoseNet(chainer.Chain):

    def __init__(
            self, n_fg_class=21, n_point=1000,
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ):
        super(PoseNet, self).__init__()
        with self.init_scope():
            # extractor
            self.resnet_extractor = ResNet18Extractor()
            self.pspnet_extractor = PSPNetExtractor()
            self.posenet_extractor = PoseNetExtractor(n_point)
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
            self.conv4_rot = L.Convolution1D(128, n_fg_class * 4, 1)
            self.conv4_trans = L.Convolution1D(128, n_fg_class * 3, 1)
            self.conv4_conf = L.Convolution1D(128, n_fg_class, 1)

        self.n_fg_class = n_fg_class
        self.n_point = n_point
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

    def __call__(self, img, pcd, pcd_indice):
        assert img.shape[0] == 1
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
        cls_conf = self.conv4_conf(h_conf)
        cls_rot = cls_rot.reshape((B, self.n_fg_class, 4, self.n_point))
        cls_trans = cls_trans.reshape((B, self.n_fg_class, 3, self.n_point))
        cls_conf = cls_conf.reshape((B, self.n_fg_class, self.n_point))
        return cls_rot, cls_trans, cls_conf

    def prepare(self, img):
        img = ((img - self.mean) / self.std).astype(np.float32, copy=False)
        return img

    def predict(self, imgs, depths, lbl_imgs, intrinsics):
        prepared_imgs = []
        for img in imgs:
            img = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)

        transforms = []
        labels = []
        scores = []
        for img, depth, lbl_img, intrinsic in zip(
                imgs, depths, lbl_imgs, intrinsics):
            # generete organized pcd
            H, W = img.shape[1:]
            fx, fy, cx, cy = intrinsic
            xmap = np.array(
                [[j for i in range(W)] for j in range(H)],
                dtype=np.float32)
            ymap = np.array(
                [[i for i in range(W)] for j in range(H)],
                dtype=np.float32)
            pcd_x = (ymap - cx) * depth / fx
            pcd_y = (xmap - cy) * depth / fy
            organized_pcd = np.concatenate(
                (pcd_x[None], pcd_y[None], depth[None]), axis=0)

            label = []
            transform = []
            score = []
            for lbl in np.unique(lbl_img):
                if lbl < 0:
                    continue
                bb = mask_to_bbox((lbl_img == lbl)[None])[0]
                ymin, xmin, ymax, xmax = bb
                masked_img = img[:, ymin:ymax, xmin:xmax]
                msk = np.logical_and(lbl_img == lbl, depth != 0)
                pcd_indice = np.where(msk[ymin:ymax, xmin:xmax].flatten())[0]
                if len(pcd_indice) > self.n_point:
                    pcd_indice_msk = np.zeros(len(pcd_indice), dtype=bool)
                    pcd_indice_msk[:self.n_point] = True
                    pcd_indice = pcd_indice[
                        np.random.permutation(pcd_indice_msk)]
                else:
                    pcd_indice = np.pad(
                        pcd_indice,
                        (0, self.n_point - len(pcd_indice)), 'wrap')
                pcd = organized_pcd[:, ymin:ymax, xmin:xmax].flatten()
                masked_pcd = pcd[:, pcd_indice]

                with chainer.using_config('train', False), \
                        chainer.function.no_backprop_mode():
                    masked_img_var = chainer.Variable(
                        self.xp.array(masked_img[None]))
                    masked_pcd_var = chainer.Variable(
                        self.xp.array(masked_pcd[None]))
                    pcd_indice_var = chainer.Variable(
                            self.xp.array(pcd_indice[None]))
                    cls_rot, cls_trans, cls_conf = \
                        self.__call__(
                            masked_img_var, masked_pcd_var, pcd_indice_var)
                # (B, C, 4, N) -> (N, 4)
                rot = cls_rot[0, lbl].transpose((1, 0))
                rot = rot / self.xp.norm(rot, axis=1)[:, None]
                # (B, C, 3, N) -> (N, 3)
                trans = cls_trans[0, lbl].transpose((1, 0))
                trans = trans + pcd.transpose((1, 0))
                # (B, C, N) -> (N, )
                conf = cls_conf[0, lbl]
                maxid = self.xp.argmax(conf)
                max_conf = conf[maxid]
                max_rot = cuda.to_cpu(rot[maxid])
                max_trans = cuda.to_cpu(trans[maxid])
                rot_matrix = quaternion_to_rotation_matrix(max_rot)
                rot_matrix[3, :3] = max_trans

                label.append(label)
                transform.append(rot_matrix)
                score.append(max_conf)

            transform = np.concatenate(transform, axis=0)
            label = np.concatenate(label, axis=0)
            score = np.concatenate(score, axis=0)
            transforms.append(transform)
            labels.append(label)
            scores.append(score)
        return transforms, labels, scores


class PoseNetExtractor(chainer.Chain):

    def __init__(self, n_point):
        super(PoseNetExtractor, self).__init__()
        with self.init_scope():
            # conv1
            self.conv1_pcd = L.Convolution1D(3, 64, 1)
            self.conv1_img = L.Convolution1D(32, 64, 1)
            # conv2
            self.conv2_pcd = L.Convolution1D(64, 128, 1)
            self.conv2_img = L.Convolution1D(64, 128, 1)
            # conv3, conv4
            self.conv3 = L.Convolution1D(256, 512, 1)
            self.conv4 = L.Convolution1D(512, 1024, 1)

        self.n_point = n_point

    def __call__(self, h_img, pcd):
        # conv1
        h_pcd = F.relu(self.conv1_pcd(pcd))
        h_img = F.relu(self.conv1_img(h_img))
        feat1 = F.concat((h_pcd, h_img), axis=1)
        # conv2
        h_pcd = F.relu(self.conv2_pcd(h_pcd))
        h_img = F.relu(self.conv2_img(h_img))
        feat2 = F.concat((h_pcd, h_img), axis=1)
        # conv3, conv4
        h = F.relu(self.conv3(feat2))
        h = F.relu(self.conv4(h))
        h = F.average_pooling_1d(h, self.n_point)
        B = h.shape[0]
        h = h.reshape((B, 1024, 1))
        feat3 = h.repeat(h, self.n_point, axis=2)
        feat = F.concat((feat1, feat2, feat3), axis=1)
        return feat
