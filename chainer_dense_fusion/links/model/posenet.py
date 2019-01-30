import chainer
import chainer.functions as F
import chainer.links as L

from chainer_dense_fusion.links.model.pspnet import PSPNetExtractor
from chainer_dense_fusion.links.model.resnet import ResNet18Extractor


class PoseNet(chainer.Chain):

    def __init__(self, n_fg_class=21, n_point=1000):
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

    def __call__(self, img, pcd, pcd_indice, lbl):
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
        h_rot = self.conv4_rot(h_rot)
        h_trans = self.conv4_trans(h_trans)
        h_conf = self.conv4_conf(h_conf)
        h_rot = h_rot.reshape((B, self.n_fg_class, 4, self.n_point))
        h_trans = h_trans.reshape((B, self.n_fg_class, 3, self.n_point))
        h_conf = h_conf.reshape((B, self.n_fg_class, 1, self.n_point))

        # batch_size = 0
        rot = h_rot[0, lbl]
        trans = h_trans[0, lbl]
        conf = h_conf[0, lbl]
        return rot, trans, conf

    def predict(self, img, depth, lbl_img):
        return 0


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
