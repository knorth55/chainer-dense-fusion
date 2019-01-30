import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class PoseNet(chainer.Chain):

    def __init__(self, n_fg_class=21, n_point=1000):
        super(PoseNet, self).__init__()
        with self.init_scope():
            # extractor
            self.pspnet_extractor = PSPNetResNet18Extractor()
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
        # pspnet extractor
        h_img = self.pspnet_extractor(img)
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


class PSPNetResNet18Extractor(chainer.Chain):

    def __init__(self):
        super(PSPNetResNet18Extractor, self).__init__()
        sizes = [1, 2, 3, 6]
        with self.init_scope():
            self.extractor = ResNet18Extractor()
            self.psp = PSPModule(512, 1024, sizes)
            # 1/8 -> 1/4
            self.up1 = PSPUpsample(1024, 256)
            # 1/4 -> 1/2
            self.up2 = PSPUpsample(256, 64)
            # 1/2 -> 1
            self.up3 = PSPUpsample(64, 64)
            self.conv1 = L.Convolution2D(64, 32, 1)

    def __call__(self, img):
        # resnet extractor
        h = self.extractor(img)
        # psp module
        h = self.psp(h)
        h = F.dropout(h, 0.3)
        # upsample
        h = F.dropout(self.up1(h), 0.15)
        h = F.dropout(self.up2(h), 0.15)
        h = self.up3(h)
        # head
        h = self.conv1(h)
        feat = F.log_softmax(h)
        return feat


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


class ResNet18Extractor(chainer.Chain):

    def __init__(self):
        super(ResNet18Extractor, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, nobias=True)
            self.res2 = ResBlock(2, 64, 64, 1, 1, residual_conv=False)
            self.res3 = ResBlock(2, 64, 128, 2, 1)
            self.res4 = ResBlock(2, 128, 256, 1, 2)
            self.res5 = ResBlock(2, 256, 512, 1, 4)

    def __call__(self, img):
        # 1 -> 1/2
        h = self.relu(self.conv1(img))
        # 1/2 -> 1/4
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        # 1/4 -> 1/8
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        return h


class PSPModule(chainer.Chain):

    def __init__(self, in_channels, out_channels, sizes):
        super(PSPModule, self).__init__()
        with self.init_scope():
            for i in range(len(sizes)):
                setattr(
                    self, 'conv{}'.format(i+1),
                    L.Convolution2D(in_channels, in_channels, 1, nobias=True))
            self.bottleneck = L.Convolution2D(
                in_channels * (len(sizes) + 1), out_channels, 1)
        self.sizes = sizes

    def __call__(self, x):
        H, W = x.shape[2:]
        kh = H // np.array(self.sizes)
        kw = W // np.array(self.sizes)
        ksizes = list(zip(kh, kw))

        # extract
        hs = [x]
        for i, ksize in enumerate(ksizes):
            h = F.average_pooling_2d(x, ksize, ksize)
            h = getattr(self, 'conv{}'.format(i+1))(h)
            h = F.resize_images(h, (H, W))
            hs.append(h)
        h = F.relu(self.bottleneck(F.concat(hs, axis=1)))
        return h


class PSPUpsample(chainer.Chain):

    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, 3, 1)
            self.prelu = L.PReLU((1,))

    def __call___(self, x):
        H, W = x.shape[2:]
        h = F.resize_images(x, (H*2, W*2))
        h = self.prelu(self.conv(h))
        return h


class ResBlock(chainer.Chain):

    def __init__(self, n_layer, in_channels, out_channels,
                 stride, dilate=1, residual_conv=True, initialW=None):
        super(ResBlock, self).__init__()
        # Dilate option is applied to all bottlenecks.
        with self.init_scope():
            self.a = Bottleneck(
                in_channels, out_channels, stride, dilate,
                initialW=initialW, residual_conv=residual_conv)
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = Bottleneck(
                    out_channels, out_channels, 1, dilate,
                    initialW=initialW, residual_conv=False)
                setattr(self, name, bottleneck)


class Bottleneck(chainer.Chain):

    def __init__(self, in_channels, out_channels,
                 stride=1, dilate=1, initialW=None,
                 residual_conv=False):
        super(Bottleneck, self).__init__()
        with self.init_scope():
            # pad = dilate
            self.conv1 = L.Convolution2D(
                in_channels, out_channels, 3, stride,
                pad=dilate, dilate=dilate,
                nobias=True, initialW=initialW)
            self.conv2 = L.Convolution2D(
                out_channels, out_channels, 3, 1,
                pad=dilate, dilate=dilate,
                nobias=True, initialW=initialW)
            if residual_conv:
                self.residual_conv = L.Convolution2D(
                    in_channels, out_channels, 1, stride, 0,
                    nobias=True, initialW=initialW)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)

        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(x)
        else:
            residual = x
        h += residual
        h = F.relu(h)
        return h
