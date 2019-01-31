import chainer
import chainer.functions as F
import chainer.links as L


class ResNet18Extractor(chainer.Chain):

    def __init__(self):
        super(ResNet18Extractor, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, nobias=True)
            self.res2 = ResBlock(2, 64, 64, 1, 1, residual_conv=False)
            self.res3 = ResBlock(2, 64, 128, 2, 1)
            self.res4 = ResBlock(2, 128, 256, 1, 2)
            self.res5 = ResBlock(2, 256, 512, 1, 4)

    def __call__(self, x):
        # 1 -> 1/2
        h = F.relu(self.conv1(x))
        # 1/2 -> 1/4
        h = F.max_pooling_2d(h, ksize=3, stride=2, pad=1, cover_all=False)
        # 1/4 -> 1/8
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        return h


class ResBlock(chainer.Chain):

    def __init__(self, n_layer, in_channels, out_channels,
                 stride, dilate, residual_conv=True):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.a = Bottleneck(
                in_channels, out_channels, stride, 1,
                residual_conv=residual_conv)
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = Bottleneck(
                    out_channels, out_channels, 1, dilate,
                    residual_conv=False)
                setattr(self, name, bottleneck)
        self.n_layer = n_layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(self.n_layer - 1):
            h = getattr(self, 'b{}'.format(i + 1))(h)
        return h


class Bottleneck(chainer.Chain):

    def __init__(self, in_channels, out_channels, stride, dilate,
                 initialW=None, residual_conv=False):
        super(Bottleneck, self).__init__()
        with self.init_scope():
            # pad = dilate
            self.conv1 = L.Convolution2D(
                in_channels, out_channels, 3, stride,
                pad=dilate, dilate=dilate, nobias=True)
            self.conv2 = L.Convolution2D(
                out_channels, out_channels, 3, 1,
                pad=dilate, dilate=dilate, nobias=True)
            if residual_conv:
                self.residual_conv = L.Convolution2D(
                    in_channels, out_channels, 1, stride,
                    nobias=True)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)

        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(x)
        else:
            residual = x
        h = h + residual
        h = F.relu(h)
        return h
