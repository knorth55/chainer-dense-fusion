import numpy as np
import os.path as osp
from PIL import Image

from chainer.dataset import download
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv import utils
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation
import scipy.io


ycb_root = 'pfnet/chainer-dense-fusion/ycb'
tool_root = 'pfnet/chainer-dense-fusion/YCB_Video_toolbox'
this_dir = osp.dirname(osp.realpath(__file__))


class YCBVideoDatasetSegmented(GetterDataset):

    depth_scale = 10000

    def __init__(self, split='val'):
        if split not in ['val']:
            raise ValueError(
                'YCB dataset split {} is not supported'.format(split))
        super(YCBVideoDatasetSegmented, self).__init__()

        self.data_dir = download.get_dataset_directory(ycb_root)
        self.tool_dir = download.get_dataset_directory(tool_root)
        with open(osp.join(this_dir, './config/label_names.txt')) as f:
            self.label_names = f.read().split('\n')[:-1]

        with open(osp.join(this_dir, './config/{}.txt'.format(split))) as f:
            self.ids = f.read().split('\n')

        self.add_getter('img', self._get_image)
        self.add_getter('depth', self._get_depth)
        self.add_getter('label', self._get_label)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        imgpath = osp.join(self.data_dir, '{}-color.png'.format(self.ids[i]))
        img = utils.read_image(imgpath, dtype=np.float32, color=True)
        return img

    def _get_depth(self, i):
        depthpath = osp.join(self.data_dir, '{}-depth.png'.format(self.ids[i]))
        depth = np.asarray(Image.open(depthpath))
        depth = depth[None] / self.depth_scale
        return depth

    def _get_label(self, i):
        datapath = osp.join(
            self.tool_dir, './results_PoseCNN_RSS2018/{0:06d}.mat'.format(i))
        data = scipy.io.loadmat(datapath)
        label = np.array(data['labels'], dtype=np.int32)
        label = label
        return label

    def visualize(self, i):
        import matplotlib.pyplot as plt
        img = self._get_image(i)
        depth = self._get_depth(i)
        label = self._get_label(i)
        f, axes = plt.subplots(1, 3, sharey=True)
        vis_image(img, ax=axes[0])
        axes[1].imshow(depth[0])
        _, legend_handles = vis_semantic_segmentation(
            img, label, label_names=self.label_names, ax=axes[2])
        axes[2].legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
        plt.show()
