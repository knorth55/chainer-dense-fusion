import numpy as np
import os.path as osp
from PIL import Image

from chainer.dataset import download
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv import utils
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation
import matplotlib.pyplot as plt
import open3d
from open3d import create_point_cloud_from_rgbd_image
from open3d import create_rgbd_image_from_color_and_depth
from open3d import PinholeCameraIntrinsic
import scipy.io


ycb_root = 'pfnet/chainer-dense-fusion/ycb'
tool_root = 'pfnet/chainer-dense-fusion/YCB_Video_toolbox'
this_dir = osp.dirname(osp.realpath(__file__))


class YCBVideoDataset(GetterDataset):

    def __init__(self, split='train'):
        if split not in ['train', 'val']:
            raise ValueError(
                'YCB video dataset split {} is not supported'.format(split))
        super(YCBVideoDataset, self).__init__()

        self.data_dir = download.get_dataset_directory(ycb_root)
        self.tool_dir = download.get_dataset_directory(tool_root)
        with open(osp.join(this_dir, './config/label_names.txt')) as f:
            self.label_names = f.read().split('\n')[:-1]

        with open(osp.join(this_dir, './config/{}.txt'.format(split))) as f:
            self.ids = f.read().split('\n')

        self.add_getter('img', self._get_image)
        self.add_getter('depth', self._get_depth)
        self.add_getter('label', self._get_label)
        self.add_getter('pose', self._get_pose)
        self.add_getter('intrinsic', self._get_intrinsic)
        self.keys = ('img', 'depth', 'label', 'pose', 'intrinsic')

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        imgpath = osp.join(self.data_dir, '{}-color.png'.format(self.ids[i]))
        img = utils.read_image(imgpath, dtype=np.float32, color=True)
        return img

    def _get_depth(self, i):
        depthpath = osp.join(self.data_dir, '{}-depth.png'.format(self.ids[i]))
        depth = np.asarray(Image.open(depthpath)).astype(np.float32)
        depth_scale = self._get_depth_scale(i)
        depth = depth[None] / depth_scale
        return depth

    def _get_label(self, i):
        metapath = osp.join(self.data_dir, '{}-meta.mat'.format(self.ids[i]))
        object_ids = scipy.io.loadmat(metapath)['cls_indexes'].flatten()
        object_ids = object_ids - 1
        return object_ids

    def _get_pose(self, i):
        metapath = osp.join(self.data_dir, '{}-meta.mat'.format(self.ids[i]))
        rt = scipy.io.loadmat(metapath)['poses'].transpose((2, 0, 1))
        pose = np.zeros((len(rt), 4, 4), dtype=np.float32)
        pose[:, 3, 3] = 1
        pose[:, :3, :3] = rt[:, :, :3]
        pose[:, :3, 3] = rt[:, :, 3]
        pose = pose.transpose((0, 2, 1))
        return pose

    def _get_depth_scale(self, i):
        metapath = osp.join(self.data_dir, '{}-meta.mat'.format(self.ids[i]))
        depth_scale = scipy.io.loadmat(metapath)['factor_depth'][0][0]
        return depth_scale

    def _get_intrinsic(self, i):
        metapath = osp.join(self.data_dir, '{}-meta.mat'.format(self.ids[i]))
        intrinsic_matrix = scipy.io.loadmat(metapath)['intrinsic_matrix']
        fx = intrinsic_matrix[0][0]
        fy = intrinsic_matrix[1][1]
        cx = intrinsic_matrix[0][2]
        cy = intrinsic_matrix[1][2]
        return fx, fy, cx, cy

    def get_object_pcd(self, lbl, pse):
        xyzpath = osp.join(
            self.data_dir,
            './models/{}/points.xyz'.format(self.label_names[lbl]))
        obj_pcd = open3d.read_point_cloud(xyzpath)
        obj_pcd.transform(pse.transpose((1, 0)))
        obj_pcd.transform(
            [[1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]])
        return obj_pcd

    def get_camera_pcd(self, i):
        img = self._get_image(i)
        depth = self._get_depth(i)
        _, H, W = img.shape
        fx, fy, cx, cy = self._get_intrinsic(i)
        intrinsic = PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        img = open3d.Image(img.transpose((1, 2, 0)).astype(np.uint8))
        depth = open3d.Image(depth[0])
        rgbd = create_rgbd_image_from_color_and_depth(
            img, depth, depth_scale=1.0, convert_rgb_to_intensity=False)
        pcd = create_point_cloud_from_rgbd_image(rgbd, intrinsic)
        pcd.transform(
            [[1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]])
        return pcd

    def visualize(self, i):
        img = self._get_image(i)
        depth = self._get_depth(i)
        f, axes = plt.subplots(1, 2, sharey=True)
        vis_image(img, ax=axes[0])
        axes[1].imshow(depth[0])
        plt.show()

    def visualize_3d(self, i):
        label = self._get_label(i)
        pose = self._get_pose(i)
        pcds = []
        # camera pcd
        pcd = self.get_camera_pcd(i)
        pcds.append(pcd)

        # model pcd
        for lbl, pse in zip(label, pose):
            obj_pcd = self.get_object_pcd(lbl, pse)
            pcds.append(obj_pcd)
        open3d.draw_geometries(pcds)


class YCBVideoDatasetPoseCNNSegmented(YCBVideoDataset):

    def __init__(self, split='val'):
        if split != 'val':
            raise ValueError(
                'YCB video segmented dataset split {} is not supported'
                .format(split))

        super(YCBVideoDatasetPoseCNNSegmented, self).__init__(split)
        self.add_getter('lbl_img', self._get_lbl_img)
        self.keys = ('img', 'depth', 'lbl_img', 'label', 'pose', 'intrinsic')

    def _get_lbl_img(self, i):
        datapath = osp.join(
            self.tool_dir, './results_PoseCNN_RSS2018/{0:06d}.mat'.format(i))
        data = scipy.io.loadmat(datapath)
        lbl_img = np.array(data['labels'], dtype=np.int32)
        lbl_img = lbl_img - 1
        return lbl_img

    def visualize(self, i):
        img = self._get_image(i)
        depth = self._get_depth(i)
        lbl_img = self._get_lbl_img(i)
        f, axes = plt.subplots(1, 3, sharey=True)
        vis_image(img, ax=axes[0])
        axes[1].imshow(depth[0])
        _, legend_handles = vis_semantic_segmentation(
            img, lbl_img + 1,
            label_names=['background'] + self.label_names, ax=axes[2])
        axes[2].legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
        plt.show()
