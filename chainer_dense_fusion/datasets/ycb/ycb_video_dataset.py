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

from chainer_dense_fusion.visualizations import vis_6d_pose_estimation


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
        self.add_getter('pose', self._get_pose)
        self.add_getter('pose_label', self._get_pose_label)
        self.add_getter('intrinsic', self._get_intrinsic)
        self.keys = ('img', 'depth', 'pose', 'pose_label', 'intrinsic')

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
        depth = depth / depth_scale
        return depth

    def _get_pose(self, i):
        metapath = osp.join(self.data_dir, '{}-meta.mat'.format(self.ids[i]))
        rt = scipy.io.loadmat(metapath)['poses'].transpose((2, 0, 1))
        pose = np.zeros((len(rt), 4, 4), dtype=np.float32)
        pose[:, 3, 3] = 1
        pose[:, :3, :3] = rt[:, :, :3]
        pose[:, :3, 3] = rt[:, :, 3]
        pose = pose.transpose((0, 2, 1))
        return pose

    def _get_pose_label(self, i):
        metapath = osp.join(self.data_dir, '{}-meta.mat'.format(self.ids[i]))
        object_ids = scipy.io.loadmat(metapath)['cls_indexes'].flatten()
        object_ids = object_ids - 1
        return object_ids

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

    def get_object_pcd(self, lbl):
        xyzpath = osp.join(
            self.data_dir,
            './models/{}/points.xyz'.format(self.label_names[lbl]))
        obj_pcd = open3d.read_point_cloud(xyzpath)
        return obj_pcd

    def get_camera_pcd(self, i):
        img = self._get_image(i)
        depth = self._get_depth(i)
        _, H, W = img.shape
        fx, fy, cx, cy = self._get_intrinsic(i)
        intrinsic = PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        img = open3d.Image(img.transpose((1, 2, 0)).astype(np.uint8))
        depth = open3d.Image(depth)
        rgbd = create_rgbd_image_from_color_and_depth(
            img, depth, depth_scale=1.0, convert_rgb_to_intensity=False)
        pcd = create_point_cloud_from_rgbd_image(rgbd, intrinsic)
        return pcd

    def visualize(self, i):
        img = self._get_image(i)
        depth = self._get_depth(i)
        f, axes = plt.subplots(1, 2, sharey=True)
        vis_image(img, ax=axes[0])
        axes[1].imshow(depth)
        plt.show()

    def visualize_3d(self, i):
        label = self._get_label(i)
        pose = self._get_pose(i)

        # camera pcd
        camera_pcd = self.get_camera_pcd(i)
        # model pcd
        object_pcds = []
        for lbl in label:
            object_pcds.append(self.get_object_pcd(lbl))
        vis_6d_pose_estimation(
            camera_pcd, object_pcds, pose)


class YCBVideoDatasetPoseCNNSegmented(YCBVideoDataset):

    def __init__(self, split='val'):
        if split != 'val':
            raise ValueError(
                'YCB video segmented dataset split {} is not supported'
                .format(split))

        super(YCBVideoDatasetPoseCNNSegmented, self).__init__(split)
        self.add_getter('lbl_img', self._get_lbl_img)
        self.add_getter('bbox', self._get_bbox)
        self.add_getter('bbox_label', self._get_bbox_label)
        self.keys = ('img', 'depth', 'lbl_img', 'bbox', 'bbox_label',
                     'pose', 'pose_label', 'intrinsic')

    def _get_lbl_img(self, i):
        datapath = osp.join(
            self.tool_dir, './results_PoseCNN_RSS2018/{0:06d}.mat'.format(i))
        data = scipy.io.loadmat(datapath)
        lbl_img = np.array(data['labels'], dtype=np.int32)
        lbl_img = lbl_img - 1
        return lbl_img

    def _get_bbox(self, i):
        datapath = osp.join(
            self.tool_dir, './results_PoseCNN_RSS2018/{0:06d}.mat'.format(i))
        data = scipy.io.loadmat(datapath)
        img = self._get_image(i)
        _, H, W = img.shape
        rois = np.array(data['rois'][:, 2:6], dtype=np.float32)
        rois = rois[:, [1, 0, 3, 2]]
        rois[:, :2] = rois[:, :2] + 1
        rois[:, 2:] = rois[:, 2:] - 1
        rois_yc = ((rois[:, 2:3] + rois[:, 0:1]) / 2).astype(np.int32)
        rois_h = rois[:, 2:3] - rois[:, 0:1]
        rois_h = (rois_h // 40 + 1) * 40
        rois_w = rois[:, 3:4] - rois[:, 1:2]
        rois_w = (rois_w // 40 + 1) * 40
        rois_xc = ((rois[:, 3:4] + rois[:, 1:2]) / 2).astype(np.int32)
        bbox = np.concatenate(
            (rois_yc - (rois_h / 2).astype(np.int32),
             rois_xc - (rois_w / 2).astype(np.int32),
             rois_yc + (rois_h / 2).astype(np.int32),
             rois_xc + (rois_w / 2).astype(np.int32)), axis=1)
        for bb in bbox:
            if bb[0] < 0:
                bb[2] = bb[2] - bb[0]
                bb[0] = 0
            if bb[1] < 0:
                bb[3] = bb[3] - bb[1]
                bb[1] = 0
            if bb[2] > H:
                bb[0] = bb[0] - bb[2] + H
                bb[2] = H
            if bb[2] > W:
                bb[0] = bb[0] - bb[2] + W
                bb[2] = W
        return bbox

    def _get_bbox_label(self, i):
        datapath = osp.join(
            self.tool_dir, './results_PoseCNN_RSS2018/{0:06d}.mat'.format(i))
        data = scipy.io.loadmat(datapath)
        bbox_label = np.array(data['rois'][:, 1], dtype=np.int32) - 1
        return bbox_label

    def visualize(self, i):
        img = self._get_image(i)
        depth = self._get_depth(i)
        lbl_img = self._get_lbl_img(i)
        f, axes = plt.subplots(1, 3, sharey=True)
        vis_image(img, ax=axes[0])
        axes[1].imshow(depth)
        _, legend_handles = vis_semantic_segmentation(
            img, lbl_img + 1,
            label_names=['background'] + self.label_names, ax=axes[2])
        axes[2].legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
        plt.show()
