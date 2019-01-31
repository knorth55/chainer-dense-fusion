import copy
import numpy as np

from chainercv.visualizations.colormap import voc_colormap

import open3d


flip_rot_matrix = np.array(
    [[1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, -1, 0],
     [0, 0, 0, 1]],
    dtype=np.float32)


def vis_6d_pose_estimation(camera_pcd, object_pcds, pose):
    pcds = []
    pcd = copy.deepcopy(camera_pcd)
    pcd.transform(flip_rot_matrix)
    pcds.append(pcd)

    obj_pcds = copy.deepcopy(object_pcds)
    colors = voc_colormap(list(range(1, len(pose) + 1)))
    for obj_pcd, pse, color in zip(obj_pcds, pose, colors):
        obj_pcd.paint_uniform_color(color)
        obj_pcd.transform(pse.transpose((1, 0)))
        obj_pcd.transform(flip_rot_matrix)
        pcds.append(obj_pcd)
    open3d.draw_geometries(pcds)
