import copy

import open3d


def vis_6d_pose_estimation(camera_pcd, object_pcds, pose):
    pcds = []
    pcd = copy.copy(camera_pcd)
    pcd.transform(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]])
    pcds.append(pcd)

    obj_pcds = copy.copy(object_pcds)
    for obj_pcd, pse in zip(obj_pcds, pose):
        obj_pcd.transform(pse.transpose((1, 0)))
        obj_pcd.transform(
            [[1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]])
        pcds.append(obj_pcd)
    open3d.draw_geometries(pcds)
