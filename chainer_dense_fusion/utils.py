import numpy as np


def generate_organized_pcd(depth, intrinsic):
    H, W = depth.shape
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
    return organized_pcd


def generate_pose(rot, trans):
    pose = quaternion_to_rotation_matrix(rot)
    pose[3, :3] = trans
    return pose


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
