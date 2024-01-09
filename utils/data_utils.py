import numpy as np
import os
import pickle

import pytorch3d.ops
import torch
import pytorch3d


# def generate_partial_point_ball(points, ids, save_pth=''):
#     # input 2048 x 3
#     # output: occ_points  1024 x 3,  point_occ_mask  1024
#     # indicates the idx of selected points
#     nofp = points.shape[0]
#     assert nofp == 2048
#
#     # circle based occlusion point generation
#     dis_mat_path = os.path.join(save_pth, str(int(ids)) + '.pickle')
#     if os.path.exists(dis_mat_path):
#         with open(dis_mat_path, 'rb') as dis_mat_file:
#             try:
#                 dis_mat = pickle.load(dis_mat_file)
#             finally:
#                 return generate_partial_point_random(points)
#     else:
#         dis_mat_dir = os.path.dirname(dis_mat_path)
#         os.makedirs(dis_mat_dir, exist_ok=True)
#         with open(dis_mat_path, 'wb') as dis_mat_file:
#             points_t = torch.from_numpy(points)
#             points_t = points_t.unsqueeze(0)
#             knn_cal = pytorch3d.ops.knn_points(points_t, points_t, K=nofp // 2, return_sorted=True)
#             knn_cal = knn_cal.idx
#             dis_mat = knn_cal.squeeze().numpy()
#             pickle.dump(dis_mat, dis_mat_file)
#
#     # until here we get the dis_mat
#     center_p_choice = np.array([1, 2, 4, 8])
#     nofcenter = np.random.choice(center_p_choice)  # select a number from center_p_choice
#     selected_points = np.random.choice(nofp, size=nofcenter, replace=False)
#     cancel_num = nofp // 2 // nofcenter  # for each selected points, we need to cancel cancel_num_points
#     center_p_dis = dis_mat[selected_points, :]  # nofcenter x 1024
#     cancel_candidate = center_p_dis[:, :cancel_num]
#     cancel_candidate = cancel_candidate.reshape(-1)
#     point_occ_mask = np.arange(nofp)
#     point_occ_mask[cancel_candidate] = -1
#     selected_occ_mask = point_occ_mask[point_occ_mask >= 0]
#     nofoccp = len(selected_occ_mask)
#     if nofoccp > nofp // 2:
#         new_cal_selection = np.random.choice(nofoccp, size=nofoccp - nofp // 2, replace=False)
#         selected_occ_mask[new_cal_selection] = -1
#         selected_occ_mask = selected_occ_mask[selected_occ_mask >= 0]
#     occ_points = points[selected_occ_mask, :]
#     return occ_points, selected_occ_mask


def generate_partial_point_slice(points, keep_number):
    # input: points N x 3, keep_number
    # output: occ_points  keep_number x 3,  point_occ_mask  keep_number
    # indicates the idx of selected points
    nofp = points.shape[0]

    # select a point as center
    center_ids = np.random.randint(nofp, size=1)
    center_pts = points[center_ids, :]  # 3
    # generate a plane normal
    selected_direction = np.random.uniform(low=-1.0, high=1.0, size=3)  # generate a point
    while np.linalg.norm(selected_direction) < 1e-3:
        selected_direction = np.random.uniform(low=-1.0, high=1.0, size=3)
    selected_direction = selected_direction / np.linalg.norm(selected_direction)
    # calculate the distance from each point to the defined plane
    points_res = points - center_pts.reshape(1, 3)
    dis_mat = np.dot(points_res, selected_direction.reshape(3, 1))  # nx1
    dis_idx = np.argsort(dis_mat.reshape(-1))
    selected_occ_mask = dis_idx[:keep_number]
    occ_points = points[selected_occ_mask, :]
    return occ_points, selected_occ_mask


def sample_points(points, keep_number):
    nofp = points.shape[0]
    selected_points = np.random.choice(nofp, size=keep_number, replace=False)
    point_occ_mask = selected_points
    point_occ = points[selected_points, :]
    return point_occ, point_occ_mask


def quaternion_rotation_matrix(Q):
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

    return rot_matrix


def generate_random_3d_rotation(max_rotation=1.0):
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    x1, x2, x3 = x1 * max_rotation, x2 * max_rotation, x3 * max_rotation
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M

def genertate_random_translation(max_translation=3.0):
    x1, x2, x3 = np.random.rand(3) - 0.5
    x1, x2, x3 = x1 * max_translation, x2 * max_translation, x3 * max_translation
    T = np.array([x1, x2, x3])
    return T

def generate_random_scale(max_scale=2.0, min_scale=0.5):
    return np.random.rand() * (max_scale - min_scale) + min_scale

