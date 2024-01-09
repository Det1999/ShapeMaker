""" Utility functions for processing point clouds.
Author: Charles R. Qi, Hao Su
Date: November 2016
"""

import os
import sys
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations, Translate, Transform3d
import torch
import open3d as o3d
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Draw point cloud
from .eulerangles import euler2mat

# Point cloud IO
import numpy as np
from .plyfile import PlyData, PlyElement

EPS = 1e-10


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


# a = np.zeros((16,1024,3))
# print point_cloud_to_volume_batch(a, 12, 1.0, False).shape

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=135, yrot=10, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3

    image = image / np.max(image)
    return image


def point_cloud_three_views(points):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img2 = draw_point_cloud(points, zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img3 = draw_point_cloud(points, zrot=180.0 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


import matplotlib.pyplot as plt


def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)


def rotate(points, rot, device=torch.device('cpu'), return_trot=False, t=0.1, uniform=False):
    '''

    :param points:
        - A torch tensor of shape (B,3,N)
    :param rot:
        - String one of [z, so3]
    :return:
        - Rotated points
    '''
    trot = None
    trot_R_ = None
    t_ = torch.zeros((points.shape[0],1,3), device=device)
    if rot == 'z':
        trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Y", degrees=True).to(device)
    elif rot == 'so3':
        trot = Rotate(R=random_rotations(points.shape[0])).to(device)
        trot_R_ = trot.get_matrix()[:,:3,:3].detach()
    elif rot == 'se3':
        trot_R = Rotate(R=random_rotations(points.shape[0])).to(device)
        # if uniform:
        t_ = t * (2 * torch.rand(points.shape[0], 3, device=device) - 1)
        # else:
        #     t_ = t * torch.randn(points.shape[0], 3, device=device)
        trot_T = Translate(t_)
        trot = trot_R.compose(trot_T)
        #trot_back = torch.eye(4).unsqueeze(0).repeat(points.shape[0], 1, 1).to(device)
        #trot_back[:,:3,:3] = trot_R.get_matrix()[:,:3,:3].detach()
        #trot_back[:,:3,3] = t_
        trot_R_ = trot_R.get_matrix()[:,:3,:3].detach()
        t_ = t_.unsqueeze(1).detach()
        
    if trot is not None:
    #    points = trot.transform_points(points.transpose(1, 2)).transpose(1, 2)
         points = (torch.matmul(points.transpose(1,2),trot_R_) + t_).transpose(1,2)

    if return_trot:
        #return points, trot
        return points,trot_R_,t_
    else:
        return points


def save_numpy_to_pcd(xyz, path_):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(path_, pcd)


def load_pcd_to_numpy(path_):
    pcd = o3d.io.read_point_cloud(path_)
    return np.asarray(pcd.points)


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, gts, preds, atten=None, side="both", reduce=True):
        # atten: BMN?
        P = self.batch_pairwise_dist(gts, preds)
        if atten is not None:
            P = P / (atten + 1e-3)
        if reduce:
            mins, _ = torch.min(P, 1)
            loss_1 = torch.mean(mins)
            mins, _ = torch.min(P, 2)
            loss_2 = torch.mean(mins)
            if side == "both":
                return loss_1 + loss_2
            elif side == "left":
                return loss_1
            elif side == "right":
                return loss_2
        else:
            mins, _ = torch.min(P, 1)
            loss_1 = torch.mean(mins, dim=1)
            mins, _ = torch.min(P, 2)
            loss_2 = torch.mean(mins, dim=1)
            if side == "both":
                return loss_1 + loss_2
            elif side == "left":
                return loss_1
            elif side == "right":
                return loss_2

    def batch_pairwise_dist(self, x, y):
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        xx = torch.sum(x ** 2, dim=2, keepdim=True)
        yy = torch.sum(y ** 2, dim=2)[:, None]
        xy = torch.matmul(x, y.transpose(2, 1))
        P = (xx + yy - 2 * xy)
        return P

    def batch_pairwise_dist_deprecated(self, x, y):
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def remove_knn(x, source_id_to_reomve, k=20, device=torch.device('cuda')):
    x = x.clone()
    batch_size = x.size(0)
    num_points = x.size(2)

    knn_idx = knn(x, k=k).view(batch_size * num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points
    knn_ind_to_remove = knn_idx[idx_base + source_id_to_reomve, :].squeeze(1) + idx_base
    x = x.transpose(2, 1)
    all_points_mask = torch.ones(batch_size * num_points, dtype=torch.bool, device=device)
    all_points_mask[knn_ind_to_remove.view(-1)] = 0
    x_ = x.contiguous().view(batch_size * num_points, -1)[all_points_mask, :].view(batch_size, (num_points - k), -1)

    return x_.transpose(1, 2)


def sample(pc, num_samples, device=torch.device('cuda')):
    # pc: Bx3xN
    # Sample the same loc across the batch
    id_to_keep = torch.randint(0, pc.size(2), (num_samples,), device=device)
    pc_ = pc.clone().detach()[:, :, id_to_keep].clone().contiguous()
    return pc_


def to_rotation_mat(self, rot, which_rot='svd'):
    if which_rot == 'svd':
        u, s, v = torch.svd(rot)
        M_TM_pow_minus_half = torch.matmul(v / (s + EPS).unsqueeze(1), v.transpose(2, 1))
        rot_mat = torch.matmul(rot, M_TM_pow_minus_half)
        # If gradient trick is rqeuired:
        # rot_mat = (rot_mat - rot).detach() + rot
    else:
        # Gram–Schmidt
        rot_vec0 = rot[:, 0, :]
        rot_vec1 = rot[:, 1, :] - rot_vec0 * torch.sum(rot_vec0 * rot[:, 1, :], dim=-1, keepdim=True) \
                   / (torch.sum(rot_vec0 ** 2, dim=-1, keepdim=True) + EPS)

        rot_vec2 = rot[:, 2, :] - rot_vec0 * torch.sum(rot_vec0 * rot[:, 2, :], dim=-1, keepdim=True) \
                   / (torch.sum(rot_vec0 ** 2, dim=-1, keepdim=True) + EPS)
        rot_vec2 = rot_vec2 - rot_vec1 * torch.sum(rot_vec1 * rot[:, 2, :], dim=-1, keepdim=True) \
                   / (torch.sum(rot_vec1 ** 2, dim=-1, keepdim=True) + EPS)
        rot_mat = torch.stack([rot_vec0, rot_vec1, rot_vec2], dim=1)
        rot_mat = rot_mat / torch.sqrt((torch.sum(rot_mat ** 2, dim=2, keepdim=True) + EPS))
    return rot_mat


def farthest_point_sample_xyz(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    fps_points = []
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        fps_points.append(centroid)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    fps_points = torch.cat(fps_points, dim=1)
    return centroids, fps_points


def batched_trace(mat):
    return mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def cal_angular_metric(R1, R2):
    M = torch.matmul(R1, R2.transpose(1, 2))
    dist = torch.acos(torch.clamp((batched_trace(M) - 1) / 2., -1 + EPS, 1 - EPS))
    dist = (180 / np.pi) * dist
    return dist


def to_rotation_mat(rot):
    u, s, v = torch.svd(rot)
    M_TM_pow_minus_half = torch.matmul(v / (s + EPS).unsqueeze(1), v.transpose(2, 1))
    rot_mat = torch.matmul(rot, M_TM_pow_minus_half)

    return rot_mat
