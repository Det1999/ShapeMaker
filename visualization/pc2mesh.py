
import torch
import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Pointclouds,Meshes
from pytorch3d.io import save_obj
import os

def pc2mesh(pc, device, color=None):
    import open3d as o3d
    pc = pc.cpu().numpy()
    obj_num = pc.shape[0]

    verts_list = []
    faces_list = []
    texture_list = []

    for obj_idx in range(obj_num):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[obj_idx])
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                 std_ratio=1.0)
        pcd = pcd.select_by_index(ind)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                 std_ratio=1.0)
        pcd = pcd.select_by_index(ind)
        pcd.normals = o3d.utility.Vector3dVector(np.zeros(
            (1, 3)))  # invalidate existing normals
        pcd.estimate_normals()

        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        alpha = 0.2
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        verts = np.asarray(mesh.vertices)
        verts = torch.from_numpy(verts).float().to(device)
        verts_list.append(verts)
        faces = np.asarray(mesh.triangles)
        faces_list.append(torch.from_numpy(faces).float().to(device))
        # mesh.compute_vertex_normals()
        if color is not None:
            colors = torch.ones_like(verts)
            colors[:, ...] = color
            texture = colors
            texture_list.append(texture)

    if color is not None:
        texture_list = TexturesVertex(texture_list)
        mesh_output = Meshes(verts_list, faces_list, texture_list)
    else:
        mesh_output = Meshes(verts_list, faces_list)

    return mesh_output

def pc_filter(pc, device, color=None):
    import open3d as o3d
    pc = pc.cpu().numpy().astype(np.float64)
    obj_num = pc.shape[0]

    verts_list = []
    if color is not None:
        textures_list = []
    else:
        textures_list = None

    for obj_idx in range(obj_num):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[obj_idx])
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                 std_ratio=1.0)
        pcd = pcd.select_by_index(ind)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                 std_ratio=1.0)
        pcd = pcd.select_by_index(ind)
        pc_curr = np.asarray(pcd.points)
        pc_curr = torch.from_numpy(pc_curr).float().to(device)
        verts_list.append(pc_curr)
        if color is not None:
            colors = torch.ones_like(pc_curr)
            colors[:, ...] = color
            textures_list.append(colors)

    pc_all = Pointclouds(verts_list, features=textures_list)

    return pc_all


if __name__ =="__main__":
    pass