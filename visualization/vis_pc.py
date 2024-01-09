import open3d as o3d
import numpy as np
import torch


def show_pc(raw_pc):
    if raw_pc.shape[0] == 3:
        raw_pc = raw_pc.transpose()
    raw_point = raw_pc  # 读取1.npy数据  N*[x,y,z]

    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="shapenet")
    # 设置点云大小
    vis.get_render_option().point_size = 5
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)
    # 设置点的颜色为白色
    pcd.paint_uniform_color([1, 1, 1])
    # 将点云加入到窗口中
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()
    
def show_pc_kp(raw_pc,key_points):
    if raw_pc.shape[0] == 3:
        raw_pc = raw_pc.transpose()
    
    if key_points.shape[0] == 3:
        key_points = key_points.transpose()
    
    
    raw_point = raw_pc  # 读取1.npy数据  N*[x,y,z]

    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="shapenet")
    # 设置点云大小
    vis.get_render_option().point_size = 5
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)
    # 设置点的颜色为白色
    pcd.paint_uniform_color([1, 1, 1])
    # 将点云加入到窗口中
    vis.add_geometry(pcd)
    
    # create key-points 
    kpcd = o3d.open3d.geometry.PointCloud()
    kpcd.points = o3d.open3d.utility.Vector3dVector(key_points)
    kpcd.paint_uniform_color([1, 0, 0])
    vis.add_geometry(kpcd)


    vis.run()
    vis.destroy_window()

def show_pc_seg(raw_pc,seg_pc):
    if raw_pc.shape[0] == 3:
        raw_pc = raw_pc.transpose()
    
    if seg_pc.shape[0] == 12:
        seg_pc = seg_pc.transpose()
    
    raw_point = raw_pc  # 读取1.npy数据  N*[x,y,z]
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="shapenet")
    vis.get_render_option().point_size = 5
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    colors = [[1,1,1],[1,0,0],[0,1,0],[0,0,1],
              [1,1,0],[0,1,1],[1,0,1],[0.5,0.5,0.5],
              [0.5,1,1],[1,0.5,1],[1,1,0.5],[0.5,0,0.5]]
    
    seg_idx = torch.topk(seg_pc,k=1,dim=-1)[-1].squeeze(-1)
    for i in range(12):
        now_pc = raw_point[seg_idx == i,:]
        pcd = o3d.open3d.geometry.PointCloud()
        pcd.points = o3d.open3d.utility.Vector3dVector(now_pc)
        pcd.paint_uniform_color(colors[i])
        vis.add_geometry(pcd)
    
    #DEBUG_A = raw_pc[seg_idx == 0,:]
    #DEBUG_A = 0
    
    
    # 创建点云对象
    #pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    #pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)
    # 设置点的颜色为白色
    #pcd.paint_uniform_color([1, 1, 1])
    # 将点云加入到窗口中
    #vis.add_geometry(pcd)
    


    vis.run()
    vis.destroy_window()