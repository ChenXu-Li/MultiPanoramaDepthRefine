"""
坐标变换工具：深度 -> 世界点
实现 P_i = T_i ( D'_i * r(theta, phi) )
"""
import numpy as np
import torch
import pycolmap
from typing import Optional
import time

from ..camera.spherical_camera import pixel_to_spherical_coords_torch, spherical_coords_to_directions_torch

def log_time(msg: str):
    """打印带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def depth_to_world_points(
    depth: torch.Tensor,
    cam_from_world: pycolmap.Rigid3d,
    height: int,
    width: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    将深度图转换为世界坐标系中的3D点
    
    实现：P_i = T_i ( D'_i * r(theta, phi) )
    
    Args:
        depth: (H, W) 或 (B, H, W) 深度图（米）
        cam_from_world: pycolmap.Rigid3d，相机到世界的变换（实际上是 world_from_cam）
        height: 图像高度
        width: 图像宽度
        device: 计算设备
        
    Returns:
        points_world: (H, W, 3) 或 (B, H, W, 3) 世界坐标系中的3D点
    """
    t0 = time.time()
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)  # (1, H, W)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, H, W = depth.shape
    
    # 生成 UV 网格
    t1 = time.time()
    u = torch.linspace(0, 1, W, device=device, dtype=torch.float32)
    v = torch.linspace(0, 1, H, device=device, dtype=torch.float32)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')  # (H, W)
    
    # 扩展到 batch 维度
    u_batch = u_grid.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    v_batch = v_grid.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    
    # 转换为球面坐标
    t2 = time.time()
    theta, phi = pixel_to_spherical_coords_torch(u_batch, v_batch)  # (B, H, W)
    
    # 转换为单位方向向量
    t3 = time.time()
    directions = spherical_coords_to_directions_torch(theta, phi)  # (B, H, W, 3)
    
    # 计算相机坐标系中的点：P_cam = depth * direction
    t4 = time.time()
    points_cam = depth.unsqueeze(-1) * directions  # (B, H, W, 3)
    
    # 转换为世界坐标系：P_world = world_from_cam(P_cam)
    # cam_from_world 实际上是 world_from_cam（COLMAP 的命名）
    t5 = time.time()
    R = cam_from_world.rotation.matrix()  # (3, 3)
    t_vec = cam_from_world.translation  # (3,)
    
    # 转换为 torch tensor（直接在目标设备上创建）
    R_torch = torch.tensor(R, dtype=torch.float32, device=device)  # (3, 3)
    t_torch = torch.tensor(t_vec, dtype=torch.float32, device=device)  # (3,)
    
    # 应用变换：P_world = R @ P_cam + t
    # 需要将 (B, H, W, 3) 重塑为 (B*H*W, 3) 进行矩阵乘法
    t6 = time.time()
    B_H_W = B * H * W
    points_cam_flat = points_cam.reshape(B_H_W, 3)  # (B*H*W, 3)
    points_world_flat = points_cam_flat @ R_torch.T + t_torch.unsqueeze(0)  # (B*H*W, 3)
    points_world = points_world_flat.reshape(B, H, W, 3)  # (B, H, W, 3)
    
    if squeeze_output:
        points_world = points_world.squeeze(0)  # (H, W, 3)
    
    return points_world


def get_camera_center_world(cam_from_world: pycolmap.Rigid3d) -> np.ndarray:
    """
    获取相机中心在世界坐标系中的位置
    
    Args:
        cam_from_world: pycolmap.Rigid3d（实际上是 world_from_cam）
        
    Returns:
        camera_center: (3,) 相机中心位置
    """
    # cam_from_world 的平移部分就是相机中心
    return cam_from_world.translation.copy()


def get_camera_ray_directions_world(
    cam_from_world: pycolmap.Rigid3d,
    height: int,
    width: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    获取相机在世界坐标系中的射线方向
    
    Args:
        cam_from_world: pycolmap.Rigid3d（实际上是 world_from_cam）
        height: 图像高度
        width: 图像宽度
        device: 计算设备
        
    Returns:
        ray_directions_world: (H, W, 3) 世界坐标系中的射线方向（单位向量）
    """
    # 生成 UV 网格
    u = torch.linspace(0, 1, width, device=device, dtype=torch.float32)
    v = torch.linspace(0, 1, height, device=device, dtype=torch.float32)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')  # (H, W)
    
    # 转换为球面坐标
    theta, phi = pixel_to_spherical_coords_torch(u_grid, v_grid)  # (H, W)
    
    # 转换为相机坐标系中的方向向量
    directions_cam = spherical_coords_to_directions_torch(theta, phi)  # (H, W, 3)
    
    # 转换为世界坐标系
    R = cam_from_world.rotation.matrix()  # (3, 3)
    R_torch = torch.tensor(R, dtype=torch.float32, device=device)  # (3, 3)
    
    # 应用旋转：direction_world = R @ direction_cam
    H_W = height * width
    directions_cam_flat = directions_cam.reshape(H_W, 3)  # (H*W, 3)
    directions_world_flat = directions_cam_flat @ R_torch.T  # (H*W, 3)
    directions_world = directions_world_flat.reshape(height, width, 3)  # (H, W, 3)
    
    # 归一化（确保是单位向量）
    norms = torch.norm(directions_world, dim=-1, keepdim=True)  # (H, W, 1)
    directions_world = directions_world / (norms + 1e-8)
    
    return directions_world
