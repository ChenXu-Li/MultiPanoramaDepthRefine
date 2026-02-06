"""
Ray-space 深度一致性（弱约束）
实现 L_depth = |log D'_j - log D_{i→j}|
"""
import torch
import numpy as np
from typing import Optional

from .coordinate_transform import (
    depth_to_world_points,
    get_camera_ray_directions_world,
    get_camera_center_world,
)


def project_point_to_camera(
    point_world: torch.Tensor,
    cam_from_world: "pycolmap.Rigid3d",
    height: int,
    width: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将世界点投影到相机坐标系，然后投影到像素坐标
    
    Args:
        point_world: (3,) 或 (N, 3) 世界点
        cam_from_world: pycolmap.Rigid3d（实际上是 world_from_cam）
        height: 图像高度
        width: 图像宽度
        device: 计算设备
        
    Returns:
        u: (N,) 归一化 U 坐标 [0, 1]
        v: (N,) 归一化 V 坐标 [0, 1]
        valid: (N,) 布尔数组，表示点是否在视野内
    """
    if point_world.dim() == 1:
        point_world = point_world.unsqueeze(0)  # (1, 3)
    
    N = point_world.shape[0]
    
    # 转换为相机坐标系
    # world_from_cam 的逆是 cam_from_world
    R_world_cam = cam_from_world.rotation.matrix()  # (3, 3)
    t_world_cam = cam_from_world.translation  # (3,)
    
    # cam_from_world = (world_from_cam)^(-1)
    R_cam_world = R_world_cam.T  # (3, 3)
    t_cam_world = -R_cam_world @ t_world_cam  # (3,)
    
    R_torch = torch.tensor(R_cam_world, dtype=torch.float32, device=device)
    t_torch = torch.tensor(t_cam_world, dtype=torch.float32, device=device)
    
    # P_cam = R_cam_world @ P_world + t_cam_world
    points_cam = point_world @ R_torch.T + t_torch.unsqueeze(0)  # (N, 3)
    
    # 计算深度和方向
    depths = torch.norm(points_cam, dim=-1)  # (N,)
    directions_cam = points_cam / (depths.unsqueeze(-1) + 1e-8)  # (N, 3)
    
    # 转换为球面坐标
    x, y, z = directions_cam[:, 0], directions_cam[:, 1], directions_cam[:, 2]
    
    # DAP 约定：theta/phi
    phi = torch.acos(torch.clamp(z, -1.0, 1.0))  # [0, pi]
    theta = torch.atan2(y, x)  # [-pi, pi]
    theta = torch.fmod(theta + 2.0 * np.pi, 2.0 * np.pi)  # [0, 2*pi)
    
    # 转换为 UV
    u = 1.0 - theta / (2.0 * np.pi)  # [0, 1]
    v = phi / np.pi  # [0, 1]
    
    # 检查是否在视野内
    valid = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & (depths > 0)
    
    return u, v, valid


def compute_ray_space_depth_consistency_loss(
    depth_i: torch.Tensor,
    depth_j: torch.Tensor,
    cam_from_world_i: "pycolmap.Rigid3d",
    cam_from_world_j: "pycolmap.Rigid3d",
    height: int,
    width: int,
    mask: Optional[torch.Tensor] = None,
    use_robust_loss: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """
    计算 Ray-space 深度一致性损失
    
    对于视角 i 的每个像素：
    1. 计算对应的世界点 P_i
    2. 将 P_i 投影到视角 j，得到像素坐标 (u_j, v_j)
    3. 计算深度一致性：|log D'_j(u_j, v_j) - log D_{i→j}|
    
    Args:
        depth_i: (H, W) 视角 i 的深度图
        depth_j: (H, W) 视角 j 的深度图
        cam_from_world_i: 视角 i 的相机变换
        cam_from_world_j: 视角 j 的相机变换
        height: 图像高度
        width: 图像宽度
        mask: (H, W) 可选，有效像素 mask
        use_robust_loss: 是否使用 robust loss
        device: 计算设备
        
    Returns:
        loss: 标量损失值
    """
    H, W = depth_i.shape
    
    # 将视角 i 的深度转换为世界点
    points_world_i = depth_to_world_points(
        depth_i, cam_from_world_i, height, width, device=device
    )  # (H, W, 3)
    
    # 计算视角 j 的射线方向
    ray_directions_j = get_camera_ray_directions_world(
        cam_from_world_j, height, width, device=device
    )  # (H, W, 3)
    
    # 获取视角 j 的相机中心
    camera_center_j = torch.tensor(
        get_camera_center_world(cam_from_world_j),
        dtype=torch.float32,
        device=device
    )  # (3,)
    
    # 向量化实现：避免双重循环
    log_depth_j = torch.log(depth_j + 1e-8)  # (H, W)
    
    # 将 points_world_i 重塑为 (H*W, 3)
    H_W = H * W
    points_world_flat = points_world_i.reshape(H_W, 3)  # (H*W, 3)
    
    # 转换为相机坐标系（向量化）
    R_world_cam = cam_from_world_j.rotation.matrix()  # (3, 3)
    t_world_cam = cam_from_world_j.translation  # (3,)
    R_cam_world = R_world_cam.T  # (3, 3)
    t_cam_world = -R_cam_world @ t_world_cam  # (3,)
    
    R_torch = torch.tensor(R_cam_world, dtype=torch.float32, device=device)
    t_torch = torch.tensor(t_cam_world, dtype=torch.float32, device=device)
    
    # P_cam = R_cam_world @ P_world + t_cam_world (向量化)
    points_cam_flat = points_world_flat @ R_torch.T + t_torch.unsqueeze(0)  # (H*W, 3)
    
    # 计算深度和方向（向量化）
    depths_cam = torch.norm(points_cam_flat, dim=-1)  # (H*W,)
    directions_cam_flat = points_cam_flat / (depths_cam.unsqueeze(-1) + 1e-8)  # (H*W, 3)
    
    # 转换为球面坐标（向量化）
    x, y, z = directions_cam_flat[:, 0], directions_cam_flat[:, 1], directions_cam_flat[:, 2]
    phi = torch.acos(torch.clamp(z, -1.0, 1.0))  # (H*W,)
    theta = torch.atan2(y, x)  # (H*W,)
    theta = torch.fmod(theta + 2.0 * np.pi, 2.0 * np.pi)  # (H*W,)
    
    # 转换为 UV（向量化）
    u_flat = 1.0 - theta / (2.0 * np.pi)  # (H*W,)
    v_flat = phi / np.pi  # (H*W,)
    
    # 检查有效性（向量化）
    valid_flat = (u_flat >= 0) & (u_flat <= 1) & (v_flat >= 0) & (v_flat <= 1) & (depths_cam > 0)  # (H*W,)
    
    # 应用 mask（如果有）
    if mask is not None:
        mask_flat = mask.reshape(H_W)  # (H*W,)
        valid_flat = valid_flat & mask_flat
    
    # 转换为像素坐标
    u_pix_flat = u_flat * width  # (H*W,)
    v_pix_flat = v_flat * height  # (H*W,)
    
    # 边界检查
    u_pix_flat = torch.clamp(u_pix_flat, 0, width - 1)
    v_pix_flat = torch.clamp(v_pix_flat, 0, height - 1)
    
    # 最近邻采样索引
    u_idx_flat = u_pix_flat.long()  # (H*W,)
    v_idx_flat = v_pix_flat.long()  # (H*W,)
    
    # 计算投影深度：从视角 j 的射线方向计算（向量化）
    vec_to_point_flat = points_world_flat - camera_center_j.unsqueeze(0)  # (H*W, 3)
    
    # 获取对应的射线方向（需要从 ray_directions_j 中采样）
    # ray_directions_j 是 (H, W, 3)，需要根据 v_idx, u_idx 采样
    # 使用 advanced indexing
    ray_dir_flat = ray_directions_j[v_idx_flat, u_idx_flat]  # (H*W, 3)
    
    # 投影长度就是深度（向量化）
    depth_proj_flat = torch.sum(vec_to_point_flat * ray_dir_flat, dim=-1)  # (H*W,)
    
    # 计算 log_depth_i_to_j（只保留有效的）
    log_depth_i_to_j_flat = torch.zeros_like(depths_cam)
    valid_mask_flat = valid_flat & (depth_proj_flat > 0)
    log_depth_i_to_j_flat[valid_mask_flat] = torch.log(depth_proj_flat[valid_mask_flat] + 1e-8)
    
    # 重塑回 (H, W)
    log_depth_i_to_j = log_depth_i_to_j_flat.reshape(H, W)
    valid_mask = valid_mask_flat.reshape(H, W)  # 投影有效性 mask (H, W)
    
    # 计算一致性损失（仅在有效像素上统计）
    diff = log_depth_j - log_depth_i_to_j  # (H, W)
    
    if mask is not None:
        # 仅对 mask 内的像素计算损失，避免远景/无效区域稀释损失
        # valid_mask 已经包含了投影有效性，现在再与传入的 mask 结合
        valid = mask & valid_mask  # 与投影有效性共同约束
        if valid.sum() == 0:
            # 没有有效像素时返回 0，避免 NaN
            return torch.tensor(0.0, device=device, requires_grad=True)
        if use_robust_loss:
            # L1 loss
            loss = torch.abs(diff[valid]).mean()
        else:
            # L2 loss
            loss = (diff[valid] ** 2).mean()
    else:
        # 无 mask 时，仅对投影有效的像素计算损失
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if use_robust_loss:
            loss = torch.abs(diff[valid_mask]).mean()
        else:
            loss = (diff[valid_mask] ** 2).mean()
    
    return loss
