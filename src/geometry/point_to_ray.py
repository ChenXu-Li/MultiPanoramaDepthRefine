"""
Point-to-Ray 距离计算（主约束）
实现 L_p2r = |(P_i - C_j) - ((P_i - C_j) · r_j) r_j|
"""
import torch
import numpy as np
from typing import Optional


def point_to_ray_distance(
    point: torch.Tensor,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
) -> torch.Tensor:
    """
    计算点到射线的距离
    
    公式：d = |(P - C) - ((P - C) · r) r|
    
    其中：
    - P: 点位置
    - C: 射线原点（相机中心）
    - r: 射线方向（单位向量）
    
    Args:
        point: (..., 3) 点位置
        ray_origin: (..., 3) 射线原点
        ray_direction: (..., 3) 射线方向（单位向量）
        
    Returns:
        distance: (...,) 点到射线的距离
    """
    # 计算向量 P - C
    vec = point - ray_origin  # (..., 3)
    
    # 计算投影长度：(P - C) · r
    proj_length = torch.sum(vec * ray_direction, dim=-1, keepdim=True)  # (..., 1)
    
    # 计算投影向量：((P - C) · r) r
    proj_vec = proj_length * ray_direction  # (..., 3)
    
    # 计算垂直向量：(P - C) - ((P - C) · r) r
    perp_vec = vec - proj_vec  # (..., 3)
    
    # 计算距离：|垂直向量|
    distance = torch.norm(perp_vec, dim=-1)  # (...)
    
    return distance


def compute_point_to_ray_loss(
    points_i: torch.Tensor,
    camera_center_j: torch.Tensor,
    ray_directions_j: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_robust_loss: bool = True,
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """
    计算 Point-to-Ray 损失
    
    Args:
        points_i: (H_i, W_i, 3) 视角 i 的世界点
        camera_center_j: (3,) 视角 j 的相机中心
        ray_directions_j: (H_j, W_j, 3) 视角 j 的射线方向
        mask: (H_i, W_i) 可选，有效像素 mask
        use_robust_loss: 是否使用 Huber loss
        huber_delta: Huber loss 阈值（米）
        
    Returns:
        loss: 标量损失值
    """
    H_i, W_i, _ = points_i.shape
    H_j, W_j, _ = ray_directions_j.shape
    
    # 扩展维度以便广播
    camera_center_j_expanded = camera_center_j.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
    
    # 将 points_i 重塑为 (H_i*W_i, 3)
    points_i_flat = points_i.reshape(H_i * W_i, 3)  # (H_i*W_i, 3)
    
    # 对于每个点，找到视角 j 中最近的射线
    # 简化实现：使用最近邻匹配（可以优化为更精确的方法）
    # 这里假设 points_i 和 ray_directions_j 已经对齐（通过投影）
    
    # 计算每个点到每条射线的距离
    # 简化：假设对应关系（实际应该通过投影建立对应关系）
    # 这里先实现基本版本，后续可以优化
    
    # 将 ray_directions_j 重塑
    ray_directions_j_flat = ray_directions_j.reshape(H_j * W_j, 3)  # (H_j*W_j, 3)
    
    # 计算所有点到所有射线的距离（简化版本）
    # 实际应用中应该通过投影建立对应关系
    # 这里先实现一个简化版本：假设对应关系
    
    # 如果 H_i*W_i == H_j*W_j，直接对应
    if H_i * W_i == H_j * W_j:
        distances = point_to_ray_distance(
            points_i_flat,
            camera_center_j_expanded.expand(H_i * W_i, -1),
            ray_directions_j_flat,
        )  # (H_i*W_i,)
    else:
        # 需要建立对应关系（通过投影）
        # 简化：使用最近邻
        # TODO: 实现更精确的投影对应关系
        raise NotImplementedError("需要实现投影对应关系建立")
    
    # 应用 mask（如果提供）
    if mask is not None:
        mask_flat = mask.reshape(-1)  # (H_i*W_i,)
        distances = distances * mask_flat.float()
        valid_count = mask_flat.sum().float()
        if valid_count > 0:
            distances = distances / valid_count
    
    # 应用 robust loss（如果启用）
    if use_robust_loss:
        # Huber loss
        abs_distances = torch.abs(distances)
        huber_mask = abs_distances <= huber_delta
        loss = torch.where(
            huber_mask,
            0.5 * distances ** 2,
            huber_delta * abs_distances - 0.5 * huber_delta ** 2
        )
        loss = loss.mean()
    else:
        # L2 loss
        loss = (distances ** 2).mean()
    
    return loss
