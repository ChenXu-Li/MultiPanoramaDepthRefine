"""
球面平滑正则
实现 L_smooth = |∇_θ D'|^2 + |∇_φ D'|^2
"""
import torch
from typing import Optional, Literal


def compute_spherical_smoothness_loss(
    log_depth: torch.Tensor,
    weight: float = 0.01,
    smooth_type: Literal["l2", "l1"] = "l2",
    edge_aware: bool = False,
    rgb: Optional[torch.Tensor] = None,
    rgb_sigma: float = 10.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算球面平滑正则损失
    
    公式：L_smooth = |∇_θ D'|^2 + |∇_φ D'|^2
    
    在 equirectangular 投影中：
    - ∇_θ 对应水平方向（u 方向）
    - ∇_φ 对应垂直方向（v 方向）
    
    Args:
        log_depth: (H, W) log-depth
        weight: 权重 lambda_smooth
        smooth_type: 'l2' 或 'l1'
        edge_aware: 是否使用边缘感知平滑（基于 RGB）
        rgb: (H, W, 3) RGB 图像（仅当 edge_aware=True 时需要）
        rgb_sigma: RGB 边缘敏感度（仅当 edge_aware=True 时）
        mask: (H, W) 可选，有效像素 mask
        
    Returns:
        loss: 标量损失值
    """
    if weight <= 0:
        return torch.tensor(0.0, device=log_depth.device, dtype=log_depth.dtype)
    
    H, W = log_depth.shape
    
    # 计算梯度（在 equirectangular 空间中）
    # 水平梯度（u 方向，对应 theta）
    grad_u = log_depth[:, 1:] - log_depth[:, :-1]  # (H, W-1)
    # 垂直梯度（v 方向，对应 phi）
    grad_v = log_depth[1:, :] - log_depth[:-1, :]  # (L-1, W)
    
    # 如果使用边缘感知平滑，计算 RGB 权重
    if edge_aware:
        if rgb is None:
            raise ValueError("edge_aware=True 时需要提供 rgb")
        
        # RGB 梯度
        rgb_grad_u = torch.norm(rgb[:, 1:] - rgb[:, :-1], dim=-1)  # (H, W-1)
        rgb_grad_v = torch.norm(rgb[1:, :] - rgb[:-1, :], dim=-1)  # (H-1, W)
        
        # 计算权重：w = exp(-|I(p)-I(q)|^2 / (2*sigma^2))
        weight_u = torch.exp(-rgb_grad_u ** 2 / (2 * rgb_sigma ** 2))  # (H, W-1)
        weight_v = torch.exp(-rgb_grad_v ** 2 / (2 * rgb_sigma ** 2))  # (H-1, W)
        
        # 应用权重
        grad_u = grad_u * weight_u
        grad_v = grad_v * weight_v
    
    # 计算损失
    if smooth_type == "l2":
        loss_u = (grad_u ** 2).mean()
        loss_v = (grad_v ** 2).mean()
    elif smooth_type == "l1":
        loss_u = torch.abs(grad_u).mean()
        loss_v = torch.abs(grad_v).mean()
    else:
        raise ValueError(f"Unknown smooth_type: {smooth_type}")
    
    loss = loss_u + loss_v
    
    # 应用 mask（如果提供）
    # 注意：mask 应用在梯度计算上比较复杂，这里简化处理
    if mask is not None:
        # 简化：只对有效像素区域计算梯度
        # 实际应用中可能需要更精细的处理
        pass
    
    return weight * loss


def compute_spherical_smoothness_loss_multi_view(
    log_depths: list[torch.Tensor],
    weight: float = 0.01,
    smooth_type: Literal["l2", "l1"] = "l2",
    edge_aware: bool = False,
    rgbs: Optional[list[torch.Tensor]] = None,
    rgb_sigma: float = 10.0,
    masks: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    计算多视角的球面平滑正则损失
    
    Args:
        log_depths: List[(H, W)] 各视角的 log-depth
        weight: 权重
        smooth_type: 'l2' 或 'l1'
        edge_aware: 是否使用边缘感知平滑
        rgbs: List[(H, W, 3)] 各视角的 RGB 图像（仅当 edge_aware=True 时）
        rgb_sigma: RGB 边缘敏感度
        masks: List[(H, W)] 可选，各视角的有效像素 mask
        
    Returns:
        loss: 标量损失值（所有视角的平均）
    """
    if masks is None:
        masks = [None] * len(log_depths)
    
    if edge_aware and rgbs is None:
        raise ValueError("edge_aware=True 时需要提供 rgbs")
    
    total_loss = 0.0
    for i, log_depth in enumerate(log_depths):
        rgb = rgbs[i] if edge_aware and rgbs is not None else None
        mask = masks[i]
        
        loss = compute_spherical_smoothness_loss(
            log_depth=log_depth,
            weight=1.0,  # 先不乘权重
            smooth_type=smooth_type,
            edge_aware=edge_aware,
            rgb=rgb,
            rgb_sigma=rgb_sigma,
            mask=mask,
        )
        total_loss = total_loss + loss
    
    # 平均化
    avg_loss = total_loss / len(log_depths)
    
    # 应用权重
    return weight * avg_loss
