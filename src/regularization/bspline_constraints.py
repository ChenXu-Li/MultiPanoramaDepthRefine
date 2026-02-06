"""
B-spline grid 约束损失函数
包括：单调性约束、方向平滑正则、远景渐近约束
"""
import torch
import torch.nn as nn
from typing import List


def compute_monotonicity_loss(
    control_points: torch.Tensor,
    weight: float = 0.1,
) -> torch.Tensor:
    """
    单调性约束损失
    
    沿 log-depth 轴（axis 1）确保单调性：
    L_mono = sum max(0, -(c_{i,j+1} - c_{i,j}))
    
    保证：∂d'/∂d > 0（深度修正不会破坏深度顺序）
    
    Args:
        control_points: (N_alpha, N_depth) 控制点网格
        weight: 权重
        
    Returns:
        loss: 标量损失值
    """
    if weight <= 0:
        return torch.tensor(0.0, device=control_points.device, dtype=control_points.dtype)
    
    N_alpha, N_depth = control_points.shape
    
    if N_depth < 2:
        return torch.tensor(0.0, device=control_points.device, dtype=control_points.dtype)
    
    # 计算相邻控制点的差值（沿 log-depth 方向）
    diff = control_points[:, 1:] - control_points[:, :-1]  # (N_alpha, N_depth-1)
    
    # 单调性约束：c_{i,j+1} >= c_{i,j}，即 diff >= 0
    # 违反约束的惩罚：max(0, -diff)
    violation = torch.clamp(-diff, min=0.0)  # (N_alpha, N_depth-1)
    
    # 平均损失
    loss = violation.mean()
    
    return weight * loss


def compute_directional_smoothness_loss(
    control_points: torch.Tensor,
    weight: float = 1e-3,
) -> torch.Tensor:
    """
    方向平滑正则损失
    
    只在 alpha 方向（axis 0）：
    L_smooth = sum ||c_{i+1,j} - c_{i,j}||^2
    
    防止相邻方向抖动
    
    Args:
        control_points: (N_alpha, N_depth) 控制点网格
        weight: 权重（推荐 1e-3 ~ 1e-4）
        
    Returns:
        loss: 标量损失值
    """
    if weight <= 0:
        return torch.tensor(0.0, device=control_points.device, dtype=control_points.dtype)
    
    N_alpha, N_depth = control_points.shape
    
    if N_alpha < 2:
        return torch.tensor(0.0, device=control_points.device, dtype=control_points.dtype)
    
    # 计算相邻控制点的差值（沿 alpha 方向）
    diff = control_points[1:, :] - control_points[:-1, :]  # (N_alpha-1, N_depth)
    
    # L2 正则化
    loss = (diff ** 2).mean()
    
    return weight * loss


def compute_far_field_asymptotic_loss(
    control_points: torch.Tensor,
    weight: float = 0.1,
    far_column_idx: int = -1,
) -> torch.Tensor:
    """
    远景渐近约束损失
    
    限制最远 depth knot 的控制点值：
    lim_{d→∞} Δ(alpha, d) → 0
    
    实现方式：对最远一列控制点加 L2 约束，使其接近 0
    
    Args:
        control_points: (N_alpha, N_depth) 控制点网格
        weight: 权重
        far_column_idx: 最远列索引（默认 -1，即最后一列）
        
    Returns:
        loss: 标量损失值
    """
    if weight <= 0:
        return torch.tensor(0.0, device=control_points.device, dtype=control_points.dtype)
    
    N_alpha, N_depth = control_points.shape
    
    if N_depth == 0:
        return torch.tensor(0.0, device=control_points.device, dtype=control_points.dtype)
    
    # 获取最远一列控制点
    if far_column_idx < 0:
        far_column_idx = N_depth + far_column_idx
    
    far_column = control_points[:, far_column_idx]  # (N_alpha,)
    
    # L2 约束：使最远列接近 0
    loss = (far_column ** 2).mean()
    
    return weight * loss


def compute_bspline_constraints_loss(
    control_points: torch.Tensor,
    lambda_mono: float = 0.1,
    lambda_smooth: float = 1e-3,
    lambda_far: float = 0.1,
    far_column_idx: int = -1,
) -> dict[str, torch.Tensor]:
    """
    计算所有 B-spline grid 约束损失
    
    Args:
        control_points: (N_alpha, N_depth) 控制点网格
        lambda_mono: 单调性约束权重
        lambda_smooth: 方向平滑正则权重
        lambda_far: 远景渐近约束权重
        far_column_idx: 最远列索引
        
    Returns:
        loss_dict: 包含各项损失的字典
            - 'total': 总损失
            - 'monotonicity': 单调性约束损失
            - 'smoothness': 方向平滑正则损失
            - 'far_field': 远景渐近约束损失
    """
    mono_loss = compute_monotonicity_loss(control_points, weight=lambda_mono)
    smooth_loss = compute_directional_smoothness_loss(control_points, weight=lambda_smooth)
    far_loss = compute_far_field_asymptotic_loss(
        control_points, weight=lambda_far, far_column_idx=far_column_idx
    )
    
    total_loss = mono_loss + smooth_loss + far_loss
    
    return {
        'total': total_loss,
        'monotonicity': mono_loss,
        'smoothness': smooth_loss,
        'far_field': far_loss,
    }


def compute_bspline_constraints_loss_multi_view(
    control_points_list: List[torch.Tensor],
    lambda_mono: float = 0.1,
    lambda_smooth: float = 1e-3,
    lambda_far: float = 0.1,
    far_column_idx: int = -1,
) -> dict[str, torch.Tensor]:
    """
    计算多视角的 B-spline grid 约束损失
    
    Args:
        control_points_list: List[(N_alpha, N_depth)] 各视角的控制点网格
        lambda_mono: 单调性约束权重
        lambda_smooth: 方向平滑正则权重
        lambda_far: 远景渐近约束权重
        far_column_idx: 最远列索引
        
    Returns:
        loss_dict: 包含各项损失的字典（所有视角的平均）
    """
    if len(control_points_list) == 0:
        device = 'cpu'
        dtype = torch.float32
        return {
            'total': torch.tensor(0.0, device=device, dtype=dtype),
            'monotonicity': torch.tensor(0.0, device=device, dtype=dtype),
            'smoothness': torch.tensor(0.0, device=device, dtype=dtype),
            'far_field': torch.tensor(0.0, device=device, dtype=dtype),
        }
    
    # 计算每个视角的损失
    total_mono = 0.0
    total_smooth = 0.0
    total_far = 0.0
    
    for control_points in control_points_list:
        mono_loss = compute_monotonicity_loss(control_points, weight=1.0)
        smooth_loss = compute_directional_smoothness_loss(control_points, weight=1.0)
        far_loss = compute_far_field_asymptotic_loss(
            control_points, weight=1.0, far_column_idx=far_column_idx
        )
        
        total_mono = total_mono + mono_loss
        total_smooth = total_smooth + smooth_loss
        total_far = total_far + far_loss
    
    # 平均化
    num_views = len(control_points_list)
    avg_mono = total_mono / num_views
    avg_smooth = total_smooth / num_views
    avg_far = total_far / num_views
    
    # 应用权重
    mono_loss_weighted = lambda_mono * avg_mono
    smooth_loss_weighted = lambda_smooth * avg_smooth
    far_loss_weighted = lambda_far * avg_far
    
    total_loss = mono_loss_weighted + smooth_loss_weighted + far_loss_weighted
    
    return {
        'total': total_loss,
        'monotonicity': mono_loss_weighted,
        'smoothness': smooth_loss_weighted,
        'far_field': far_loss_weighted,
    }
