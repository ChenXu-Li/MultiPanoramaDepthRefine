"""
方向变形约束（防止 scale 爆炸）
实现 L_g = |g|^2
其中 g 是方向相关缩放的参数（球谐系数或 B-spline grid）
"""
import torch
from typing import List


def compute_scale_constraint_loss(
    scale_modules: List[torch.nn.Module],
    weight: float = 0.01,
) -> torch.Tensor:
    """
    计算方向变形约束损失
    
    公式：L_g = |g|^2
    
    其中 g 是方向相关缩放的参数：
    - 对于球谐函数：g 是球谐系数
    - 对于 B-spline grid：g 是 grid 控制点值
    
    Args:
        scale_modules: List[ScaleModule] 各视角的方向缩放模块
        weight: 权重 lambda_g
        
    Returns:
        loss: 标量损失值
    """
    if weight <= 0:
        # 确定设备
        if scale_modules and len(scale_modules) > 0:
            if hasattr(scale_modules[0], 'coeffs'):
                device = scale_modules[0].coeffs.device
            elif hasattr(scale_modules[0], 'grid'):
                device = scale_modules[0].grid.device
            else:
                device = 'cpu'
        else:
            device = 'cpu'
        return torch.tensor(0.0, device=device)
    
    if len(scale_modules) == 0:
        return torch.tensor(0.0, device='cpu')
    
    # 确定设备和数据类型
    first_module = scale_modules[0]
    if hasattr(first_module, 'coeffs'):
        device = first_module.coeffs.device
        dtype = first_module.coeffs.dtype
    elif hasattr(first_module, 'grid'):
        device = first_module.grid.device
        dtype = first_module.grid.dtype
    else:
        device = 'cpu'
        dtype = torch.float32
    
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    valid_count = 0
    
    for scale_module in scale_modules:
        if scale_module is None:
            continue
        
        # 获取缩放参数
        if hasattr(scale_module, 'coeffs'):
            # 球谐函数：使用系数
            params = scale_module.coeffs  # (num_coeffs,)
        elif hasattr(scale_module, 'grid'):
            # B-spline grid：使用 grid 值
            params = scale_module.grid  # (theta_res, phi_res)
        else:
            # 未知类型，跳过
            continue
        
        # L2 正则化：|g|^2
        loss = (params ** 2).mean()
        total_loss = total_loss + loss
        valid_count += 1
    
    # 平均化
    if valid_count > 0:
        avg_loss = total_loss / valid_count
    else:
        avg_loss = torch.tensor(0.0, device=device, dtype=dtype)
    
    # 应用权重
    return weight * avg_loss
