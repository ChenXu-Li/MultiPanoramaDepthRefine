"""
log-depth 先验锚点（防止整体塌缩）
实现 L_prior = |log D' - log D^DAP|^2
"""
import torch
from typing import Optional


def compute_prior_anchor_loss(
    log_depth: torch.Tensor,
    log_depth_dap: torch.Tensor,
    weight: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = "l2",
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """
    计算 log-depth 先验锚点损失
    
    公式：L_prior = |log D' - log D^DAP|^2
    
    这是防退化护栏的关键：防止整体塌缩到常数深度
    
    Args:
        log_depth: (H, W) 或 (B, H, W) 优化后的 log-depth
        log_depth_dap: (H, W) 或 (B, H, W) DAP log-depth
        weight: 权重 lambda_prior（必须 > 0）
        mask: (H, W) 可选，有效像素 mask
        loss_type: 'l2' 或 'huber'
        huber_delta: Huber loss 阈值（仅当 loss_type='huber'）
        
    Returns:
        loss: 标量损失值
    """
    if weight <= 0:
        return torch.tensor(0.0, device=log_depth.device, dtype=log_depth.dtype)
    
    # 计算差异
    diff = log_depth - log_depth_dap  # (H, W) 或 (B, H, W)
    
    # 应用 mask（如果提供）
    if mask is not None:
        diff = diff * mask.float()
        valid_count = mask.sum().float()
        if valid_count > 0:
            diff = diff / valid_count
    
    # 计算损失
    if loss_type == "l2":
        loss = (diff ** 2).mean()
    elif loss_type == "huber":
        abs_diff = torch.abs(diff)
        huber_mask = abs_diff <= huber_delta
        loss = torch.where(
            huber_mask,
            0.5 * diff ** 2,
            huber_delta * abs_diff - 0.5 * huber_delta ** 2
        ).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return weight * loss


def compute_prior_anchor_loss_multi_view(
    log_depths: list[torch.Tensor],
    log_depth_daps: list[torch.Tensor],
    weight: float = 1.0,
    masks: Optional[list[torch.Tensor]] = None,
    loss_type: str = "l2",
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """
    计算多视角的 log-depth 先验锚点损失
    
    Args:
        log_depths: List[(H, W)] 各视角的优化后 log-depth
        log_depth_daps: List[(H, W)] 各视角的 DAP log-depth
        weight: 权重
        masks: List[(H, W)] 可选，各视角的有效像素 mask
        loss_type: 'l2' 或 'huber'
        huber_delta: Huber loss 阈值
        
    Returns:
        loss: 标量损失值（所有视角的平均）
    """
    if len(log_depths) != len(log_depth_daps):
        raise ValueError("log_depths 和 log_depth_daps 长度必须相同")
    
    if masks is None:
        masks = [None] * len(log_depths)
    
    total_loss = 0.0
    for log_depth, log_depth_dap, mask in zip(log_depths, log_depth_daps, masks):
        loss = compute_prior_anchor_loss(
            log_depth=log_depth,
            log_depth_dap=log_depth_dap,
            weight=1.0,  # 先不乘权重，最后统一乘
            mask=mask,
            loss_type=loss_type,
            huber_delta=huber_delta,
        )
        total_loss = total_loss + loss
    
    # 平均化
    avg_loss = total_loss / len(log_depths)
    
    # 应用权重
    return weight * avg_loss
