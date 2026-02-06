"""
远景处理（硬规则）
D^{DAP} >= 100m 的像素：
- ❌ 不参与几何对齐（Step 2）
- ✅ 仅参与平滑项（Step 3）
"""
import torch
import numpy as np
from typing import List, Optional


def create_far_field_mask(
    depth_dap: torch.Tensor,
    far_threshold: float = 100.0,
) -> torch.Tensor:
    """
    创建远景 mask
    
    Args:
        depth_dap: (H, W) DAP 深度图（米）
        far_threshold: 远景阈值（米），默认 100.0
        
    Returns:
        far_mask: (H, W) bool，True 表示远景像素
    """
    return depth_dap >= far_threshold


def create_far_field_masks(
    depth_daps: List[torch.Tensor],
    far_threshold: float = 100.0,
) -> List[torch.Tensor]:
    """
    为多个视角创建远景 mask
    
    Args:
        depth_daps: List[(H, W)] 各视角的 DAP 深度图
        far_threshold: 远景阈值（米）
        
    Returns:
        far_masks: List[(H, W)] 各视角的远景 mask
    """
    return [create_far_field_mask(d, far_threshold) for d in depth_daps]


def apply_far_field_mask_to_loss(
    loss_map: torch.Tensor,
    far_mask: torch.Tensor,
    exclude_far: bool = True,
) -> torch.Tensor:
    """
    将远景 mask 应用到损失图上
    
    Args:
        loss_map: (H, W) 损失图
        far_mask: (H, W) bool，远景 mask
        exclude_far: True 表示排除远景，False 表示只保留远景
        
    Returns:
        masked_loss_map: (H, W) 应用 mask 后的损失图
    """
    if exclude_far:
        # 排除远景：远景处损失为 0
        return loss_map * (~far_mask).float()
    else:
        # 只保留远景：非远景处损失为 0
        return loss_map * far_mask.float()
