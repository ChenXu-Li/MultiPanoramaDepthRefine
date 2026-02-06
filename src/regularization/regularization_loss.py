"""
防退化护栏：组合所有正则化项
"""
import torch
from typing import List, Optional

from .prior_anchor import compute_prior_anchor_loss_multi_view
from .smoothness import compute_spherical_smoothness_loss_multi_view
from .scale_guard import compute_scale_constraint_loss
from .far_field_mask import create_far_field_masks


class RegularizationLoss:
    """
    防退化护栏损失
    
    组合：
    1. log-depth 先验锚点（防止整体塌缩）
    2. 球面平滑正则（保持连续性）
    3. 方向变形约束（防止 scale 爆炸）
    """
    
    def __init__(
        self,
        lambda_prior: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_scale: float = 0.01,
        far_threshold: float = 100.0,
        prior_loss_type: str = "l2",
        prior_huber_delta: float = 0.1,
        smooth_type: str = "l2",
        edge_aware: bool = False,
        rgb_sigma: float = 10.0,
    ):
        """
        Args:
            lambda_prior: 先验锚点权重（必须 > 0，防止塌缩）
            lambda_smooth: 平滑正则权重
            lambda_scale: 方向变形约束权重
            far_threshold: 远景阈值（米）
            prior_loss_type: 先验损失类型（'l2' 或 'huber'）
            prior_huber_delta: 先验 Huber loss 阈值
            smooth_type: 平滑类型（'l2' 或 'l1'）
            edge_aware: 是否使用边缘感知平滑
            rgb_sigma: RGB 边缘敏感度
        """
        self.lambda_prior = lambda_prior
        self.lambda_smooth = lambda_smooth
        self.lambda_scale = lambda_scale
        self.far_threshold = far_threshold
        self.prior_loss_type = prior_loss_type
        self.prior_huber_delta = prior_huber_delta
        self.smooth_type = smooth_type
        self.edge_aware = edge_aware
        self.rgb_sigma = rgb_sigma
        
        # 验证权重
        if lambda_prior <= 0:
            raise ValueError(f"lambda_prior ({lambda_prior}) 必须 > 0（防止塌缩）")
    
    def compute_loss(
        self,
        log_depths: List[torch.Tensor],
        log_depth_daps: List[torch.Tensor],
        depth_daps: List[torch.Tensor],
        scale_modules: Optional[List[torch.nn.Module]] = None,
        rgbs: Optional[List[torch.Tensor]] = None,
        masks: Optional[List[torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        计算防退化护栏损失
        
        Args:
            log_depths: List[(H, W)] 各视角的优化后 log-depth
            log_depth_daps: List[(H, W)] 各视角的 DAP log-depth
            depth_daps: List[(H, W)] 各视角的 DAP 深度（用于远景 mask）
            scale_modules: List[ScaleModule] 可选，各视角的方向缩放模块
            rgbs: List[(H, W, 3)] 可选，各视角的 RGB 图像（用于边缘感知平滑）
            masks: List[(H, W)] 可选，各视角的有效像素 mask
            
        Returns:
            loss_dict: 包含各项损失的字典
                - 'total': 总损失
                - 'prior': 先验锚点损失
                - 'smooth': 平滑正则损失
                - 'scale': 方向变形约束损失
        """
        if masks is None:
            masks = [None] * len(log_depths)
        
        # 创建远景 mask
        far_masks = create_far_field_masks(depth_daps, self.far_threshold)
        
        # 1. log-depth 先验锚点损失
        prior_loss = 0.0
        if self.lambda_prior > 0:
            prior_loss = compute_prior_anchor_loss_multi_view(
                log_depths=log_depths,
                log_depth_daps=log_depth_daps,
                weight=self.lambda_prior,
                masks=masks,
                loss_type=self.prior_loss_type,
                huber_delta=self.prior_huber_delta,
            )
        
        # 2. 球面平滑正则损失
        smooth_loss = 0.0
        if self.lambda_smooth > 0:
            # 对于远景，只参与平滑项
            # 这里简化处理：对所有像素计算平滑，但远景不参与几何对齐（在 Step 2 中处理）
            smooth_loss = compute_spherical_smoothness_loss_multi_view(
                log_depths=log_depths,
                weight=self.lambda_smooth,
                smooth_type=self.smooth_type,
                edge_aware=self.edge_aware,
                rgbs=rgbs,
                rgb_sigma=self.rgb_sigma,
                masks=masks,
            )
        
        # 3. 方向变形约束损失
        scale_loss = 0.0
        if self.lambda_scale > 0 and scale_modules is not None:
            scale_loss = compute_scale_constraint_loss(
                scale_modules=scale_modules,
                weight=self.lambda_scale,
            )
        
        # 总损失
        total_loss = prior_loss + smooth_loss + scale_loss
        
        return {
            'total': total_loss,
            'prior': prior_loss,
            'smooth': smooth_loss,
            'scale': scale_loss,
        }
