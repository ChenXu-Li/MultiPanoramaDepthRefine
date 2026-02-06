"""
Step 3 单元测试
测试防退化护栏
"""
import numpy as np
import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.regularization import (
    compute_prior_anchor_loss,
    compute_spherical_smoothness_loss,
    create_far_field_mask,
    RegularizationLoss,
)
from src.deformation import SphericalHarmonicsScale


class TestPriorAnchor:
    """测试 log-depth 先验锚点"""
    
    def test_prior_anchor_zero_deformation(self):
        """测试零变形情况：log_depth == log_depth_dap → loss = 0"""
        height, width = 100, 200
        
        log_depth = torch.ones(height, width) * 2.0  # log(10)
        log_depth_dap = torch.ones(height, width) * 2.0
        
        loss = compute_prior_anchor_loss(
            log_depth=log_depth,
            log_depth_dap=log_depth_dap,
            weight=1.0,
        )
        
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), f"零变形损失应为 0，实际: {loss.item()}"
    
    def test_prior_anchor_collapse_prevention(self):
        """测试塌缩预防：当深度偏离 DAP 时，损失增大"""
        height, width = 100, 200
        
        log_depth_dap = torch.ones(height, width) * 2.0  # log(10)
        
        # 情况 1：深度接近 DAP
        log_depth_close = torch.ones(height, width) * 2.1
        loss_close = compute_prior_anchor_loss(
            log_depth=log_depth_close,
            log_depth_dap=log_depth_dap,
            weight=1.0,
        )
        
        # 情况 2：深度远离 DAP（塌缩）
        log_depth_collapsed = torch.ones(height, width) * 0.0  # log(1)，严重塌缩
        loss_collapsed = compute_prior_anchor_loss(
            log_depth=log_depth_collapsed,
            log_depth_dap=log_depth_dap,
            weight=1.0,
        )
        
        # 塌缩情况的损失应该显著大于接近情况
        assert loss_collapsed.item() > loss_close.item() * 10, "塌缩预防失败"
    
    def test_prior_anchor_weight_zero(self):
        """测试权重为 0 时损失为 0"""
        height, width = 100, 200
        
        log_depth = torch.ones(height, width) * 0.0
        log_depth_dap = torch.ones(height, width) * 2.0
        
        loss = compute_prior_anchor_loss(
            log_depth=log_depth,
            log_depth_dap=log_depth_dap,
            weight=0.0,
        )
        
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), "权重为 0 时损失应为 0"


class TestSmoothness:
    """测试球面平滑正则"""
    
    def test_smoothness_constant_depth(self):
        """测试常数深度：梯度为 0 → loss = 0"""
        height, width = 100, 200
        
        log_depth = torch.ones(height, width) * 2.0
        
        loss = compute_spherical_smoothness_loss(
            log_depth=log_depth,
            weight=1.0,
            smooth_type="l2",
        )
        
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), f"常数深度损失应为 0，实际: {loss.item()}"
    
    def test_smoothness_noisy_depth(self):
        """测试噪声深度：梯度大 → loss 大"""
        height, width = 100, 200
        
        # 常数深度
        log_depth_smooth = torch.ones(height, width) * 2.0
        
        # 噪声深度
        log_depth_noisy = torch.ones(height, width) * 2.0
        log_depth_noisy += torch.randn(height, width) * 0.1
        
        loss_smooth = compute_spherical_smoothness_loss(
            log_depth=log_depth_smooth,
            weight=1.0,
            smooth_type="l2",
        )
        
        loss_noisy = compute_spherical_smoothness_loss(
            log_depth=log_depth_noisy,
            weight=1.0,
            smooth_type="l2",
        )
        
        # 噪声深度损失应该更大
        assert loss_noisy.item() > loss_smooth.item(), "平滑正则未正确惩罚噪声"


class TestFarFieldMask:
    """测试远景 mask"""
    
    def test_far_field_mask(self):
        """测试远景 mask 创建"""
        height, width = 100, 200
        
        # 创建包含远景的深度图
        depth_dap = torch.ones(height, width) * 50.0  # 50 米
        depth_dap[50:, :] = 150.0  # 下半部分是远景
        
        far_mask = create_far_field_mask(depth_dap, far_threshold=100.0)
        
        # 检查远景 mask
        assert far_mask[0, 0].item() == False, "近景不应被标记为远景"
        assert far_mask[75, 100].item() == True, "远景应被正确标记"
        
        # 统计
        far_count = far_mask.sum().item()
        expected_far_count = height // 2 * width
        assert abs(far_count - expected_far_count) < width, f"远景数量不正确: {far_count} vs {expected_far_count}"


class TestRegularizationLoss:
    """测试防退化护栏损失组合"""
    
    def test_collapse_prevention(self):
        """测试塌缩预防：关闭 prior → 观察塌缩，打开 prior → 塌缩消失"""
        height, width = 100, 200
        
        log_depth_dap = torch.ones(height, width) * 2.0
        depth_dap = torch.exp(log_depth_dap)
        
        # 塌缩深度（所有深度为常数）
        log_depth_collapsed = torch.ones(height, width) * 0.0
        
        # 情况 1：关闭 prior（lambda_prior = 0）
        loss_fn_no_prior = RegularizationLoss(
            lambda_prior=0.0,
            lambda_smooth=0.01,
            lambda_scale=0.01,
        )
        
        loss_dict_no_prior = loss_fn_no_prior.compute_loss(
            log_depths=[log_depth_collapsed],
            log_depth_daps=[log_depth_dap],
            depth_daps=[depth_dap],
        )
        
        # 情况 2：打开 prior（lambda_prior > 0）
        loss_fn_with_prior = RegularizationLoss(
            lambda_prior=1.0,
            lambda_smooth=0.01,
            lambda_scale=0.01,
        )
        
        loss_dict_with_prior = loss_fn_with_prior.compute_loss(
            log_depths=[log_depth_collapsed],
            log_depth_daps=[log_depth_dap],
            depth_daps=[depth_dap],
        )
        
        # 有 prior 时的损失应该显著大于无 prior 时
        assert loss_dict_with_prior['total'].item() > loss_dict_no_prior['total'].item() * 10, \
            "Prior 未正确防止塌缩"
    
    def test_far_field_stability(self):
        """测试远景稳定性：远景深度不随迭代剧烈变化"""
        height, width = 100, 200
        
        # 创建包含远景的深度图
        log_depth_dap = torch.ones(height, width) * 2.0
        depth_dap = torch.exp(log_depth_dap)
        depth_dap[50:, :] = 150.0  # 下半部分是远景
        
        # 远景深度应该保持稳定（接近 DAP）
        log_depth_stable = torch.log(depth_dap + 1e-8)
        
        loss_fn = RegularizationLoss(
            lambda_prior=1.0,
            lambda_smooth=0.01,
            lambda_scale=0.01,
        )
        
        loss_dict = loss_fn.compute_loss(
            log_depths=[log_depth_stable],
            log_depth_daps=[log_depth_dap],
            depth_daps=[depth_dap],
        )
        
        # 损失应该有限（不是 inf 或 nan）
        assert torch.isfinite(loss_dict['total']), "损失不是有限值"
    
    def test_scale_constraint(self):
        """测试方向变形约束：防止 scale 爆炸"""
        height, width = 100, 200
        
        log_depth = torch.ones(height, width) * 2.0
        log_depth_dap = torch.ones(height, width) * 2.0
        depth_dap = torch.exp(log_depth_dap)
        
        # 创建方向缩放模块
        scale_module = SphericalHarmonicsScale(max_degree=4, max_scale_log=0.3)
        
        # 情况 1：正常 scale（系数为 0）
        loss_fn_normal = RegularizationLoss(
            lambda_prior=1.0,
            lambda_smooth=0.01,
            lambda_scale=0.01,
        )
        
        loss_dict_normal = loss_fn_normal.compute_loss(
            log_depths=[log_depth],
            log_depth_daps=[log_depth_dap],
            depth_daps=[depth_dap],
            scale_modules=[scale_module],
        )
        
        # 情况 2：scale 爆炸（设置大系数）
        with torch.no_grad():
            scale_module.coeffs.data.fill_(10.0)  # 大系数
        
        loss_dict_exploded = loss_fn_normal.compute_loss(
            log_depths=[log_depth],
            log_depth_daps=[log_depth_dap],
            depth_daps=[depth_dap],
            scale_modules=[scale_module],
        )
        
        # scale 爆炸时的损失应该更大
        assert loss_dict_exploded['scale'].item() > loss_dict_normal['scale'].item() * 100, \
            "Scale 约束未正确惩罚爆炸"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
