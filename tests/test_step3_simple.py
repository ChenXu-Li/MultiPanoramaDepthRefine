#!/usr/bin/env python3
"""
Step 3 简单测试脚本
快速验证防退化护栏的基本功能
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_prior_anchor():
    """测试 log-depth 先验锚点"""
    print("=" * 60)
    print("测试 1: log-depth 先验锚点")
    print("=" * 60)
    
    from src.regularization import compute_prior_anchor_loss
    
    height, width = 100, 200
    
    # 零变形情况
    log_depth = torch.ones(height, width) * 2.0
    log_depth_dap = torch.ones(height, width) * 2.0
    
    loss_zero = compute_prior_anchor_loss(
        log_depth=log_depth,
        log_depth_dap=log_depth_dap,
        weight=1.0,
    )
    print(f"零变形损失: {loss_zero.item():.6f}")
    assert torch.isclose(loss_zero, torch.tensor(0.0), atol=1e-5), "零变形损失应为 0"
    
    # 塌缩情况
    log_depth_collapsed = torch.ones(height, width) * 0.0
    loss_collapsed = compute_prior_anchor_loss(
        log_depth=log_depth_collapsed,
        log_depth_dap=log_depth_dap,
        weight=1.0,
    )
    print(f"塌缩损失: {loss_collapsed.item():.6f}")
    assert loss_collapsed.item() > 1.0, "塌缩损失应较大"
    
    print("✅ 先验锚点测试通过\n")


def test_smoothness():
    """测试球面平滑正则"""
    print("=" * 60)
    print("测试 2: 球面平滑正则")
    print("=" * 60)
    
    from src.regularization import compute_spherical_smoothness_loss
    
    height, width = 100, 200
    
    # 常数深度
    log_depth_smooth = torch.ones(height, width) * 2.0
    loss_smooth = compute_spherical_smoothness_loss(
        log_depth=log_depth_smooth,
        weight=1.0,
        smooth_type="l2",
    )
    print(f"平滑深度损失: {loss_smooth.item():.6f}")
    assert torch.isclose(loss_smooth, torch.tensor(0.0), atol=1e-5), "平滑深度损失应为 0"
    
    # 噪声深度
    log_depth_noisy = torch.ones(height, width) * 2.0
    log_depth_noisy += torch.randn(height, width) * 0.1
    loss_noisy = compute_spherical_smoothness_loss(
        log_depth=log_depth_noisy,
        weight=1.0,
        smooth_type="l2",
    )
    print(f"噪声深度损失: {loss_noisy.item():.6f}")
    assert loss_noisy.item() > loss_smooth.item(), "噪声深度损失应更大"
    
    print("✅ 平滑正则测试通过\n")


def test_far_field_mask():
    """测试远景 mask"""
    print("=" * 60)
    print("测试 3: 远景 mask")
    print("=" * 60)
    
    from src.regularization import create_far_field_mask
    
    height, width = 100, 200
    
    # 创建包含远景的深度图
    depth_dap = torch.ones(height, width) * 50.0
    depth_dap[50:, :] = 150.0  # 下半部分是远景
    
    far_mask = create_far_field_mask(depth_dap, far_threshold=100.0)
    
    near_count = (~far_mask).sum().item()
    far_count = far_mask.sum().item()
    
    print(f"近景像素数: {near_count}")
    print(f"远景像素数: {far_count}")
    
    assert near_count > 0, "应该有近景像素"
    assert far_count > 0, "应该有远景像素"
    assert near_count + far_count == height * width, "像素总数应正确"
    
    print("✅ 远景 mask 测试通过\n")


def test_regularization_loss():
    """测试防退化护栏损失组合"""
    print("=" * 60)
    print("测试 4: 防退化护栏损失组合")
    print("=" * 60)
    
    from src.regularization import RegularizationLoss
    from src.deformation import SphericalHarmonicsScale
    
    height, width = 100, 200
    
    log_depth_dap = torch.ones(height, width) * 2.0
    depth_dap = torch.exp(log_depth_dap)
    
    # 正常深度
    log_depth_normal = torch.ones(height, width) * 2.0
    
    # 塌缩深度
    log_depth_collapsed = torch.ones(height, width) * 0.0
    
    # 创建损失函数
    loss_fn = RegularizationLoss(
        lambda_prior=1.0,
        lambda_smooth=0.01,
        lambda_scale=0.01,
    )
    
    # 正常情况
    loss_dict_normal = loss_fn.compute_loss(
        log_depths=[log_depth_normal],
        log_depth_daps=[log_depth_dap],
        depth_daps=[depth_dap],
    )
    
    # 塌缩情况
    loss_dict_collapsed = loss_fn.compute_loss(
        log_depths=[log_depth_collapsed],
        log_depth_daps=[log_depth_dap],
        depth_daps=[depth_dap],
    )
    
    print(f"正常情况总损失: {loss_dict_normal['total'].item():.6f}")
    print(f"塌缩情况总损失: {loss_dict_collapsed['total'].item():.6f}")
    print(f"先验损失（正常）: {loss_dict_normal['prior'].item():.6f}")
    print(f"先验损失（塌缩）: {loss_dict_collapsed['prior'].item():.6f}")
    
    # 塌缩情况的损失应该显著大于正常情况
    assert loss_dict_collapsed['total'].item() > loss_dict_normal['total'].item() * 10, \
        "防退化护栏未正确工作"
    
    print("✅ 防退化护栏损失测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Step 3 单元测试")
    print("=" * 60 + "\n")
    
    try:
        test_prior_anchor()
        test_smoothness()
        test_far_field_mask()
        test_regularization_loss()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
