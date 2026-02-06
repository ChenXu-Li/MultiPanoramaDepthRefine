#!/usr/bin/env python3
"""
测试方向 × log-depth B-spline grid 模块
"""
import sys
from pathlib import Path
import numpy as np
import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deformation import DepthReparameterization
from src.regularization import compute_bspline_constraints_loss


def test_directional_bspline_creation():
    """测试创建方向 B-spline grid 模块"""
    print("=" * 60)
    print("测试 1: 创建方向 B-spline grid 模块")
    print("=" * 60)
    
    H, W = 100, 200
    
    module = DepthReparameterization(
        height=H,
        width=W,
        use_directional_bspline=True,
        n_alpha=12,
        n_depth=10,
        alpha_method="asin",
        max_delta_log=0.5,
        log_depth_min=-3.0,
        log_depth_max=5.0,
    )
    
    print(f"✅ 模块创建成功")
    print(f"   use_directional_bspline: {module.use_directional_bspline}")
    print(f"   directional_bspline: {module.directional_bspline is not None}")
    print(f"   控制点形状: {module.directional_bspline.get_control_points().shape}")
    
    return module


def test_forward_pass(module):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试 2: 前向传播")
    print("=" * 60)
    
    H, W = 100, 200
    
    # 创建测试输入（log-depth）
    log_depth_dap = torch.randn(H, W) * 2.0 + 1.0  # 大致在 [-1, 3] 范围
    
    # 前向传播
    depth_transformed = module(log_depth_dap)
    log_depth_transformed = torch.log(depth_transformed + 1e-8)
    
    print(f"✅ 前向传播成功")
    print(f"   输入形状: {log_depth_dap.shape}")
    print(f"   输出形状: {depth_transformed.shape}")
    print(f"   输入 log-depth 范围: [{log_depth_dap.min():.3f}, {log_depth_dap.max():.3f}]")
    print(f"   输出深度范围: [{depth_transformed.min():.3f}, {depth_transformed.max():.3f}]")
    print(f"   输出 log-depth 范围: [{log_depth_transformed.min():.3f}, {log_depth_transformed.max():.3f}]")
    
    # 检查输出是否合理
    assert depth_transformed.shape == (H, W), "输出形状不正确"
    assert torch.all(depth_transformed > 0), "深度值必须 > 0"
    assert torch.all(torch.isfinite(depth_transformed)), "深度值必须有限"
    
    print(f"✅ 输出验证通过")


def test_bspline_constraints(module):
    """测试 B-spline 约束损失"""
    print("\n" + "=" * 60)
    print("测试 3: B-spline 约束损失")
    print("=" * 60)
    
    control_points = module.directional_bspline.get_control_points()
    
    # 计算约束损失
    constraint_losses = compute_bspline_constraints_loss(
        control_points=control_points,
        lambda_mono=0.1,
        lambda_smooth=0.001,
        lambda_far=0.1,
    )
    
    print(f"✅ 约束损失计算成功")
    print(f"   单调性约束: {constraint_losses['monotonicity'].item():.6f}")
    print(f"   方向平滑正则: {constraint_losses['smoothness'].item():.6f}")
    print(f"   远景渐近约束: {constraint_losses['far_field'].item():.6f}")
    print(f"   总约束损失: {constraint_losses['total'].item():.6f}")
    
    # 检查损失是否合理（初始应该很小，因为控制点初始化为 0）
    assert constraint_losses['total'].item() >= 0, "损失必须 >= 0"
    assert constraint_losses['monotonicity'].item() >= 0, "单调性损失必须 >= 0"
    assert constraint_losses['smoothness'].item() >= 0, "平滑损失必须 >= 0"
    assert constraint_losses['far_field'].item() >= 0, "远景损失必须 >= 0"
    
    print(f"✅ 约束损失验证通过")


def test_gradient_flow(module):
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("测试 4: 梯度流")
    print("=" * 60)
    
    H, W = 100, 200
    log_depth_dap = torch.randn(H, W) * 2.0 + 1.0
    
    # 前向传播
    depth_transformed = module(log_depth_dap)
    loss = depth_transformed.mean()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    control_points = module.directional_bspline.get_control_points()
    has_grad = control_points.grad is not None
    
    print(f"✅ 梯度流测试成功")
    print(f"   控制点是否有梯度: {has_grad}")
    if has_grad:
        grad_norm = control_points.grad.norm().item()
        print(f"   梯度范数: {grad_norm:.6f}")
    
    assert has_grad, "控制点必须有梯度"
    
    print(f"✅ 梯度流验证通过")


def main():
    """主测试函数"""
    print("开始测试方向 × log-depth B-spline grid 模块\n")
    
    try:
        # 测试 1: 创建模块
        module = test_directional_bspline_creation()
        
        # 测试 2: 前向传播
        test_forward_pass(module)
        
        # 测试 3: B-spline 约束损失
        test_bspline_constraints(module)
        
        # 测试 4: 梯度流
        test_gradient_flow(module)
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
