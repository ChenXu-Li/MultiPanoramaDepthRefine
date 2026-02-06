#!/usr/bin/env python3
"""
Step 1 简单测试脚本
快速验证深度重参数化的基本功能
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera.spherical_camera import pixel_to_spherical_coords, pixel_to_directions
from src.deformation import DepthReparameterization, LinearMonotonicSpline, SphericalHarmonicsScale


def test_spherical_coords():
    """测试球面坐标转换"""
    print("=" * 60)
    print("测试 1: 球面坐标转换")
    print("=" * 60)
    
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    v = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    theta, phi = pixel_to_spherical_coords(u, v)
    print(f"U: {u}")
    print(f"V: {v}")
    print(f"Theta: {theta}")
    print(f"Phi: {phi}")
    
    # 检查范围
    assert np.all(theta >= 0) and np.all(theta <= 2 * np.pi), "Theta 超出范围"
    assert np.all(phi >= 0) and np.all(phi <= np.pi), "Phi 超出范围"
    
    # 测试方向向量
    directions = pixel_to_directions(u, v)
    norms = np.linalg.norm(directions, axis=-1)
    print(f"方向向量范数: {norms}")
    assert np.allclose(norms, 1.0), "方向向量不是单位向量"
    
    print("✅ 球面坐标转换测试通过\n")


def test_monotonic_spline():
    """测试单调 Spline"""
    print("=" * 60)
    print("测试 2: 单调 Spline")
    print("=" * 60)
    
    # 测试恒等映射时，不使用参考点冻结（避免破坏恒等映射）
    spline = LinearMonotonicSpline(
        num_knots=10,
        log_depth_min=-3.0,
        log_depth_max=5.0,
        freeze_reference_point=False,  # 禁用参考点冻结以测试恒等映射
        reference_log_depth=0.0,
    )
    
    # 测试恒等映射
    log_depth = torch.linspace(-3.0, 5.0, 100)
    output = spline(log_depth)
    
    print(f"输入范围: [{log_depth.min():.2f}, {log_depth.max():.2f}]")
    print(f"输出范围: [{output.min():.2f}, {output.max():.2f}]")
    
    # 检查恒等映射
    # 注意：线性插值在非 knot 位置会有误差，这是正常的
    # 我们主要检查 knot 位置是否恒等，以及整体偏差是否合理
    diff = torch.abs(output - log_depth)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"最大偏差: {max_diff:.6f}")
    print(f"平均偏差: {mean_diff:.6f}")
    
    # 检查 knot 位置是否恒等
    # 直接检查 knots_y 是否等于 knots_x（更精确）
    knot_diffs = torch.abs(spline.knots_y - spline.knots_x)
    max_knot_diff = knot_diffs.max().item()
    print(f"Knot 位置最大偏差: {max_knot_diff:.6f}")
    
    # Knot 位置应该完全恒等（初始化时）
    assert max_knot_diff < 1e-5, f"Knot 位置恒等映射偏差过大: {max_knot_diff}"
    # 整体偏差应该较小（线性插值的误差）
    assert max_diff < 0.5, f"整体偏差过大: {max_diff}"
    
    # 检查单调性
    diff_output = output[1:] - output[:-1]
    violation_count = (diff_output < -1e-6).sum().item()
    print(f"单调性违反次数: {violation_count}/99")
    assert violation_count == 0, "单调性违反"
    
    print("✅ 单调 Spline 测试通过\n")


def test_directional_scale():
    """测试方向相关缩放"""
    print("=" * 60)
    print("测试 3: 方向相关缩放")
    print("=" * 60)
    
    scale_module = SphericalHarmonicsScale(max_degree=4, max_scale_log=0.3)
    
    theta = torch.linspace(0, 2 * np.pi, 10)
    phi = torch.linspace(0, np.pi, 10)
    
    scale = scale_module(theta, phi)
    
    print(f"缩放范围: [{scale.min():.4f}, {scale.max():.4f}]")
    print(f"缩放均值: {scale.mean():.4f}")
    
    # 检查范围：s = exp(g), |g| < 0.3
    scale_min = np.exp(-0.3)
    scale_max = np.exp(0.3)
    assert torch.all(scale >= scale_min * 0.9), "缩放因子过小"
    assert torch.all(scale <= scale_max * 1.1), "缩放因子过大"
    
    # 初始状态应该接近 1
    assert torch.allclose(scale, torch.ones_like(scale), atol=0.1), "初始缩放不是 1"
    
    print("✅ 方向相关缩放测试通过\n")


def test_depth_reparameterization():
    """测试深度重参数化"""
    print("=" * 60)
    print("测试 4: 深度重参数化")
    print("=" * 60)
    
    height, width = 100, 200
    
    depth_reparam = DepthReparameterization(
        height=height,
        width=width,
        spline_type="linear",
        num_knots=10,
        scale_method="spherical_harmonics",
        sh_max_degree=4,
    )
    
    # 创建测试深度图
    depth_dap = torch.ones(height, width) * 10.0  # 10 米
    log_depth_dap = torch.log(depth_dap)
    
    print(f"输入深度: {depth_dap.mean():.2f} 米")
    
    # 应用变换
    depth_transformed = depth_reparam(log_depth_dap)
    
    print(f"输出深度: {depth_transformed.mean():.2f} 米")
    print(f"深度范围: [{depth_transformed.min():.2f}, {depth_transformed.max():.2f}] 米")
    
    # 检查正值性
    assert torch.all(depth_transformed > 0), "存在非正深度值"
    
    # 检查零变形一致性（初始状态应该接近恒等映射）
    diff = torch.abs(depth_transformed - depth_dap)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"最大偏差: {max_diff:.4f} 米")
    print(f"平均偏差: {mean_diff:.4f} 米")
    assert max_diff < 1.0, f"零变形一致性偏差过大: {max_diff}"
    
    print("✅ 深度重参数化测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Step 1 单元测试")
    print("=" * 60 + "\n")
    
    try:
        test_spherical_coords()
        test_monotonic_spline()
        test_directional_scale()
        test_depth_reparameterization()
        
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
