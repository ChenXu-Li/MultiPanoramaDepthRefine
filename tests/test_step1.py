"""
Step 1 单元测试
测试深度重参数化的正确性
"""
import numpy as np
import torch
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera.spherical_camera import (
    pixel_to_spherical_coords,
    spherical_coords_to_directions,
    pixel_to_directions,
)
from src.deformation import (
    DepthReparameterization,
    MonotonicCubicSpline,
    LinearMonotonicSpline,
    SphericalHarmonicsScale,
)


class TestSphericalCoordinates:
    """测试球面坐标转换"""
    
    def test_pixel_to_spherical_coords(self):
        """测试像素到球面坐标转换"""
        u = np.array([0.0, 0.5, 1.0])
        v = np.array([0.0, 0.5, 1.0])
        
        theta, phi = pixel_to_spherical_coords(u, v)
        
        # 检查范围
        assert np.all(theta >= 0) and np.all(theta <= 2 * np.pi)
        assert np.all(phi >= 0) and np.all(phi <= np.pi)
        
        # 检查边界值
        theta_0, phi_0 = pixel_to_spherical_coords(np.array([0.0]), np.array([0.0]))
        assert np.isclose(theta_0[0], 2 * np.pi) or np.isclose(theta_0[0], 0)
        assert np.isclose(phi_0[0], 0)
    
    def test_spherical_coords_to_directions(self):
        """测试球面坐标到方向向量转换"""
        theta = np.array([0.0, np.pi / 2, np.pi])
        phi = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        
        directions = spherical_coords_to_directions(theta, phi)
        
        # 检查形状
        assert directions.shape == (3, 3)
        
        # 检查单位向量
        norms = np.linalg.norm(directions, axis=-1)
        assert np.allclose(norms, 1.0)
    
    def test_pixel_to_directions(self):
        """测试像素到方向向量转换"""
        u = np.array([[0.0, 0.5], [0.5, 1.0]])
        v = np.array([[0.0, 0.5], [0.5, 1.0]])
        
        directions = pixel_to_directions(u, v)
        
        # 检查形状
        assert directions.shape == (2, 2, 3)
        
        # 检查单位向量
        norms = np.linalg.norm(directions, axis=-1)
        assert np.allclose(norms, 1.0)


class TestMonotonicSpline:
    """测试单调 Spline"""
    
    def test_identity_mapping(self):
        """测试恒等映射（零变形）"""
        spline = LinearMonotonicSpline(
            num_knots=10,
            log_depth_min=-3.0,
            log_depth_max=5.0,
            freeze_reference_point=True,
            reference_log_depth=0.0,
        )
        
        # 初始化后应该是恒等映射
        log_depth = torch.linspace(-3.0, 5.0, 100)
        output = spline(log_depth)
        
        # 应该接近恒等映射（允许小的数值误差）
        assert torch.allclose(output, log_depth, atol=1e-5)
    
    def test_monotonicity(self):
        """测试单调性"""
        spline = LinearMonotonicSpline(
            num_knots=10,
            log_depth_min=-3.0,
            log_depth_max=5.0,
        )
        
        # 创建单调递增输入
        log_depth = torch.linspace(-3.0, 5.0, 100)
        output = spline(log_depth)
        
        # 检查输出是否单调递增
        diff = output[1:] - output[:-1]
        assert torch.all(diff >= -1e-6)  # 允许小的数值误差
    
    def test_reference_point_freeze(self):
        """测试参考点冻结"""
        spline = LinearMonotonicSpline(
            num_knots=10,
            log_depth_min=-3.0,
            log_depth_max=5.0,
            freeze_reference_point=True,
            reference_log_depth=0.0,
        )
        
        # 参考点应该保持恒等映射
        ref_input = torch.tensor([0.0])
        ref_output = spline(ref_input)
        assert torch.isclose(ref_output[0], torch.tensor(0.0))


class TestDirectionalScale:
    """测试方向相关缩放"""
    
    def test_zero_scale(self):
        """测试零缩放（g=0，s=1）"""
        scale_module = SphericalHarmonicsScale(max_degree=4, max_scale_log=0.3)
        
        # 初始系数为 0，应该对应 s = exp(0) = 1
        theta = torch.linspace(0, 2 * np.pi, 10)
        phi = torch.linspace(0, np.pi, 10)
        
        scale = scale_module(theta, phi)
        assert torch.allclose(scale, torch.ones_like(scale), atol=1e-5)
    
    def test_scale_range(self):
        """测试缩放范围限制"""
        scale_module = SphericalHarmonicsScale(max_degree=4, max_scale_log=0.3)
        
        # 设置较大的系数
        with torch.no_grad():
            scale_module.coeffs.data.fill_(1.0)
        
        theta = torch.linspace(0, 2 * np.pi, 10)
        phi = torch.linspace(0, np.pi, 10)
        
        scale = scale_module(theta, phi)
        
        # 检查范围：s = exp(g)，|g| < 0.3
        scale_min = np.exp(-0.3)
        scale_max = np.exp(0.3)
        assert torch.all(scale >= scale_min * 0.9)  # 允许小的数值误差
        assert torch.all(scale <= scale_max * 1.1)


class TestDepthReparameterization:
    """测试深度重参数化"""
    
    def test_zero_deformation(self):
        """测试零变形一致性"""
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
        
        # 应用变换（初始状态应该是恒等映射）
        depth_transformed = depth_reparam(log_depth_dap)
        
        # 应该接近原始深度（允许小的数值误差）
        assert torch.allclose(depth_transformed, depth_dap, rtol=1e-3)
    
    def test_monotonicity(self):
        """测试单调性保持"""
        height, width = 100, 200
        
        depth_reparam = DepthReparameterization(
            height=height,
            width=width,
            spline_type="linear",
            num_knots=10,
            scale_method="spherical_harmonics",
            sh_max_degree=4,
        )
        
        # 创建单调递增的深度图
        u = torch.linspace(0, 1, width)
        v = torch.linspace(0, 1, height)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
        
        # 深度随 u 增加
        depth_dap = (u_grid * 50.0 + 1.0).float()  # [1, 51] 米
        log_depth_dap = torch.log(depth_dap)
        
        # 应用变换
        depth_transformed = depth_reparam(log_depth_dap)
        
        # 检查同一行的深度是否单调（允许小的数值误差）
        for i in range(height):
            row_depths = depth_transformed[i, :]
            diff = row_depths[1:] - row_depths[:-1]
            # 允许小的非单调性（由于方向缩放）
            violation_rate = (diff < -0.1).float().mean()
            assert violation_rate < 0.1  # 违反率 < 10%
    
    def test_positive_depth(self):
        """测试正值性"""
        height, width = 100, 200
        
        depth_reparam = DepthReparameterization(
            height=height,
            width=width,
            spline_type="linear",
            num_knots=10,
            scale_method="spherical_harmonics",
            sh_max_degree=4,
        )
        
        # 创建正深度图
        depth_dap = torch.ones(height, width) * 5.0
        log_depth_dap = torch.log(depth_dap)
        
        # 应用变换
        depth_transformed = depth_reparam(log_depth_dap)
        
        # 所有深度应该 > 0
        assert torch.all(depth_transformed > 0)
    
    def test_continuity(self):
        """测试连续性"""
        height, width = 100, 200
        
        depth_reparam = DepthReparameterization(
            height=height,
            width=width,
            spline_type="linear",
            num_knots=10,
            scale_method="spherical_harmonics",
            sh_max_degree=4,
        )
        
        # 创建平滑深度图
        depth_dap = torch.ones(height, width) * 10.0
        log_depth_dap = torch.log(depth_dap)
        
        # 应用变换
        depth_transformed = depth_reparam(log_depth_dap)
        
        # 检查相邻像素的深度差（应该较小）
        grad_x = torch.abs(depth_transformed[:, 1:] - depth_transformed[:, :-1])
        grad_y = torch.abs(depth_transformed[1:, :] - depth_transformed[:-1, :])
        
        # 平均梯度应该较小（深度场连续）
        mean_grad = (grad_x.mean() + grad_y.mean()) / 2
        assert mean_grad < 1.0  # 平均梯度 < 1 米


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
