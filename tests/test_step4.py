"""
Step 4 单元测试
测试最终联合优化
"""
import numpy as np
import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import (
    JointOptimizationConfig,
    optimize_multi_view_depth,
    validate_optimized_depths,
)
from src.deformation import DepthReparameterization


class MockRigid3d:
    """模拟 pycolmap.Rigid3d"""
    def __init__(self, R: np.ndarray, t: np.ndarray):
        self.rotation = MockRotation(R)
        self.translation = t


class MockRotation:
    """模拟 pycolmap.Rotation"""
    def __init__(self, R: np.ndarray):
        self._R = R
    
    def matrix(self) -> np.ndarray:
        return self._R


def create_test_camera_pose(translation: np.ndarray) -> MockRigid3d:
    """创建测试用的相机位姿"""
    R = np.eye(3)
    return MockRigid3d(R, translation)


class TestJointOptimization:
    """测试联合优化"""
    
    def test_optimization_convergence(self):
        """测试优化收敛性"""
        height, width = 100, 200
        
        # 创建深度重参数化模块
        depth_reparam_1 = DepthReparameterization(
            height=height,
            width=width,
            spline_type="linear",
            num_knots=10,
            scale_method="spherical_harmonics",
            sh_max_degree=4,
        )
        depth_reparam_2 = DepthReparameterization(
            height=height,
            width=width,
            spline_type="linear",
            num_knots=10,
            scale_method="spherical_harmonics",
            sh_max_degree=4,
        )
        
        # DAP 深度
        log_depth_dap = torch.ones(height, width) * 2.0  # log(10)
        depth_dap = torch.exp(log_depth_dap)
        
        # 相机位姿
        cam_pose_1 = create_test_camera_pose(np.array([0.0, 0.0, 0.0]))
        cam_pose_2 = create_test_camera_pose(np.array([1.0, 0.0, 0.0]))
        
        # 配置
        config = JointOptimizationConfig(
            lambda_p2r=1.0,
            lambda_depth=0.1,
            lambda_prior=1.0,
            lambda_smooth=0.01,
            lambda_scale=0.01,
            max_iter=10,  # 少量迭代用于测试
            early_stop_threshold=1e-6,
            device="cpu",
        )
        
        # 优化
        depths_opt, report = optimize_multi_view_depth(
            depth_reparam_modules=[depth_reparam_1, depth_reparam_2],
            log_depth_daps=[log_depth_dap, log_depth_dap],
            depth_daps=[depth_dap, depth_dap],
            cam_from_world_list=[cam_pose_1, cam_pose_2],
            config=config,
        )
        
        # 检查输出
        assert len(depths_opt) == 2, "输出深度数量不正确"
        assert depths_opt[0].shape == (height, width), "深度图形状不正确"
        assert report['iterations'] > 0, "迭代次数应为正"
        assert np.isfinite(report['final_energy']), "最终能量应为有限值"
    
    def test_no_collapse(self):
        """测试无塌缩：深度标准差应 > 阈值"""
        height, width = 100, 200
        
        # 创建正常深度（有变化）
        depth_normal = np.random.rand(height, width) * 50.0 + 10.0  # [10, 60] 米
        depth_dap = np.ones((height, width)) * 10.0
        
        is_valid, report = validate_optimized_depths(
            depths=[depth_normal],
            depth_daps=[depth_dap],
            max_collapse_std=0.1,
        )
        
        assert is_valid, "正常深度应通过验证"
        assert not report['collapse_detected'], "不应检测到塌缩"
    
    def test_collapse_detection(self):
        """测试塌缩检测：常数深度应被检测为塌缩"""
        height, width = 100, 200
        
        # 创建塌缩深度（常数）
        depth_collapsed = np.ones((height, width)) * 5.0
        depth_dap = np.ones((height, width)) * 10.0
        
        is_valid, report = validate_optimized_depths(
            depths=[depth_collapsed],
            depth_daps=[depth_dap],
            max_collapse_std=0.1,
        )
        
        assert not is_valid, "塌缩深度应未通过验证"
        assert report['collapse_detected'], "应检测到塌缩"
    
    def test_shell_detection(self):
        """测试球壳检测：100m 球壳应被检测"""
        height, width = 100, 200
        
        # 创建球壳深度（接近 100m）
        depth_shell = np.ones((height, width)) * 100.0
        depth_dap = np.ones((height, width)) * 10.0
        
        is_valid, report = validate_optimized_depths(
            depths=[depth_shell],
            depth_daps=[depth_dap],
            far_threshold=100.0,
        )
        
        assert not is_valid, "球壳深度应未通过验证"
        assert report['shell_detected'], "应检测到球壳"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
