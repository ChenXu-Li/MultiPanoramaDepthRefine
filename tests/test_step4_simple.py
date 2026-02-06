"""
Step 4 简化测试
测试最终联合优化的基本功能
"""
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import JointOptimizationConfig, optimize_multi_view_depth
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


def test_basic_optimization():
    """基本优化测试"""
    print("测试基本优化功能...")
    
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
        max_iter=5,  # 少量迭代用于测试
        early_stop_threshold=1e-6,
        device="cpu",
    )
    
    # 优化
    try:
        depths_opt, report = optimize_multi_view_depth(
            depth_reparam_modules=[depth_reparam_1, depth_reparam_2],
            log_depth_daps=[log_depth_dap, log_depth_dap],
            depth_daps=[depth_dap, depth_dap],
            cam_from_world_list=[cam_pose_1, cam_pose_2],
            config=config,
        )
        
        print(f"  ✅ 优化完成")
        print(f"  迭代次数: {report['iterations']}")
        print(f"  最终能量: {report['final_energy']:.6f}")
        print(f"  输出深度形状: {depths_opt[0].shape}")
        
        # 检查输出
        assert len(depths_opt) == 2, "输出深度数量不正确"
        assert depths_opt[0].shape == (height, width), "深度图形状不正确"
        assert np.all(np.isfinite(depths_opt[0])), "深度值应为有限值"
        
        print("  ✅ 所有检查通过")
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_basic_optimization()
    print("\n✅ Step 4 基本测试完成")
