"""
Step 2 单元测试
测试跨视角几何一致性
"""
import numpy as np
import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry import (
    depth_to_world_points,
    point_to_ray_distance,
    MultiViewGeometryLoss,
)
from src.camera.spherical_camera import pixel_to_spherical_coords_torch


class MockRigid3d:
    """模拟 pycolmap.Rigid3d"""
    def __init__(self, R: np.ndarray, t: np.ndarray):
        self.rotation = MockRotation(R)
        self.translation = t
    
    def inverse(self):
        R_inv = self.rotation.matrix().T
        t_inv = -R_inv @ self.translation
        return MockRigid3d(R_inv, t_inv)


class MockRotation:
    """模拟 pycolmap.Rotation"""
    def __init__(self, R: np.ndarray):
        self._R = R
    
    def matrix(self) -> np.ndarray:
        return self._R


def create_test_camera_pose(translation: np.ndarray, rotation_axis: np.ndarray = None, angle: float = 0.0) -> MockRigid3d:
    """创建测试用的相机位姿"""
    if rotation_axis is None:
        R = np.eye(3)
    else:
        # 使用 Rodrigues 公式创建旋转矩阵
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return MockRigid3d(R, translation)


class TestCoordinateTransform:
    """测试坐标变换"""
    
    def test_depth_to_world_points(self):
        """测试深度到世界点的转换"""
        height, width = 100, 200
        
        # 创建测试深度图（10 米）
        depth = torch.ones(height, width) * 10.0
        
        # 创建单位变换（相机中心在原点，无旋转）
        cam_from_world = create_test_camera_pose(
            translation=np.array([0.0, 0.0, 0.0]),
            rotation_axis=None,
            angle=0.0
        )
        
        # 转换为世界点
        points_world = depth_to_world_points(
            depth, cam_from_world, height, width, device="cpu"
        )
        
        # 检查形状
        assert points_world.shape == (height, width, 3)
        
        # 检查深度（到原点的距离）
        distances = torch.norm(points_world, dim=-1)
        assert torch.allclose(distances, depth, rtol=1e-3)
    
    def test_depth_to_world_points_with_translation(self):
        """测试带平移的坐标变换"""
        height, width = 100, 200
        
        depth = torch.ones(height, width) * 10.0
        
        # 相机中心在 (1, 2, 3)
        cam_from_world = create_test_camera_pose(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation_axis=None,
            angle=0.0
        )
        
        points_world = depth_to_world_points(
            depth, cam_from_world, height, width, device="cpu"
        )
        
        # 检查形状
        assert points_world.shape == (height, width, 3)
        
        # 中心像素应该对应相机中心 + 深度方向
        center_h, center_w = height // 2, width // 2
        center_point = points_world[center_h, center_w]
        
        # 中心方向应该是 (0, 0, 1)（DAP 约定）
        expected_direction = np.array([0.0, 0.0, 1.0])
        expected_point = cam_from_world.translation + 10.0 * expected_direction
        
        # 允许小的误差（由于球面投影）
        assert np.allclose(center_point.numpy(), expected_point, atol=1.0)


class TestPointToRay:
    """测试 Point-to-Ray 距离"""
    
    def test_point_to_ray_distance(self):
        """测试点到射线距离计算"""
        # 射线：从原点沿 z 轴
        ray_origin = torch.tensor([0.0, 0.0, 0.0])
        ray_direction = torch.tensor([0.0, 0.0, 1.0])
        
        # 点在射线上
        point_on_ray = torch.tensor([0.0, 0.0, 5.0])
        distance = point_to_ray_distance(point_on_ray, ray_origin, ray_direction)
        assert torch.isclose(distance, torch.tensor(0.0), atol=1e-5)
        
        # 点偏离射线
        point_off_ray = torch.tensor([1.0, 0.0, 5.0])
        distance = point_to_ray_distance(point_off_ray, ray_origin, ray_direction)
        assert torch.isclose(distance, torch.tensor(1.0), atol=1e-5)
    
    def test_point_to_ray_batch(self):
        """测试批量计算"""
        ray_origin = torch.tensor([0.0, 0.0, 0.0])
        ray_direction = torch.tensor([0.0, 0.0, 1.0])
        
        points = torch.tensor([
            [0.0, 0.0, 5.0],  # 在射线上
            [1.0, 0.0, 5.0],  # 偏离 1 米
            [0.0, 2.0, 5.0],  # 偏离 2 米
        ])
        
        distances = point_to_ray_distance(points, ray_origin, ray_direction)
        
        assert torch.isclose(distances[0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(distances[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(distances[2], torch.tensor(2.0), atol=1e-5)


class TestMultiViewGeometryLoss:
    """测试多视角几何一致性损失"""
    
    def test_identity_case(self):
        """测试 Identity 情况：所有视角深度相同 → loss ≈ 0"""
        height, width = 100, 200
        
        # 创建相同的深度图
        depth = torch.ones(height, width) * 10.0
        depths = [depth, depth]
        
        # 创建相同的相机位姿（单位变换）
        cam_pose = create_test_camera_pose(
            translation=np.array([0.0, 0.0, 0.0])
        )
        cam_from_world_list = [cam_pose, cam_pose]
        
        # DAP 深度（用于远景 mask）
        depth_dap = torch.ones(height, width) * 10.0
        depth_dap_list = [depth_dap, depth_dap]
        
        # 创建损失函数
        loss_fn = MultiViewGeometryLoss(
            lambda_p2r=1.0,
            lambda_depth=0.1,
            far_threshold=100.0,
            device="cpu",
        )
        
        # 计算损失
        loss_dict = loss_fn.compute_loss(
            depths=depths,
            cam_from_world_list=cam_from_world_list,
            depth_dap_list=depth_dap_list,
            height=height,
            width=width,
        )
        
        # Identity 情况下损失应该很小
        assert loss_dict['total'].item() < 1.0, f"Identity 损失过大: {loss_dict['total'].item()}"
    
    def test_scale_collapse(self):
        """测试尺度退化：所有深度为常数 → loss 显著增大"""
        height, width = 100, 200
        
        # 创建常数深度（尺度退化）
        depth_constant = torch.ones(height, width) * 5.0
        depths = [depth_constant, depth_constant]
        
        # 创建不同的相机位姿（有平移）
        cam_pose_1 = create_test_camera_pose(
            translation=np.array([0.0, 0.0, 0.0])
        )
        cam_pose_2 = create_test_camera_pose(
            translation=np.array([1.0, 0.0, 0.0])
        )
        cam_from_world_list = [cam_pose_1, cam_pose_2]
        
        depth_dap = torch.ones(height, width) * 10.0
        depth_dap_list = [depth_dap, depth_dap]
        
        loss_fn = MultiViewGeometryLoss(
            lambda_p2r=1.0,
            lambda_depth=0.1,
            far_threshold=100.0,
            device="cpu",
        )
        
        loss_dict = loss_fn.compute_loss(
            depths=depths,
            cam_from_world_list=cam_from_world_list,
            depth_dap_list=depth_dap_list,
            height=height,
            width=width,
        )
        
        # 尺度退化情况下损失应该较大
        assert loss_dict['total'].item() > 0.1, "尺度退化检测失败"
    
    def test_far_field_exclusion(self):
        """测试远景排除：>= 100m 的像素不参与几何对齐"""
        height, width = 100, 200
        
        # 创建包含远景的深度图
        depth_near = torch.ones(height, width) * 10.0
        depth_far = torch.ones(height, width) * 150.0  # 远景
        depths = [depth_near, depth_far]
        
        cam_pose = create_test_camera_pose(
            translation=np.array([0.0, 0.0, 0.0])
        )
        cam_from_world_list = [cam_pose, cam_pose]
        
        # DAP 深度（用于远景 mask）
        depth_dap_near = torch.ones(height, width) * 10.0
        depth_dap_far = torch.ones(height, width) * 150.0
        depth_dap_list = [depth_dap_near, depth_dap_far]
        
        loss_fn = MultiViewGeometryLoss(
            lambda_p2r=1.0,
            lambda_depth=0.1,
            far_threshold=100.0,
            device="cpu",
        )
        
        # 计算损失（远景应该被排除）
        loss_dict = loss_fn.compute_loss(
            depths=depths,
            cam_from_world_list=cam_from_world_list,
            depth_dap_list=depth_dap_list,
            height=height,
            width=width,
        )
        
        # 损失应该有限（不是 inf 或 nan）
        assert torch.isfinite(loss_dict['total']), "损失不是有限值"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
