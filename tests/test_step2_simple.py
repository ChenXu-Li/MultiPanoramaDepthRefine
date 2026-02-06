#!/usr/bin/env python3
"""
Step 2 简单测试脚本
快速验证跨视角几何一致性的基本功能
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


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


def create_test_camera_pose(translation: np.ndarray) -> MockRigid3d:
    """创建测试用的相机位姿"""
    R = np.eye(3)
    return MockRigid3d(R, translation)


def test_coordinate_transform():
    """测试坐标变换"""
    print("=" * 60)
    print("测试 1: 深度 → 世界点坐标变换")
    print("=" * 60)
    
    from src.geometry import depth_to_world_points
    
    height, width = 100, 200
    depth = torch.ones(height, width) * 10.0
    
    cam_from_world = create_test_camera_pose(
        translation=np.array([0.0, 0.0, 0.0])
    )
    
    points_world = depth_to_world_points(
        depth, cam_from_world, height, width, device="cpu"
    )
    
    print(f"深度图形状: {depth.shape}")
    print(f"世界点形状: {points_world.shape}")
    print(f"世界点范围: [{points_world.min():.2f}, {points_world.max():.2f}]")
    
    # 检查深度
    distances = torch.norm(points_world, dim=-1)
    mean_distance = distances.mean().item()
    print(f"平均距离: {mean_distance:.2f} 米")
    
    assert points_world.shape == (height, width, 3), "形状不正确"
    assert abs(mean_distance - 10.0) < 1.0, f"深度不正确: {mean_distance}"
    
    print("✅ 坐标变换测试通过\n")


def test_point_to_ray():
    """测试 Point-to-Ray 距离"""
    print("=" * 60)
    print("测试 2: Point-to-Ray 距离计算")
    print("=" * 60)
    
    from src.geometry import point_to_ray_distance
    
    # 射线：从原点沿 z 轴
    ray_origin = torch.tensor([0.0, 0.0, 0.0])
    ray_direction = torch.tensor([0.0, 0.0, 1.0])
    
    # 点在射线上
    point_on_ray = torch.tensor([0.0, 0.0, 5.0])
    distance = point_to_ray_distance(point_on_ray, ray_origin, ray_direction)
    print(f"点在射线上，距离: {distance.item():.6f}")
    assert torch.isclose(distance, torch.tensor(0.0), atol=1e-5), "点在射线上距离应为 0"
    
    # 点偏离射线
    point_off_ray = torch.tensor([1.0, 0.0, 5.0])
    distance = point_to_ray_distance(point_off_ray, ray_origin, ray_direction)
    print(f"点偏离射线 1 米，距离: {distance.item():.6f}")
    assert torch.isclose(distance, torch.tensor(1.0), atol=1e-5), "距离计算错误"
    
    print("✅ Point-to-Ray 距离测试通过\n")


def test_multi_view_loss_identity():
    """测试多视角损失：Identity 情况"""
    print("=" * 60)
    print("测试 3: 多视角几何损失（Identity 情况）")
    print("=" * 60)
    
    from src.geometry import MultiViewGeometryLoss
    
    height, width = 100, 200
    
    # 创建相同的深度图
    depth = torch.ones(height, width) * 10.0
    depths = [depth, depth]
    
    # 创建相同的相机位姿
    cam_pose = create_test_camera_pose(
        translation=np.array([0.0, 0.0, 0.0])
    )
    cam_from_world_list = [cam_pose, cam_pose]
    
    # DAP 深度
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
    
    print(f"总损失: {loss_dict['total'].item():.6f}")
    print(f"P2R 损失: {loss_dict['p2r'].item():.6f}")
    print(f"深度一致性损失: {loss_dict['depth'].item():.6f}")
    
    # Identity 情况下损失应该很小
    assert loss_dict['total'].item() < 10.0, f"Identity 损失过大: {loss_dict['total'].item()}"
    
    print("✅ 多视角损失（Identity）测试通过\n")


def test_far_field_exclusion():
    """测试远景排除"""
    print("=" * 60)
    print("测试 4: 远景排除（>= 100m）")
    print("=" * 60)
    
    from src.geometry import MultiViewGeometryLoss
    
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
    
    # 计算损失
    loss_dict = loss_fn.compute_loss(
        depths=depths,
        cam_from_world_list=cam_from_world_list,
        depth_dap_list=depth_dap_list,
        height=height,
        width=width,
    )
    
    print(f"总损失: {loss_dict['total'].item():.6f}")
    print(f"P2R 损失: {loss_dict['p2r'].item():.6f}")
    
    # 损失应该有限（不是 inf 或 nan）
    assert torch.isfinite(loss_dict['total']), "损失不是有限值"
    
    print("✅ 远景排除测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Step 2 单元测试")
    print("=" * 60 + "\n")
    
    try:
        test_coordinate_transform()
        test_point_to_ray()
        test_multi_view_loss_identity()
        test_far_field_exclusion()
        
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
