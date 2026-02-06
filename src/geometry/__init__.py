"""
几何一致性模块
"""
from .coordinate_transform import (
    depth_to_world_points,
    get_camera_center_world,
    get_camera_ray_directions_world,
)
from .point_to_ray import (
    point_to_ray_distance,
    compute_point_to_ray_loss,
)
from .ray_space_consistency import (
    project_point_to_camera,
    compute_ray_space_depth_consistency_loss,
)
from .multi_view_loss import MultiViewGeometryLoss

__all__ = [
    'depth_to_world_points',
    'get_camera_center_world',
    'get_camera_ray_directions_world',
    'point_to_ray_distance',
    'compute_point_to_ray_loss',
    'project_point_to_camera',
    'compute_ray_space_depth_consistency_loss',
    'MultiViewGeometryLoss',
]
