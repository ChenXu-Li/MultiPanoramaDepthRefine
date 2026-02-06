"""
球面相机模型：等轴柱状（equirectangular）投影
使用 DAP 约定的球面坐标转换
"""
import numpy as np
import torch


def pixel_to_spherical_coords(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    将像素坐标 (u, v) 转换为球面坐标 (theta, phi)
    
    使用 DAP 约定：
        - u ∈ [0, 1], v ∈ [0, 1]
        - theta = (1 - u) * 2*pi  # [0, 2*pi]
        - phi = v * pi  # [0, pi]
    
    Args:
        u: (H, W) 或 (N,) 归一化 U 坐标 [0, 1]
        v: (H, W) 或 (N,) 归一化 V 坐标 [0, 1]
        
    Returns:
        theta: (H, W) 或 (N,) 方位角 [0, 2*pi]
        phi: (H, W) 或 (N,) 极角 [0, pi]
    """
    theta = (1.0 - u) * (2.0 * np.pi)  # [0, 2*pi]
    phi = v * np.pi  # [0, pi]
    return theta, phi


def spherical_coords_to_pixel(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    将球面坐标 (theta, phi) 转换为像素坐标 (u, v)
    
    DAP 约定的逆变换：
        - u = 1 - theta / (2*pi)  # [0, 1]
        - v = phi / pi  # [0, 1]
    
    Args:
        theta: (H, W) 或 (N,) 方位角 [0, 2*pi]
        phi: (H, W) 或 (N,) 极角 [0, pi]
        
    Returns:
        u: (H, W) 或 (N,) 归一化 U 坐标 [0, 1]
        v: (H, W) 或 (N,) 归一化 V 坐标 [0, 1]
    """
    u = 1.0 - theta / (2.0 * np.pi)  # [0, 1]
    v = phi / np.pi  # [0, 1]
    return u, v


def spherical_coords_to_directions(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    将球面坐标 (theta, phi) 转换为单位方向向量
    
    DAP 约定：
        x = sin(phi) * cos(theta)
        y = sin(phi) * sin(theta)
        z = cos(phi)
    
    Args:
        theta: (H, W) 或 (N,) 方位角 [0, 2*pi]
        phi: (H, W) 或 (N,) 极角 [0, pi]
        
    Returns:
        directions: (H, W, 3) 或 (N, 3) 单位方向向量
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    directions = np.stack([
        sin_phi * cos_theta,
        sin_phi * sin_theta,
        cos_phi
    ], axis=-1)
    
    return directions


def pixel_to_directions(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    将像素坐标 (u, v) 直接转换为单位方向向量
    
    Args:
        u: (H, W) 或 (N,) 归一化 U 坐标 [0, 1]
        v: (H, W) 或 (N,) 归一化 V 坐标 [0, 1]
        
    Returns:
        directions: (H, W, 3) 或 (N, 3) 单位方向向量
    """
    theta, phi = pixel_to_spherical_coords(u, v)
    return spherical_coords_to_directions(theta, phi)


def image_uv_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """
    生成图像的 UV 坐标网格
    
    Args:
        height: 图像高度
        width: 图像宽度
        
    Returns:
        u_grid: (H, W) U 坐标网格 [0, 1]
        v_grid: (H, W) V 坐标网格 [0, 1]
    """
    u = np.linspace(0, 1, width, dtype=np.float32)  # [W]
    v = np.linspace(0, 1, height, dtype=np.float32)  # [H]
    u_grid, v_grid = np.meshgrid(u, v)  # [H, W]
    return u_grid, v_grid


# PyTorch 版本（用于优化）
def pixel_to_spherical_coords_torch(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本的像素到球面坐标转换"""
    theta = (1.0 - u) * (2.0 * np.pi)
    phi = v * np.pi
    return theta, phi


def spherical_coords_to_directions_torch(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本的球面坐标到方向向量转换"""
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    directions = torch.stack([
        sin_phi * cos_theta,
        sin_phi * sin_theta,
        cos_phi
    ], dim=-1)
    
    return directions


def pixel_to_directions_torch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本的像素到方向向量转换"""
    theta, phi = pixel_to_spherical_coords_torch(u, v)
    return spherical_coords_to_directions_torch(theta, phi)
