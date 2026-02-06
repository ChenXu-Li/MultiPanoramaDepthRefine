"""
方向相关缩放（球谐函数或 B-spline grid）
实现 s_i(theta, phi) = exp(g_i(theta, phi))
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Literal


def spherical_harmonics_basis(theta: torch.Tensor, phi: torch.Tensor, max_degree: int) -> torch.Tensor:
    """
    计算球谐函数基函数（简化实现）
    
    使用实值球谐函数的简化形式，适用于低阶（max_degree <= 4）
    
    Args:
        theta: (...,) 方位角 [0, 2*pi]
        phi: (...,) 极角 [0, pi]
        max_degree: 最大阶数
        
    Returns:
        basis: (..., num_coeffs) 球谐基函数值
        num_coeffs = (max_degree + 1)^2
    """
    num_coeffs = (max_degree + 1) ** 2
    basis_list = []
    
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            # 实值球谐函数 Y_l^m(theta, phi) 的简化实现
            if l == 0:
                # Y_0^0 = 1 / sqrt(4*pi)
                y = torch.ones_like(phi) / np.sqrt(4.0 * np.pi)
            elif l == 1:
                if m == 0:
                    # Y_1^0 = sqrt(3/(4*pi)) * cos(phi)
                    y = cos_phi * np.sqrt(3.0 / (4.0 * np.pi))
                elif m == 1:
                    # Y_1^1 = sqrt(3/(8*pi)) * sin(phi) * cos(theta)
                    y = sin_phi * torch.cos(theta) * np.sqrt(3.0 / (8.0 * np.pi))
                else:  # m == -1
                    # Y_1^{-1} = sqrt(3/(8*pi)) * sin(phi) * sin(theta)
                    y = sin_phi * torch.sin(theta) * np.sqrt(3.0 / (8.0 * np.pi))
            elif l == 2:
                if m == 0:
                    # Y_2^0 = sqrt(5/(16*pi)) * (3*cos^2(phi) - 1)
                    y = (3 * cos_phi**2 - 1) * np.sqrt(5.0 / (16.0 * np.pi))
                elif m == 1:
                    # Y_2^1 = sqrt(15/(8*pi)) * sin(phi) * cos(phi) * cos(theta)
                    y = sin_phi * cos_phi * torch.cos(theta) * np.sqrt(15.0 / (8.0 * np.pi))
                elif m == -1:
                    # Y_2^{-1} = sqrt(15/(8*pi)) * sin(phi) * cos(phi) * sin(theta)
                    y = sin_phi * cos_phi * torch.sin(theta) * np.sqrt(15.0 / (8.0 * np.pi))
                elif m == 2:
                    # Y_2^2 = sqrt(15/(32*pi)) * sin^2(phi) * cos(2*theta)
                    y = sin_phi**2 * torch.cos(2 * theta) * np.sqrt(15.0 / (32.0 * np.pi))
                else:  # m == -2
                    # Y_2^{-2} = sqrt(15/(32*pi)) * sin^2(phi) * sin(2*theta)
                    y = sin_phi**2 * torch.sin(2 * theta) * np.sqrt(15.0 / (32.0 * np.pi))
            else:
                # 更高阶使用近似（简化）
                if m == 0:
                    y = torch.cos(l * phi) * np.sqrt((2 * l + 1) / (4.0 * np.pi))
                elif m > 0:
                    y = sin_phi * torch.cos(m * theta) * np.sqrt((2 * l + 1) / (4.0 * np.pi))
                else:
                    y = sin_phi * torch.sin(-m * theta) * np.sqrt((2 * l + 1) / (4.0 * np.pi))
            
            basis_list.append(y)
    
    basis = torch.stack(basis_list, dim=-1)  # (..., num_coeffs)
    return basis


class SphericalHarmonicsScale(nn.Module):
    """
    基于球谐函数的方向相关缩放
    
    s(theta, phi) = exp(g(theta, phi))
    其中 g(theta, phi) = sum_{l,m} c_{l,m} * Y_l^m(theta, phi)
    """
    
    def __init__(
        self,
        max_degree: int = 4,
        max_scale_log: float = 0.3,
    ):
        """
        Args:
            max_degree: 球谐函数最大阶数
            max_scale_log: g 的最大值（log 空间），|g| < max_scale_log
        """
        super().__init__()
        
        self.max_degree = max_degree
        self.max_scale_log = max_scale_log
        self.num_coeffs = (max_degree + 1) ** 2
        
        # 球谐系数（初始化为 0，对应 s = exp(0) = 1）
        self.coeffs = nn.Parameter(torch.zeros(self.num_coeffs))
    
    def forward(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        计算方向相关缩放因子
        
        Args:
            theta: (...,) 方位角 [0, 2*pi]
            phi: (...,) 极角 [0, pi]
            
        Returns:
            scale: (...,) 缩放因子 s = exp(g)，其中 g 被限制在 [-max_scale_log, max_scale_log]
        """
        # 限制系数范围（使用 tanh 确保范围）
        coeffs_max = torch.max(torch.abs(self.coeffs))
        if coeffs_max > 0:
            # 归一化系数，然后缩放到 [-max_scale_log, max_scale_log]
            coeffs_normalized = self.coeffs / coeffs_max * self.max_scale_log
        else:
            coeffs_normalized = self.coeffs
        
        # 计算球谐基函数
        basis = spherical_harmonics_basis(theta, phi, self.max_degree)  # (..., num_coeffs)
        
        # 计算 g(theta, phi) = sum(coeffs * basis)
        g = torch.sum(basis * coeffs_normalized.unsqueeze(0), dim=-1)  # (...)
        
        # 限制 g 的范围（双重保险）
        g = torch.clamp(g, -self.max_scale_log, self.max_scale_log)
        
        # 计算缩放因子 s = exp(g)
        scale = torch.exp(g)
        
        return scale


class BSplineGridScale(nn.Module):
    """
    基于 B-spline grid 的方向相关缩放
    
    s(theta, phi) = exp(g(theta, phi))
    其中 g(theta, phi) 由 B-spline grid 插值得到
    """
    
    def __init__(
        self,
        grid_resolution: tuple[int, int] = (16, 8),
        max_scale_log: float = 0.3,
    ):
        """
        Args:
            grid_resolution: (theta_resolution, phi_resolution) B-spline grid 分辨率
            max_scale_log: g 的最大值（log 空间）
        """
        super().__init__()
        
        self.grid_resolution = grid_resolution
        self.max_scale_log = max_scale_log
        theta_res, phi_res = grid_resolution
        
        # B-spline grid 控制点（初始化为 0）
        self.grid = nn.Parameter(torch.zeros(theta_res, phi_res))
    
    def forward(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        计算方向相关缩放因子
        
        Args:
            theta: (...,) 方位角 [0, 2*pi]
            phi: (...,) 极角 [0, pi]
            
        Returns:
            scale: (...,) 缩放因子
        """
        # 归一化到 grid 索引范围
        theta_norm = theta / (2.0 * np.pi) * self.grid_resolution[0]  # [0, theta_res]
        phi_norm = phi / np.pi * self.grid_resolution[1]  # [0, phi_res]
        
        # 双线性插值
        theta_idx = torch.clamp(torch.floor(theta_norm), 0, self.grid_resolution[0] - 1).long()
        phi_idx = torch.clamp(torch.floor(phi_norm), 0, self.grid_resolution[1] - 1).long()
        
        # 简化实现：最近邻插值（可以改为双线性插值）
        scale_log = self.grid[theta_idx, phi_idx]
        
        # 限制范围
        scale_log = torch.clamp(scale_log, -self.max_scale_log, self.max_scale_log)
        
        # 计算缩放因子
        scale = torch.exp(scale_log)
        
        return scale


def create_directional_scale(
    method: Literal["spherical_harmonics", "bspline_grid"],
    **kwargs
) -> nn.Module:
    """
    创建方向相关缩放模块
    
    Args:
        method: 'spherical_harmonics' 或 'bspline_grid'
        **kwargs: 传递给对应构造函数的参数
        
    Returns:
        scale_module: 方向缩放模块
    """
    if method == "spherical_harmonics":
        return SphericalHarmonicsScale(**kwargs)
    elif method == "bspline_grid":
        return BSplineGridScale(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
