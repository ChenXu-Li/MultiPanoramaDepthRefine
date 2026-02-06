"""
深度重参数化主模块
组合全局单调 spline 和方向相关缩放
D' = exp(S(log(D_DAP))) * s(theta, phi)
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional

from .monotonic_spline import MonotonicCubicSpline, LinearMonotonicSpline
from .directional_scale import create_directional_scale, SphericalHarmonicsScale, BSplineGridScale
from ..camera.spherical_camera import pixel_to_spherical_coords_torch, image_uv_grid


class DepthReparameterization(nn.Module):
    """
    深度重参数化模块
    
    实现：
        D' = exp(S(log(D_DAP))) * s(theta, phi)
    
    其中：
        - S: 全局单调 spline（log-depth 空间）
        - s: 方向相关缩放（球面函数）
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        # Spline 配置
        spline_type: Literal["monotonic_cubic", "linear"] = "monotonic_cubic",
        num_knots: int = 10,
        log_depth_min: float = -3.0,
        log_depth_max: float = 5.0,
        freeze_reference_point: bool = True,
        reference_log_depth: float = 0.0,
        # 方向缩放配置
        scale_method: Literal["spherical_harmonics", "bspline_grid"] = "spherical_harmonics",
        sh_max_degree: int = 4,
        bspline_grid_resolution: tuple[int, int] = (16, 8),
        max_scale_log: float = 0.3,
    ):
        """
        Args:
            height: 图像高度
            width: 图像宽度
            spline_type: Spline 类型
            num_knots: Spline 控制点数量
            log_depth_min: log-depth 最小值
            log_depth_max: log-depth 最大值
            freeze_reference_point: 是否冻结参考点
            reference_log_depth: 参考点位置
            scale_method: 方向缩放方法
            sh_max_degree: 球谐最大阶数（仅当 scale_method=spherical_harmonics）
            bspline_grid_resolution: B-spline grid 分辨率（仅当 scale_method=bspline_grid）
            max_scale_log: 缩放幅度限制（log 空间）
        """
        super().__init__()
        
        self.height = height
        self.width = width
        
        # 创建单调 spline
        if spline_type == "monotonic_cubic":
            self.spline = MonotonicCubicSpline(
                num_knots=num_knots,
                log_depth_min=log_depth_min,
                log_depth_max=log_depth_max,
                freeze_reference_point=freeze_reference_point,
                reference_log_depth=reference_log_depth,
            )
        elif spline_type == "linear":
            self.spline = LinearMonotonicSpline(
                num_knots=num_knots,
                log_depth_min=log_depth_min,
                log_depth_max=log_depth_max,
                freeze_reference_point=freeze_reference_point,
                reference_log_depth=reference_log_depth,
            )
        else:
            raise ValueError(f"Unknown spline_type: {spline_type}")
        
        # 创建方向缩放模块
        if scale_method == "spherical_harmonics":
            self.scale_module = create_directional_scale(
                method="spherical_harmonics",
                max_degree=sh_max_degree,
                max_scale_log=max_scale_log,
            )
        elif scale_method == "bspline_grid":
            self.scale_module = create_directional_scale(
                method="bspline_grid",
                grid_resolution=bspline_grid_resolution,
                max_scale_log=max_scale_log,
            )
        else:
            raise ValueError(f"Unknown scale_method: {scale_method}")
        
        # 预计算 UV 网格（用于方向缩放）
        u_grid, v_grid = image_uv_grid(height, width)
        self.register_buffer('u_grid', torch.from_numpy(u_grid))
        self.register_buffer('v_grid', torch.from_numpy(v_grid))
        
        # 预计算球面坐标
        theta, phi = pixel_to_spherical_coords_torch(
            torch.from_numpy(u_grid),
            torch.from_numpy(v_grid)
        )
        self.register_buffer('theta_grid', theta)
        self.register_buffer('phi_grid', phi)
    
    def forward(self, log_depth_dap: torch.Tensor) -> torch.Tensor:
        """
        应用深度重参数化
        
        Args:
            log_depth_dap: (B, H, W) 或 (H, W) DAP log-depth
            
        Returns:
            depth_transformed: (B, H, W) 或 (H, W) 变换后的深度
        """
        # 确保输入是 3D tensor
        if log_depth_dap.dim() == 2:
            log_depth_dap = log_depth_dap.unsqueeze(0)  # (1, H, W)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, H, W = log_depth_dap.shape
        
        # 1. 应用全局单调 spline: z' = S(log(D_DAP))
        log_depth_transformed = self.spline(log_depth_dap)  # (B, H, W)
        
        # 2. 计算方向相关缩放: s(theta, phi)
        # 扩展 theta_grid 和 phi_grid 到 batch 维度
        theta_batch = self.theta_grid.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        phi_batch = self.phi_grid.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        
        scale = self.scale_module(theta_batch, phi_batch)  # (B, H, W)
        
        # 3. 合成深度: D' = exp(z') * s
        depth_transformed = torch.exp(log_depth_transformed) * scale  # (B, H, W)
        
        if squeeze_output:
            depth_transformed = depth_transformed.squeeze(0)  # (H, W)
        
        return depth_transformed
    
    def forward_log_depth(self, log_depth_dap: torch.Tensor) -> torch.Tensor:
        """
        返回变换后的 log-depth（用于优化）
        
        Args:
            log_depth_dap: (B, H, W) 或 (H, W) DAP log-depth
            
        Returns:
            log_depth_transformed: (B, H, W) 或 (H, W) 变换后的 log-depth
        """
        depth_transformed = self.forward(log_depth_dap)
        return torch.log(depth_transformed + 1e-8)  # 避免 log(0)
    
    def ensure_monotonicity(self):
        """确保 spline 的单调性"""
        self.spline.ensure_monotonicity()
