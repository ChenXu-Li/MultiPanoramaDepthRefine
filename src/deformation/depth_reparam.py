"""
深度重参数化主模块
组合全局单调 spline 和方向相关缩放
D' = exp(S(log(D_DAP))) * s(theta, phi)

新版本（v2）：
D' = exp(log(D_DAP) + Δ(alpha, log(D_DAP)))
其中 Δ 是方向 × log-depth 的 B-spline grid
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional

from .monotonic_spline import MonotonicCubicSpline, LinearMonotonicSpline
from .directional_scale import create_directional_scale, SphericalHarmonicsScale, BSplineGridScale
from .directional_bspline_grid import DirectionalBSplineGrid
from ..camera.spherical_camera import (
    pixel_to_spherical_coords_torch,
    image_uv_grid,
    pixel_to_directions_torch,
)


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
        # 方向缩放配置（旧版本）
        scale_method: Literal["spherical_harmonics", "bspline_grid", "directional_bspline_grid"] = "directional_bspline_grid",
        sh_max_degree: int = 4,
        bspline_grid_resolution: tuple[int, int] = (16, 8),
        max_scale_log: float = 0.3,
        # 新版本：方向 × log-depth B-spline grid 配置
        use_directional_bspline: bool = True,  # 是否使用新的方向 × log-depth B-spline
        n_alpha: int = 12,
        n_depth: int = 10,
        alpha_method: Literal["asin", "y_coord"] = "asin",
        max_delta_log: float = 0.5,
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
        self.use_directional_bspline = use_directional_bspline
        
        if use_directional_bspline:
            # 新版本：使用方向 × log-depth B-spline grid
            # 不需要全局 spline，直接用 B-spline grid 修正
            self.spline = None
            self.scale_module = None
            
            # 创建方向 × log-depth B-spline grid
            self.directional_bspline = DirectionalBSplineGrid(
                n_alpha=n_alpha,
                n_depth=n_depth,
                alpha_min=-np.pi / 2.0,
                alpha_max=np.pi / 2.0,
                log_depth_min=log_depth_min,
                log_depth_max=log_depth_max,
                spline_order=3,
                alpha_method=alpha_method,
                max_delta_log=max_delta_log,
            )
            
            # 预计算方向向量（用于 B-spline grid）
            u_grid, v_grid = image_uv_grid(height, width)
            ray_dirs = pixel_to_directions_torch(
                torch.from_numpy(u_grid),
                torch.from_numpy(v_grid)
            )  # (H, W, 3)
            self.register_buffer('ray_dirs', ray_dirs)
        else:
            # 旧版本：全局 spline + 方向缩放
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
            
            self.directional_bspline = None
            
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
        
        if self.use_directional_bspline:
            # 新版本：方向 × log-depth B-spline grid
            # d' = exp(log(d) + Δ(alpha, log(d)))
            
            # 扩展方向向量到 batch 维度
            ray_dirs_batch = self.ray_dirs.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)
            
            # 计算修正项
            delta = self.directional_bspline(
                ray_dirs=ray_dirs_batch,  # (B, H, W, 3)
                log_depth=log_depth_dap,  # (B, H, W)
            )  # (B, H, W)
            
            # 应用修正
            log_depth_transformed = log_depth_dap + delta  # (B, H, W)
            depth_transformed = torch.exp(log_depth_transformed)  # (B, H, W)
        else:
            # 旧版本：全局 spline + 方向缩放
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
        if self.spline is not None:
            self.spline.ensure_monotonicity()
    
    def get_directional_bspline(self):
        """获取方向 B-spline grid 模块（用于损失函数）"""
        return self.directional_bspline
