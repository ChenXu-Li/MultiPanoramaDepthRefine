"""
方向 × log-depth B-spline grid 表示
实现 Δ(alpha, log_depth) = BSplineGrid(alpha, log_depth)

关键设计：
- 方向变量：alpha = asin(ray_dir.y) 或 ray_dir.y（不是完整球面）
- Grid 维度：axis 0 = alpha, axis 1 = log_depth
- 使用 cubic B-spline 插值（order=3）
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Tuple


def bspline_basis_1d(u: torch.Tensor, order: int = 3) -> torch.Tensor:
    """
    计算 1D B-spline 基函数值（cubic, order=3）
    
    Args:
        u: (N,) 归一化参数 [0, 1]，表示在 knot 区间内的位置
        order: B-spline 阶数（默认 3，cubic）
        
    Returns:
        basis: (N, order+1) 基函数值，每列对应一个控制点的影响
    """
    N = u.shape[0]
    basis = torch.zeros(N, order + 1, device=u.device, dtype=u.dtype)
    
    # 对于 cubic B-spline (order=3)，需要 4 个基函数
    if order == 3:
        # Cubic B-spline 基函数（Cox-de Boor 递归公式）
        # B_0(u) = (1-u)^3 / 6
        # B_1(u) = (3u^3 - 6u^2 + 4) / 6
        # B_2(u) = (-3u^3 + 3u^2 + 3u + 1) / 6
        # B_3(u) = u^3 / 6
        
        u2 = u * u
        u3 = u2 * u
        one_minus_u = 1.0 - u
        one_minus_u2 = one_minus_u * one_minus_u
        one_minus_u3 = one_minus_u2 * one_minus_u
        
        basis[:, 0] = one_minus_u3 / 6.0
        basis[:, 1] = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0
        basis[:, 2] = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0
        basis[:, 3] = u3 / 6.0
    else:
        # 通用递归实现（较慢，但支持任意阶数）
        # 初始化：B_0^0 = 1
        basis[:, 0] = 1.0
        
        for k in range(1, order + 1):
            for i in range(k + 1):
                if i == 0:
                    basis[:, i] = (1.0 - u) * basis[:, i]
                elif i == k:
                    basis[:, i] = u * basis[:, i - 1]
                else:
                    basis[:, i] = (u * basis[:, i - 1] + (1.0 - u) * basis[:, i])
    
    return basis


def bspline_interp_2d(
    control_points: torch.Tensor,
    alpha: torch.Tensor,
    log_depth: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    log_depth_min: float,
    log_depth_max: float,
    order: int = 3,
) -> torch.Tensor:
    """
    2D B-spline 插值（使用张量积）
    
    Args:
        control_points: (N_alpha, N_depth) 控制点网格
        alpha: (...,) 方向变量值
        log_depth: (...,) log-depth 值
        alpha_min: alpha 的最小值
        alpha_max: alpha 的最大值
        log_depth_min: log-depth 的最小值
        log_depth_max: log-depth 的最大值
        order: B-spline 阶数（默认 3，cubic）
        
    Returns:
        delta: (...,) 插值得到的修正值
    """
    # 归一化到 [0, 1] 区间（用于 B-spline 插值）
    alpha_range = alpha_max - alpha_min
    depth_range = log_depth_max - log_depth_min
    
    alpha_norm = (alpha - alpha_min) / (alpha_range + 1e-8)
    log_depth_norm = (log_depth - log_depth_min) / (depth_range + 1e-8)
    
    # 裁剪到有效范围
    alpha_norm = torch.clamp(alpha_norm, 0.0, 1.0)
    log_depth_norm = torch.clamp(log_depth_norm, 0.0, 1.0)
    
    N_alpha, N_depth = control_points.shape
    
    # 转换为 grid 索引（考虑 B-spline 的 local support）
    # 对于 cubic B-spline (order=3)，每个查询点影响 4×4 个控制点
    # 索引范围：[0, N-1]，但需要访问 [idx-1, idx+2] 共 4 个点
    alpha_idx = alpha_norm * (N_alpha - 1)
    depth_idx = log_depth_norm * (N_depth - 1)
    
    # 计算在 knot 区间内的局部坐标 [0, 1]
    alpha_local = alpha_idx - torch.floor(alpha_idx)
    depth_local = depth_idx - torch.floor(depth_idx)
    
    # 展平所有维度以便批量处理
    original_shape = alpha.shape
    num_elements = alpha.numel()
    
    alpha_flat = alpha_norm.flatten()  # (N,)
    log_depth_flat = log_depth_norm.flatten()  # (N,)
    
    alpha_idx_flat = alpha_flat * (N_alpha - 1)
    depth_idx_flat = log_depth_flat * (N_depth - 1)
    
    # 计算在 knot 区间内的局部坐标 [0, 1]
    alpha_local_flat = alpha_idx_flat - torch.floor(alpha_idx_flat)
    depth_local_flat = depth_idx_flat - torch.floor(depth_idx_flat)
    
    # 获取起始索引（考虑边界，确保访问范围在 [0, N-1]）
    alpha_start_flat = torch.clamp(torch.floor(alpha_idx_flat).long() - 1, 0, N_alpha - 4)
    depth_start_flat = torch.clamp(torch.floor(depth_idx_flat).long() - 1, 0, N_depth - 4)
    
    # 计算 B-spline 基函数（使用展平的 1D tensor）
    alpha_basis_flat = bspline_basis_1d(alpha_local_flat, order=order)  # (N, 4)
    depth_basis_flat = bspline_basis_1d(depth_local_flat, order=order)  # (N, 4)
    
    # 批量计算插值（2D 张量积）
    result_flat = torch.zeros(num_elements, device=alpha.device, dtype=alpha.dtype)
    
    for i in range(4):
        for j in range(4):
            # 获取控制点索引（考虑边界）
            alpha_idx_control = torch.clamp(alpha_start_flat + i, 0, N_alpha - 1)
            depth_idx_control = torch.clamp(depth_start_flat + j, 0, N_depth - 1)
            
            # 获取控制点值（使用 advanced indexing）
            control_vals = control_points[alpha_idx_control, depth_idx_control]  # (N,)
            
            # 加权求和
            weight = alpha_basis_flat[:, i] * depth_basis_flat[:, j]  # (N,)
            result_flat = result_flat + control_vals * weight
    
    # 恢复原始形状
    result = result_flat.reshape(original_shape)
    
    return result


def direction_to_alpha(ray_dir: torch.Tensor, method: Literal["asin", "y_coord"] = "asin") -> torch.Tensor:
    """
    将方向向量转换为方向变量 alpha
    
    Args:
        ray_dir: (..., 3) 单位方向向量
        method: 'asin' 使用 asin(y)，'y_coord' 直接使用 y 坐标
        
    Returns:
        alpha: (...,) 方向变量
    """
    if method == "asin":
        # alpha = asin(ray_dir.y) ∈ [-π/2, π/2]
        alpha = torch.asin(torch.clamp(ray_dir[..., 1], -1.0, 1.0))
    elif method == "y_coord":
        # alpha = ray_dir.y ∈ [-1, 1]
        alpha = ray_dir[..., 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return alpha


class DirectionalBSplineGrid(nn.Module):
    """
    方向 × log-depth B-spline grid 表示
    
    实现：
        Δ(alpha, log_depth) = BSplineGrid(alpha, log_depth)
        
    最终深度：
        d' = exp(log(d) + Δ(alpha, log(d)))
    """
    
    def __init__(
        self,
        n_alpha: int = 12,
        n_depth: int = 10,
        alpha_min: float = -np.pi / 2.0,
        alpha_max: float = np.pi / 2.0,
        log_depth_min: float = -3.0,
        log_depth_max: float = 5.0,
        spline_order: int = 3,
        alpha_method: Literal["asin", "y_coord"] = "asin",
        max_delta_log: float = 0.5,
    ):
        """
        Args:
            n_alpha: alpha 方向的控制点数量（推荐 8-16）
            n_depth: log-depth 方向的控制点数量（推荐 8-12）
            alpha_min: alpha 的最小值
            alpha_max: alpha 的最大值
            log_depth_min: log-depth 的最小值
            log_depth_max: log-depth 的最大值
            spline_order: B-spline 阶数（默认 3，cubic）
            alpha_method: 方向变量计算方法（'asin' 或 'y_coord'）
            max_delta_log: delta 的最大值（log 空间），用于限制修正幅度
        """
        super().__init__()
        
        self.n_alpha = n_alpha
        self.n_depth = n_depth
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.log_depth_min = log_depth_min
        self.log_depth_max = log_depth_max
        self.spline_order = spline_order
        self.alpha_method = alpha_method
        self.max_delta_log = max_delta_log
        
        # 控制点网格（初始化为 0，对应 identity mapping）
        self.control_points = nn.Parameter(torch.zeros(n_alpha, n_depth))
    
    def forward(
        self,
        ray_dirs: torch.Tensor,
        log_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算深度修正项
        
        Args:
            ray_dirs: (..., 3) 单位方向向量
            log_depth: (...,) log-depth 值
            
        Returns:
            delta: (...,) 深度修正项（log 空间）
        """
        # 1. 计算方向变量 alpha
        alpha = direction_to_alpha(ray_dirs, method=self.alpha_method)  # (...)
        
        # 2. B-spline 插值
        delta = bspline_interp_2d(
            control_points=self.control_points,
            alpha=alpha,
            log_depth=log_depth,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            log_depth_min=self.log_depth_min,
            log_depth_max=self.log_depth_max,
            order=self.spline_order,
        )
        
        # 3. 限制修正幅度
        delta = torch.clamp(delta, -self.max_delta_log, self.max_delta_log)
        
        return delta
    
    def get_control_points(self) -> torch.Tensor:
        """获取控制点网格（用于损失函数）"""
        return self.control_points
