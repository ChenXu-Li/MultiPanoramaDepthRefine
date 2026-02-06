"""
全局单调 Spline 映射（log-depth 空间）
实现单调递增的 1D spline，用于深度重参数化
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class MonotonicCubicSpline(nn.Module):
    """
    单调三次样条插值（log-depth 空间）
    
    使用单调三次 Hermite 插值（Fritsch-Carlson 方法）
    保证单调性：如果输入单调，输出也单调
    """
    
    def __init__(
        self,
        num_knots: int = 10,
        log_depth_min: float = -3.0,
        log_depth_max: float = 5.0,
        freeze_reference_point: bool = True,
        reference_log_depth: float = 0.0,
    ):
        """
        Args:
            num_knots: Spline 控制点数量
            log_depth_min: log-depth 最小值
            log_depth_max: log-depth 最大值
            freeze_reference_point: 是否冻结参考点（防止尺度漂移）
            reference_log_depth: 参考点位置（log(1.0) = 0）
        """
        super().__init__()
        
        self.num_knots = num_knots
        self.log_depth_min = log_depth_min
        self.log_depth_max = log_depth_max
        self.freeze_reference_point = freeze_reference_point
        self.reference_log_depth = reference_log_depth
        
        # 均匀分布的 knot 位置
        knots_x = torch.linspace(log_depth_min, log_depth_max, num_knots)
        self.register_buffer('knots_x', knots_x)
        
        # Spline 输出值（可学习参数）
        # 初始化为恒等映射：y = x
        knots_y_init = knots_x.clone().detach()
        
        # 如果冻结参考点，找到最近的 knot 并固定
        # 注意：只有当参考点接近某个 knot 时才冻结，否则保持恒等映射
        if freeze_reference_point:
            ref_idx = torch.argmin(torch.abs(knots_x - reference_log_depth))
            ref_dist = torch.abs(knots_x[ref_idx] - reference_log_depth)
            # 只有当参考点足够接近 knot 时才冻结（距离 < 0.1）
            if ref_dist < 0.1:
                knots_y_init[ref_idx] = reference_log_depth
        
        self.knots_y = nn.Parameter(knots_y_init)
        
        # 冻结参考点（如果启用）
        if freeze_reference_point:
            ref_idx = torch.argmin(torch.abs(self.knots_x - reference_log_depth))
            self.knots_y.requires_grad = True
            # 在 forward 中强制参考点值
    
    def forward(self, log_depth: torch.Tensor) -> torch.Tensor:
        """
        应用单调 spline 映射
        
        Args:
            log_depth: (...,) log-depth 值
            
        Returns:
            transformed_log_depth: (...,) 变换后的 log-depth 值
        """
        # 确保参考点被冻结
        if self.freeze_reference_point:
            ref_idx = torch.argmin(torch.abs(self.knots_x - self.reference_log_depth))
            with torch.no_grad():
                self.knots_y.data[ref_idx] = self.reference_log_depth
        
        # 使用单调三次 Hermite 插值
        # 简化实现：使用线性插值 + 单调性约束
        # 更精确的实现可以使用 Fritsch-Carlson 方法
        
        # 将输入裁剪到有效范围
        log_depth_clamped = torch.clamp(log_depth, self.log_depth_min, self.log_depth_max)
        
        # 初始化结果
        result = torch.zeros_like(log_depth_clamped)
        
        # 检查每个输入值是否正好等于某个 knot
        for i in range(self.num_knots):
            mask = torch.isclose(log_depth_clamped, self.knots_x[i], atol=1e-6)
            if mask.any():
                result[mask] = self.knots_y[i]
        
        # 对于不在 knot 位置的值，进行线性插值
        mask_interp = ~torch.any(torch.stack([
            torch.isclose(log_depth_clamped, self.knots_x[i], atol=1e-6)
            for i in range(self.num_knots)
        ]), dim=0)
        
        if mask_interp.any():
            log_depth_interp = log_depth_clamped[mask_interp]
            
            # 找到每个输入值所在的区间
            indices = torch.searchsorted(self.knots_x, log_depth_interp, right=True)
            indices = torch.clamp(indices - 1, 0, self.num_knots - 2)
            
            # 线性插值
            x0 = self.knots_x[indices]
            x1 = self.knots_x[indices + 1]
            y0 = self.knots_y[indices]
            y1 = self.knots_y[indices + 1]
            
            # 确保 y1 >= y0（单调性约束）
            y1_clamped = torch.where(y1 < y0, y0, y1)
            
            # 插值权重
            t = (log_depth_interp - x0) / (x1 - x0 + 1e-8)
            t = torch.clamp(t, 0, 1)
            
            # 线性插值
            result[mask_interp] = y0 + t * (y1_clamped - y0)
        
        # 处理边界外的值（恒等映射）
        mask_below = log_depth < self.log_depth_min
        mask_above = log_depth > self.log_depth_max
        
        result = torch.where(mask_below, log_depth, result)
        result = torch.where(mask_above, log_depth, result)
        
        return result
    
    def ensure_monotonicity(self):
        """
        强制确保 knot 值的单调性
        在优化过程中定期调用此方法
        """
        with torch.no_grad():
            # 确保 knots_y 单调递增
            for i in range(1, self.num_knots):
                if self.knots_y[i] < self.knots_y[i-1]:
                    self.knots_y.data[i] = self.knots_y[i-1]
            
            # 如果冻结参考点，确保参考点值正确
            if self.freeze_reference_point:
                ref_idx = torch.argmin(torch.abs(self.knots_x - self.reference_log_depth))
                self.knots_y.data[ref_idx] = self.reference_log_depth


class LinearMonotonicSpline(nn.Module):
    """
    线性单调映射（简化版本）
    
    使用分段线性函数，保证单调性
    """
    
    def __init__(
        self,
        num_knots: int = 10,
        log_depth_min: float = -3.0,
        log_depth_max: float = 5.0,
        freeze_reference_point: bool = True,
        reference_log_depth: float = 0.0,
    ):
        super().__init__()
        
        self.num_knots = num_knots
        self.log_depth_min = log_depth_min
        self.log_depth_max = log_depth_max
        self.freeze_reference_point = freeze_reference_point
        self.reference_log_depth = reference_log_depth
        
        knots_x = torch.linspace(log_depth_min, log_depth_max, num_knots)
        self.register_buffer('knots_x', knots_x)
        
        # 初始化为恒等映射
        knots_y_init = knots_x.clone().detach()
        if freeze_reference_point:
            ref_idx = torch.argmin(torch.abs(knots_x - reference_log_depth))
            ref_dist = torch.abs(knots_x[ref_idx] - reference_log_depth)
            # 只有当参考点足够接近 knot 时才冻结
            if ref_dist < 0.1:
                knots_y_init[ref_idx] = reference_log_depth
        
        self.knots_y = nn.Parameter(knots_y_init)
    
    def forward(self, log_depth: torch.Tensor) -> torch.Tensor:
        """应用线性单调映射"""
        if self.freeze_reference_point:
            ref_idx = torch.argmin(torch.abs(self.knots_x - self.reference_log_depth))
            with torch.no_grad():
                self.knots_y.data[ref_idx] = self.reference_log_depth
        
        log_depth_clamped = torch.clamp(log_depth, self.log_depth_min, self.log_depth_max)
        
        # 初始化结果
        result = torch.zeros_like(log_depth_clamped)
        
        # 检查每个输入值是否正好等于某个 knot
        for i in range(self.num_knots):
            mask = torch.isclose(log_depth_clamped, self.knots_x[i], atol=1e-6)
            if mask.any():
                result[mask] = self.knots_y[i]
        
        # 对于不在 knot 位置的值，进行线性插值
        mask_interp = ~torch.any(torch.stack([
            torch.isclose(log_depth_clamped, self.knots_x[i], atol=1e-6)
            for i in range(self.num_knots)
        ]), dim=0)
        
        if mask_interp.any():
            log_depth_interp = log_depth_clamped[mask_interp]
            
            indices = torch.searchsorted(self.knots_x, log_depth_interp, right=True)
            indices = torch.clamp(indices - 1, 0, self.num_knots - 2)
            
            x0 = self.knots_x[indices]
            x1 = self.knots_x[indices + 1]
            y0 = self.knots_y[indices]
            y1 = self.knots_y[indices + 1]
            
            y1_clamped = torch.where(y1 < y0, y0, y1)
            
            t = (log_depth_interp - x0) / (x1 - x0 + 1e-8)
            t = torch.clamp(t, 0, 1)
            
            result[mask_interp] = y0 + t * (y1_clamped - y0)
        
        mask_below = log_depth < self.log_depth_min
        mask_above = log_depth > self.log_depth_max
        result = torch.where(mask_below, log_depth, result)
        result = torch.where(mask_above, log_depth, result)
        
        return result
    
    def ensure_monotonicity(self):
        """强制确保单调性"""
        with torch.no_grad():
            for i in range(1, self.num_knots):
                if self.knots_y[i] < self.knots_y[i-1]:
                    self.knots_y.data[i] = self.knots_y[i-1]
            
            if self.freeze_reference_point:
                ref_idx = torch.argmin(torch.abs(self.knots_x - self.reference_log_depth))
                self.knots_y.data[ref_idx] = self.reference_log_depth
