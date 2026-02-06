"""
优化器封装：PyTorch 优化器 + 收敛检测
"""
import torch
import torch.optim
from typing import Literal, Optional, Dict, List
import math

from ..deformation import DepthReparameterization


class MultiViewOptimizer:
    """
    多视角联合优化器
    
    优化变量：
    - 各视角的深度重参数化模块参数（spline + 方向缩放）
    """
    
    def __init__(
        self,
        depth_reparam_modules: List[DepthReparameterization],
        optimizer_type: Literal["adam", "sgd"] = "adam",
        lr: float = 1e-3,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        sgd_momentum: float = 0.9,
        device: str = "cpu",
    ):
        """
        Args:
            depth_reparam_modules: List[DepthReparameterization] 各视角的深度重参数化模块
            optimizer_type: 优化器类型
            lr: 学习率
            adam_beta1: Adam beta1 参数
            adam_beta2: Adam beta2 参数
            adam_eps: Adam epsilon 参数
            sgd_momentum: SGD 动量
            device: 计算设备
        """
        self.depth_reparam_modules = depth_reparam_modules
        self.device = device
        
        # 收集所有可学习参数
        params = []
        for module in depth_reparam_modules:
            # Spline 参数（knots_y 是 Parameter）
            if hasattr(module.spline, 'knots_y'):
                params.append(module.spline.knots_y)
            # 方向缩放参数
            if hasattr(module.scale_module, 'coeffs'):
                if isinstance(module.scale_module.coeffs, torch.nn.Parameter):
                    params.append(module.scale_module.coeffs)
            elif hasattr(module.scale_module, 'grid'):
                if isinstance(module.scale_module.grid, torch.nn.Parameter):
                    params.append(module.scale_module.grid)
        
        # 如果没有找到参数，尝试使用 parameters() 方法
        if len(params) == 0:
            for module in depth_reparam_modules:
                params.extend(list(module.parameters()))
        
        # 创建优化器
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=lr,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=sgd_momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def step(self):
        """执行一步优化"""
        self.optimizer.step()
        
        # 确保单调性和约束
        for module in self.depth_reparam_modules:
            module.ensure_monotonicity()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def check_convergence(
    current_energy: float,
    previous_energy: float,
    threshold: float = 1e-6,
) -> bool:
    """
    检查是否收敛
    
    Args:
        current_energy: 当前能量值
        previous_energy: 前一次能量值
        threshold: 收敛阈值（相对变化率）
        
    Returns:
        converged: 是否收敛
    """
    if previous_energy == 0.0:
        return False
    
    rel_change = abs(current_energy - previous_energy) / (abs(previous_energy) + 1e-8)
    return rel_change < threshold
