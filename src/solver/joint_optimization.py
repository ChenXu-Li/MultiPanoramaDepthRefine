"""
最终联合优化：组合 Step 1 + Step 2 + Step 3
"""
import numpy as np
import torch
from typing import List, Optional, Dict, Tuple
import pycolmap
from pathlib import Path
import time

from .energy_function import TotalEnergyFunction
from .optimizer import MultiViewOptimizer, check_convergence
from ..deformation import DepthReparameterization
from ..utils.io import save_depth_npy

def log_time(msg: str):
    """打印带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_item(value):
    """安全地获取 tensor 或 float 的值"""
    if isinstance(value, torch.Tensor):
        return value.item()
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return 0.0


class JointOptimizationConfig:
    """联合优化配置"""
    def __init__(
        self,
        # 几何一致性权重
        lambda_p2r: float = 1.0,
        lambda_depth: float = 0.1,
        # 正则化权重
        lambda_prior: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_scale: float = 0.01,
        # B-spline 约束权重（新版本）
        lambda_mono: float = 0.1,
        lambda_smooth_bspline: float = 0.001,
        lambda_far: float = 0.1,
        # 优化器配置
        optimizer: str = "adam",
        lr: float = 1e-3,
        max_iter: int = 1000,
        early_stop_threshold: float = 1e-6,
        # 其他参数
        far_threshold: float = 100.0,
        use_robust_p2r: bool = True,
        huber_delta_p2r: float = 0.1,
        prior_loss_type: str = "l2",
        prior_huber_delta: float = 0.1,
        smooth_type: str = "l2",
        edge_aware: bool = False,
        rgb_sigma: float = 10.0,
        device: str = "cpu",
        save_history: bool = False,
        save_interval: int = 100,
        print_interval: int = 10,
    ):
        self.lambda_p2r = lambda_p2r
        self.lambda_depth = lambda_depth
        self.lambda_prior = lambda_prior
        self.lambda_smooth = lambda_smooth
        self.lambda_scale = lambda_scale
        self.lambda_mono = lambda_mono
        self.lambda_smooth_bspline = lambda_smooth_bspline
        self.lambda_far = lambda_far
        self.optimizer = optimizer
        self.lr = lr
        self.max_iter = max_iter
        self.early_stop_threshold = early_stop_threshold
        self.far_threshold = far_threshold
        self.use_robust_p2r = use_robust_p2r
        self.huber_delta_p2r = huber_delta_p2r
        self.prior_loss_type = prior_loss_type
        self.prior_huber_delta = prior_huber_delta
        self.smooth_type = smooth_type
        self.edge_aware = edge_aware
        self.rgb_sigma = rgb_sigma
        self.device = device
        self.save_history = save_history
        self.save_interval = save_interval
        self.print_interval = print_interval


def optimize_multi_view_depth(
    depth_reparam_modules: List[DepthReparameterization],
    log_depth_daps: List[torch.Tensor],
    depth_daps: List[torch.Tensor],
    cam_from_world_list: List[pycolmap.Rigid3d],
    config: JointOptimizationConfig,
    rgbs: Optional[List[torch.Tensor]] = None,
    masks: Optional[List[torch.Tensor]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[List[np.ndarray], Dict]:
    """
    多视角联合优化
    
    Args:
        depth_reparam_modules: List[DepthReparameterization] 各视角的深度重参数化模块
        log_depth_daps: List[(H, W)] 各视角的 DAP log-depth
        depth_daps: List[(H, W)] 各视角的 DAP 深度
        cam_from_world_list: List[pycolmap.Rigid3d] 各视角的相机变换
        config: JointOptimizationConfig 配置
        rgbs: List[(H, W, 3)] 可选，各视角的 RGB 图像
        masks: List[(H, W)] 可选，各视角的有效像素 mask
        output_dir: 可选，输出目录（用于保存中间结果）
        
    Returns:
        depths_opt: List[(H, W)] 优化后的深度图
        report: dict 优化报告
    """
    num_views = len(depth_reparam_modules)
    
    # 创建总能量函数
    log_time(f"  创建总能量函数...")
    t0 = time.time()
    energy_fn = TotalEnergyFunction(
        lambda_p2r=config.lambda_p2r,
        lambda_depth=config.lambda_depth,
        lambda_prior=config.lambda_prior,
        lambda_smooth=config.lambda_smooth,
        lambda_scale=config.lambda_scale,
        lambda_mono=config.lambda_mono,
        lambda_smooth_bspline=config.lambda_smooth_bspline,
        lambda_far=config.lambda_far,
        far_threshold=config.far_threshold,
        use_robust_p2r=config.use_robust_p2r,
        huber_delta_p2r=config.huber_delta_p2r,
        prior_loss_type=config.prior_loss_type,
        prior_huber_delta=config.prior_huber_delta,
        smooth_type=config.smooth_type,
        edge_aware=config.edge_aware,
        rgb_sigma=config.rgb_sigma,
        device=config.device,
    )
    log_time(f"  ✅ 总能量函数创建完成 ({time.time() - t0:.2f}s)")
    
    # 创建优化器
    log_time(f"  创建优化器...")
    t0 = time.time()
    optimizer = MultiViewOptimizer(
        depth_reparam_modules=depth_reparam_modules,
        optimizer_type=config.optimizer,
        lr=config.lr,
        device=config.device,
    )
    log_time(f"  ✅ 优化器创建完成 ({time.time() - t0:.2f}s)")
    
    # 历史记录（总是记录，用于分析权重）
    history = {
        'total': [],
        'geometry_p2r': [],  # 原始 P2R 损失
        'geometry_depth': [],  # 原始深度一致性损失
        'regularization_prior': [],  # 原始先验损失
        'regularization_smooth': [],  # 原始平滑损失
        'regularization_scale': [],  # 原始缩放约束损失（旧版本）
        'bspline_mono': [],  # B-spline 单调性约束损失
        'bspline_smooth': [],  # B-spline 方向平滑正则损失
        'bspline_far': [],  # B-spline 远景渐近约束损失
        'weighted_p2r': [],  # 加权 P2R 损失 (lambda_p2r * p2r)
        'weighted_depth': [],  # 加权深度损失 (lambda_depth * depth)
        'weighted_prior': [],  # 加权先验损失 (lambda_prior * prior)
        'weighted_smooth': [],  # 加权平滑损失 (lambda_smooth * smooth)
        'weighted_scale': [],  # 加权缩放损失 (lambda_scale * scale)
        'weighted_bspline_mono': [],  # 加权 B-spline 单调性损失
        'weighted_bspline_smooth': [],  # 加权 B-spline 平滑损失
        'weighted_bspline_far': [],  # 加权 B-spline 远景损失
    }
    
    prev_energy = float('inf')
    
    log_time(f"开始联合优化，最大迭代次数: {config.max_iter}")
    log_time(f"  设备: {config.device}")
    log_time(f"  数据形状: depth={depth_daps[0].shape}, log_depth={log_depth_daps[0].shape}")
    
    for step in range(config.max_iter):
        iter_start_time = time.time()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 计算总能量
        try:
            loss_dict = energy_fn.compute_total_energy(
                depth_reparam_modules=depth_reparam_modules,
                log_depth_daps=log_depth_daps,
                depth_daps=depth_daps,
                cam_from_world_list=cam_from_world_list,
                rgbs=rgbs,
                masks=masks,
            )
        except Exception as e:
            log_time(f"    ❌ 能量计算失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        total_loss = loss_dict['total']
        
        # 反向传播
        try:
            total_loss.backward()
        except Exception as e:
            log_time(f"    ❌ 反向传播失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 梯度裁剪（防止梯度爆炸）
        try:
            torch.nn.utils.clip_grad_norm_(
                [p for module in depth_reparam_modules for p in module.parameters()],
                max_norm=1.0
            )
        except Exception as e:
            log_time(f"    ❌ 梯度裁剪失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 优化步骤
        try:
            optimizer.step()
        except Exception as e:
            log_time(f"    ❌ 优化步骤失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 记录历史（总是记录，用于分析权重）
        history['total'].append(safe_item(total_loss))
        history['geometry_p2r'].append(safe_item(loss_dict['geometry']['p2r']))
        history['geometry_depth'].append(safe_item(loss_dict['geometry']['depth']))
        history['regularization_prior'].append(safe_item(loss_dict['regularization']['prior']))
        history['regularization_smooth'].append(safe_item(loss_dict['regularization']['smooth']))
        history['regularization_scale'].append(safe_item(loss_dict['regularization']['scale']))
        
        # 记录 B-spline 约束损失（新版本）
        bspline_constraints = loss_dict.get('bspline_constraints', {})
        history['bspline_mono'].append(safe_item(bspline_constraints.get('monotonicity', torch.tensor(0.0))))
        history['bspline_smooth'].append(safe_item(bspline_constraints.get('smoothness', torch.tensor(0.0))))
        history['bspline_far'].append(safe_item(bspline_constraints.get('far_field', torch.tensor(0.0))))
        
        # 记录加权损失（用于分析权重影响）
        p2r_val = safe_item(loss_dict['geometry']['p2r'])
        depth_val = safe_item(loss_dict['geometry']['depth'])
        prior_val = safe_item(loss_dict['regularization']['prior'])
        smooth_val = safe_item(loss_dict['regularization']['smooth'])
        scale_val = safe_item(loss_dict['regularization']['scale'])
        bspline_mono_val = safe_item(bspline_constraints.get('monotonicity', torch.tensor(0.0)))
        bspline_smooth_val = safe_item(bspline_constraints.get('smoothness', torch.tensor(0.0)))
        bspline_far_val = safe_item(bspline_constraints.get('far_field', torch.tensor(0.0)))
        
        history['weighted_p2r'].append(config.lambda_p2r * p2r_val)
        history['weighted_depth'].append(config.lambda_depth * depth_val)
        history['weighted_prior'].append(config.lambda_prior * prior_val)
        history['weighted_smooth'].append(config.lambda_smooth * smooth_val)
        history['weighted_scale'].append(config.lambda_scale * scale_val)
        history['weighted_bspline_mono'].append(config.lambda_mono * bspline_mono_val)
        history['weighted_bspline_smooth'].append(config.lambda_smooth_bspline * bspline_smooth_val)
        history['weighted_bspline_far'].append(config.lambda_far * bspline_far_val)
        
        # 打印进度
        iter_time = time.time() - iter_start_time
        if (step + 1) % config.print_interval == 0 or step == 0:
            bspline_total = safe_item(bspline_constraints.get('total', torch.tensor(0.0)))
            log_msg = (
                f"  Iter {step+1}/{config.max_iter}: "
                f"total={safe_item(total_loss):.6f}, "
                f"p2r={safe_item(loss_dict['geometry']['p2r']):.6f}, "
                f"prior={safe_item(loss_dict['regularization']['prior']):.6f}"
            )
            if bspline_total > 0:
                log_msg += f", bspline={bspline_total:.6f}"
            log_msg += f" ({iter_time:.2f}s)"
            log_time(log_msg)
        
        # 保存中间结果
        if output_dir is not None and (step + 1) % config.save_interval == 0:
            depths_current = []
            for i, module in enumerate(depth_reparam_modules):
                log_depth_dap = log_depth_daps[i]
                depth_transformed = module(log_depth_dap)
                depths_current.append(depth_transformed.detach().cpu().numpy())
                
                save_path = output_dir / f"view_{i}_iter_{step+1}.npy"
                save_depth_npy(depths_current[-1], save_path)
        
        # 收敛检查
        current_energy = safe_item(total_loss)
        if check_convergence(current_energy, prev_energy, config.early_stop_threshold):
            print(f"  收敛于迭代 {step+1}")
            break
        
        prev_energy = current_energy
    
    # 获取最终深度
    depths_opt = []
    for i, module in enumerate(depth_reparam_modules):
        log_depth_dap = log_depth_daps[i]
        depth_transformed = module(log_depth_dap)
        depths_opt.append(depth_transformed.detach().cpu().numpy())
    
    report = {
        'iterations': step + 1,
        'final_energy': prev_energy,
        'history': history,  # 总是返回历史记录
        'config': {
            'lambda_p2r': config.lambda_p2r,
            'lambda_depth': config.lambda_depth,
            'lambda_prior': config.lambda_prior,
            'lambda_smooth': config.lambda_smooth,
            'lambda_scale': config.lambda_scale,
            'lr': config.lr,
            'max_iter': config.max_iter,
            'early_stop_threshold': config.early_stop_threshold,
        }
    }
    
    return depths_opt, report
