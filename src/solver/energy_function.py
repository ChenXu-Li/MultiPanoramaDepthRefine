"""
总能量函数：组合 Step 1 + Step 2 + Step 3
L = λ_p2r * L_p2r + λ_depth * L_depth + λ_prior * L_prior + λ_smooth * L_smooth + λ_g * L_g
"""
import torch
from typing import List, Optional, Dict
import pycolmap
import time

from ..geometry import MultiViewGeometryLoss
from ..regularization import RegularizationLoss
from ..deformation import DepthReparameterization

def log_time(msg: str):
    """打印带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TotalEnergyFunction:
    """
    总能量函数
    
    组合：
    1. 多视角几何一致性（Step 2）
    2. 防退化护栏（Step 3）
    
    注意：Step 1 的深度重参数化模块作为优化变量的一部分
    """
    
    def __init__(
        self,
        # 几何一致性权重
        lambda_p2r: float = 1.0,
        lambda_depth: float = 0.1,
        # 正则化权重
        lambda_prior: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_scale: float = 0.01,
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
    ):
        """
        Args:
            lambda_p2r: Point-to-Ray 损失权重
            lambda_depth: 深度一致性损失权重
            lambda_prior: 先验锚点权重（必须 > 0）
            lambda_smooth: 平滑正则权重
            lambda_scale: 方向变形约束权重
            far_threshold: 远景阈值（米）
            use_robust_p2r: 是否对 Point-to-Ray 使用 robust loss
            huber_delta_p2r: Point-to-Ray Huber loss 阈值
            prior_loss_type: 先验损失类型（'l2' 或 'huber'）
            prior_huber_delta: 先验 Huber loss 阈值（仅当 prior_loss_type='huber'）
            smooth_type: 平滑类型（'l2' 或 'l1'）
            edge_aware: 是否使用边缘感知平滑
            rgb_sigma: RGB 边缘敏感度
            device: 计算设备
        """
        self.device = device
        
        # 创建几何一致性损失函数
        self.geometry_loss_fn = MultiViewGeometryLoss(
            lambda_p2r=lambda_p2r,
            lambda_depth=lambda_depth,
            far_threshold=far_threshold,
            use_robust_p2r=use_robust_p2r,
            huber_delta_p2r=huber_delta_p2r,
            use_robust_depth=False,
            device=device,
        )
        
        # 创建正则化损失函数
        self.regularization_loss_fn = RegularizationLoss(
            lambda_prior=lambda_prior,
            lambda_smooth=lambda_smooth,
            lambda_scale=lambda_scale,
            far_threshold=far_threshold,
            prior_loss_type=prior_loss_type,
            prior_huber_delta=prior_huber_delta,
            smooth_type=smooth_type,
            edge_aware=edge_aware,
            rgb_sigma=rgb_sigma,
        )
    
    def compute_total_energy(
        self,
        depth_reparam_modules: List[DepthReparameterization],
        log_depth_daps: List[torch.Tensor],
        depth_daps: List[torch.Tensor],
        cam_from_world_list: List[pycolmap.Rigid3d],
        rgbs: Optional[List[torch.Tensor]] = None,
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总能量
        
        Args:
            depth_reparam_modules: List[DepthReparameterization] 各视角的深度重参数化模块
            log_depth_daps: List[(H, W)] 各视角的 DAP log-depth
            depth_daps: List[(H, W)] 各视角的 DAP 深度（用于远景 mask）
            cam_from_world_list: List[pycolmap.Rigid3d] 各视角的相机变换
            rgbs: List[(H, W, 3)] 可选，各视角的 RGB 图像
            masks: List[(H, W)] 可选，各视角的有效像素 mask
            
        Returns:
            loss_dict: 包含各项损失的字典
                - 'total': 总损失
                - 'geometry': 几何一致性损失
                    - 'p2r': Point-to-Ray 损失
                    - 'depth': 深度一致性损失
                - 'regularization': 正则化损失
                    - 'prior': 先验锚点损失
                    - 'smooth': 平滑正则损失
                    - 'scale': 方向变形约束损失
        """
        num_views = len(depth_reparam_modules)
        
        # 从深度重参数化模块获取优化后的深度
        t0 = time.time()
        depths = []
        log_depths = []
        scale_modules = []
        
        for i, depth_reparam in enumerate(depth_reparam_modules):
            # 获取该视角的 DAP log-depth
            log_depth_dap_view = log_depth_daps[i]
            
            # 应用深度重参数化
            try:
                depth_transformed = depth_reparam(log_depth_dap_view)  # (H, W)
                log_depth_transformed = torch.log(depth_transformed + 1e-8)
            except Exception as e:
                log_time(f"      ❌ 深度重参数化失败 (view {i}): {e}")
                import traceback
                traceback.print_exc()
                raise
            
            depths.append(depth_transformed)
            log_depths.append(log_depth_transformed)
            
            # 获取方向缩放模块
            scale_modules.append(depth_reparam.scale_module)
        
        # 计算几何一致性损失
        try:
            geometry_loss_dict = self.geometry_loss_fn.compute_loss(
                depths=depths,
                cam_from_world_list=cam_from_world_list,
                depth_dap_list=depth_daps,
                height=depths[0].shape[0],
                width=depths[0].shape[1],
                masks=masks,
            )
        except Exception as e:
            log_time(f"    ❌ 几何一致性损失计算失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 计算正则化损失
        try:
            regularization_loss_dict = self.regularization_loss_fn.compute_loss(
                log_depths=log_depths,
                log_depth_daps=log_depth_daps,
                depth_daps=depth_daps,
                scale_modules=scale_modules,
                rgbs=rgbs,
                masks=masks,
            )
        except Exception as e:
            log_time(f"    ❌ 正则化损失计算失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 总损失
        total_loss = (
            geometry_loss_dict['total'] +
            regularization_loss_dict['total']
        )
        
        return {
            'total': total_loss,
            'geometry': geometry_loss_dict,
            'regularization': regularization_loss_dict,
        }
