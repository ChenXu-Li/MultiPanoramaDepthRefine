"""
多视角几何一致性损失组合
组合 Point-to-Ray 和 Ray-space 深度一致性
"""
import torch
import numpy as np
from typing import List, Optional, Dict
import pycolmap
import time

from .point_to_ray import compute_point_to_ray_loss
from .ray_space_consistency import (
    compute_ray_space_depth_consistency_loss,
    project_point_to_camera,
)
from .coordinate_transform import (
    depth_to_world_points,
    get_camera_center_world,
    get_camera_ray_directions_world,
)

def log_time(msg: str):
    """打印带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class MultiViewGeometryLoss:
    """
    多视角几何一致性损失
    
    组合：
    - Point-to-Ray 距离（主约束）
    - Ray-space 深度一致性（弱约束）
    """
    
    def __init__(
        self,
        lambda_p2r: float = 1.0,
        lambda_depth: float = 0.1,
        far_threshold: float = 100.0,
        use_robust_p2r: bool = True,
        huber_delta_p2r: float = 0.1,
        use_robust_depth: bool = False,
        device: str = "cpu",
    ):
        """
        Args:
            lambda_p2r: Point-to-Ray 损失权重（必须 >> lambda_depth）
            lambda_depth: 深度一致性损失权重
            far_threshold: 远景阈值（米），>= 此深度的像素不参与几何对齐
            use_robust_p2r: 是否对 Point-to-Ray 使用 robust loss
            huber_delta_p2r: Point-to-Ray Huber loss 阈值
            use_robust_depth: 是否对深度一致性使用 robust loss
            device: 计算设备
        """
        self.lambda_p2r = lambda_p2r
        self.lambda_depth = lambda_depth
        self.far_threshold = far_threshold
        self.use_robust_p2r = use_robust_p2r
        self.huber_delta_p2r = huber_delta_p2r
        self.use_robust_depth = use_robust_depth
        self.device = device
        
        # 初始化调试计数器（每个实例独立）
        self._p2r_call_count = 0
        self._proj_debug_printed = False
        self._p2r_zero_warned = False
        
        # 验证权重合理性
        if lambda_p2r <= lambda_depth:
            raise ValueError(f"lambda_p2r ({lambda_p2r}) 必须大于 lambda_depth ({lambda_depth})")
    
    def compute_loss(
        self,
        depths: List[torch.Tensor],
        cam_from_world_list: List[pycolmap.Rigid3d],
        depth_dap_list: List[torch.Tensor],
        height: int,
        width: int,
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算多视角几何一致性损失
        
        Args:
            depths: List[(H, W)] 各视角的深度图
            cam_from_world_list: List[pycolmap.Rigid3d] 各视角的相机变换
            depth_dap_list: List[(H, W)] 各视角的 DAP 深度图（用于远景 mask）
            height: 图像高度
            width: 图像宽度
            masks: List[(H, W)] 可选，各视角的有效像素 mask
            
        Returns:
            loss_dict: 包含各项损失的字典
                - 'total': 总损失
                - 'p2r': Point-to-Ray 损失
                - 'depth': 深度一致性损失
        """
        num_views = len(depths)
        
        if masks is None:
            masks = [None] * num_views
        
        # 创建远景 mask（D^{DAP} >= far_threshold）
        far_masks = []
        for depth_dap in depth_dap_list:
            far_mask = depth_dap >= self.far_threshold  # (H, W)
            far_masks.append(far_mask)
        
        # 计算所有视角对之间的损失
        total_p2r_loss = 0.0
        total_depth_loss = 0.0
        pair_count = 0
        
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # 创建有效 mask（排除远景）
                valid_mask_i = ~far_masks[i]  # (H, W)
                valid_mask_j = ~far_masks[j]  # (H, W)
                
                if masks[i] is not None:
                    valid_mask_i = valid_mask_i & masks[i]
                if masks[j] is not None:
                    valid_mask_j = valid_mask_j & masks[j]
                
                # Point-to-Ray 损失
                if self.lambda_p2r > 0:
                    try:
                        # 将视角 i 的深度转换为世界点
                        points_world_i = depth_to_world_points(
                            depths[i],
                            cam_from_world_list[i],
                            height,
                            width,
                            device=self.device,
                        )  # (H, W, 3)
                        
                        # 获取视角 j 的相机中心和射线方向
                        camera_center_j = torch.tensor(
                            get_camera_center_world(cam_from_world_list[j]),
                            dtype=torch.float32,
                            device=self.device
                        )  # (3,)
                        
                        # 正确计算 P2R 损失：将视角 i 的世界点投影到视角 j，然后计算距离
                        p2r_loss = self._compute_p2r_loss_correct(
                            points_world_i,
                            camera_center_j,
                            cam_from_world_list[j],
                            height,
                            width,
                            valid_mask_i,
                        )
                        
                        total_p2r_loss = total_p2r_loss + p2r_loss
                    except Exception as e:
                        log_time(f"        ❌ Point-to-Ray 损失计算失败 (view {i}, {j}): {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                # Ray-space 深度一致性损失
                if self.lambda_depth > 0:
                    try:
                        depth_loss = compute_ray_space_depth_consistency_loss(
                            depths[i],
                            depths[j],
                            cam_from_world_list[i],
                            cam_from_world_list[j],
                            height,
                            width,
                            mask=valid_mask_i,
                            use_robust_loss=self.use_robust_depth,
                            device=self.device,
                        )
                        total_depth_loss = total_depth_loss + depth_loss
                    except Exception as e:
                        log_time(f"        ❌ Ray-space 损失计算失败 (view {i}, {j}): {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                pair_count += 1
        
        # 平均化
        if pair_count > 0:
            total_p2r_loss = total_p2r_loss / pair_count
            total_depth_loss = total_depth_loss / pair_count
        
        # 总损失
        total_loss = (
            self.lambda_p2r * total_p2r_loss +
            self.lambda_depth * total_depth_loss
        )
        
        return {
            'total': total_loss,
            'p2r': total_p2r_loss,
            'depth': total_depth_loss,
        }
    
    def _compute_p2r_loss_simple(
        self,
        points_world: torch.Tensor,
        camera_center: torch.Tensor,
        ray_directions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        简化版本的 Point-to-Ray 损失计算
        
        注意：此函数假设 points_world 和 ray_directions 已经通过投影建立了对应关系。
        如果直接使用相同像素坐标，会导致错误的对应关系。
        
        Args:
            points_world: (H, W, 3) 世界点（视角 i 的）
            camera_center: (3,) 相机中心（视角 j 的）
            ray_directions: (H, W, 3) 射线方向（视角 j 的，已与 points_world 对齐）
            mask: (H, W) 可选，有效像素 mask
            
        Returns:
            loss: 标量损失值
        """
        H, W, _ = points_world.shape
        
        # 计算每个点到对应射线的距离
        vec = points_world - camera_center.unsqueeze(0).unsqueeze(0)  # (H, W, 3)
        proj_length = torch.sum(vec * ray_directions, dim=-1, keepdim=True)  # (H, W, 1)
        proj_vec = proj_length * ray_directions  # (H, W, 3)
        perp_vec = vec - proj_vec  # (H, W, 3)
        distances = torch.norm(perp_vec, dim=-1)  # (H, W)
        
        # 应用 mask
        if mask is not None:
            distances = distances * mask.float()
            valid_count = mask.sum().float()
            if valid_count > 0:
                # 注意：这里不应该除以 valid_count，应该直接平均
                # 因为 distances 已经是每个像素的距离了
                pass
        
        # 应用 robust loss
        if self.use_robust_p2r:
            abs_distances = torch.abs(distances)
            huber_mask = abs_distances <= self.huber_delta_p2r
            loss = torch.where(
                huber_mask,
                0.5 * distances ** 2,
                self.huber_delta_p2r * abs_distances - 0.5 * self.huber_delta_p2r ** 2
            )
            if mask is not None:
                loss = (loss * mask.float()).sum() / (mask.sum().float() + 1e-8)
            else:
                loss = loss.mean()
        else:
            if mask is not None:
                loss = ((distances ** 2) * mask.float()).sum() / (mask.sum().float() + 1e-8)
            else:
                loss = (distances ** 2).mean()
        
        return loss
    
    def _compute_p2r_loss_correct(
        self,
        points_world_i: torch.Tensor,
        camera_center_j: torch.Tensor,
        cam_from_world_j: "pycolmap.Rigid3d",
        height: int,
        width: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        正确版本的 Point-to-Ray 损失计算
        
        对于视角 i 的每个世界点 P_i：
        1. 将 P_i 投影到视角 j，得到像素坐标 (u_j, v_j)
        2. 获取视角 j 在 (u_j, v_j) 处的射线方向
        3. 计算 P_i 到该射线的距离
        
        Args:
            points_world_i: (H, W, 3) 视角 i 的世界点
            camera_center_j: (3,) 视角 j 的相机中心
            cam_from_world_j: 视角 j 的相机变换
            height: 图像高度
            width: 图像宽度
            mask: (H, W) 可选，有效像素 mask
            
        Returns:
            loss: 标量损失值
        """
        H, W, _ = points_world_i.shape
        H_W = H * W
        
        # 将世界点重塑为 (H*W, 3)
        points_world_flat = points_world_i.reshape(H_W, 3)  # (H*W, 3)
        
        # 将每个世界点投影到视角 j 的像素坐标
        u_flat, v_flat, valid_proj = project_point_to_camera(
            points_world_flat,
            cam_from_world_j,
            height,
            width,
            device=self.device,
        )  # (H*W,), (H*W,), (H*W,)
        
        # 转换为像素索引
        u_pix_flat = torch.clamp((u_flat * width).long(), 0, width - 1)  # (H*W,)
        v_pix_flat = torch.clamp((v_flat * height).long(), 0, height - 1)  # (H*W,)
        
        # 获取视角 j 的射线方向（完整网格）
        ray_directions_j = get_camera_ray_directions_world(
            cam_from_world_j,
            height,
            width,
            device=self.device,
        )  # (H, W, 3)
        
        # 根据投影坐标采样对应的射线方向
        ray_dirs_flat = ray_directions_j[v_pix_flat, u_pix_flat]  # (H*W, 3)
        
        # 计算每个点到对应射线的距离
        vec = points_world_flat - camera_center_j.unsqueeze(0)  # (H*W, 3)
        proj_length = torch.sum(vec * ray_dirs_flat, dim=-1, keepdim=True)  # (H*W, 1)
        proj_vec = proj_length * ray_dirs_flat  # (H*W, 3)
        perp_vec = vec - proj_vec  # (H*W, 3)
        distances = torch.norm(perp_vec, dim=-1)  # (H*W,)
        
        # 应用投影有效性和 mask
        if mask is not None:
            mask_flat = mask.reshape(H_W)  # (H*W,)
            valid_mask = valid_proj & mask_flat  # (H*W,)
        else:
            valid_mask = valid_proj  # (H*W,)
        
        # 调试：检查投影有效性（只在第一次调用时打印）
        if not self._proj_debug_printed:
            valid_proj_count = valid_proj.sum().item()
            mask_count = mask.sum().item() if mask is not None else H_W
            valid_mask_count = valid_mask.sum().item()
            print(f"[DEBUG P2R投影] 总点数={H_W}, 投影有效={valid_proj_count} ({100*valid_proj_count/H_W:.1f}%), "
                  f"mask有效={mask_count}, 最终有效={valid_mask_count} ({100*valid_mask_count/H_W:.1f}%)")
            self._proj_debug_printed = True
        
        # 只计算有效点的距离
        if valid_mask.sum() == 0:
            # 没有有效点，返回 0（或一个小的非零值避免梯度消失）
            if not self._p2r_zero_warned:
                print(f"[WARNING P2R] 没有有效点！valid_mask.sum()=0")
                self._p2r_zero_warned = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        valid_distances = distances[valid_mask]  # (N_valid,)
        
        # 调试：检查距离统计（每次迭代都打印，便于诊断）
        if valid_distances.numel() > 0:
            mean_dist = valid_distances.mean().item()
            max_dist = valid_distances.max().item()
            min_dist = valid_distances.min().item()
            median_dist = valid_distances.median().item()
            p95_dist = torch.quantile(valid_distances, 0.95).item()
            # 只在第一次或每N次迭代打印，避免日志过多
            self._p2r_call_count += 1
            if self._p2r_call_count == 1 or self._p2r_call_count % 100 == 0:
                print(f"[DEBUG P2R] 调用#{self._p2r_call_count}: 有效点数={valid_distances.numel()}/{H_W} ({100*valid_distances.numel()/H_W:.1f}%), "
                      f"距离: min={min_dist:.6f}, median={median_dist:.6f}, "
                      f"mean={mean_dist:.6f}, p95={p95_dist:.6f}, max={max_dist:.6f} (米)")
        
        # 应用 robust loss
        if self.use_robust_p2r:
            abs_distances = torch.abs(valid_distances)
            huber_mask = abs_distances <= self.huber_delta_p2r
            loss = torch.where(
                huber_mask,
                0.5 * valid_distances ** 2,
                self.huber_delta_p2r * abs_distances - 0.5 * self.huber_delta_p2r ** 2
            )
            loss = loss.mean()
        else:
            loss = (valid_distances ** 2).mean()
        
        return loss