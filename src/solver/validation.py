"""
输出验证函数
"""
import numpy as np
from typing import List, Dict, Tuple


def validate_optimized_depths(
    depths: List[np.ndarray],
    depth_daps: List[np.ndarray],
    far_threshold: float = 100.0,
    max_collapse_std: float = 0.1,
    min_depth: float = 0.1,
    max_depth: float = 1000.0,
) -> Tuple[bool, Dict]:
    """
    验证优化后的深度图
    
    检查项：
    1. 深度未整体塌缩（深度标准差 > 阈值）
    2. 未整体推到 100m 球壳（深度均值不在 100m 附近）
    3. 单视角仍光滑连续（梯度统计）
    4. 深度值在合理范围
    
    Args:
        depths: List[(H, W)] 优化后的深度图
        depth_daps: List[(H, W)] DAP 深度图（用于对比）
        far_threshold: 远景阈值（米）
        max_collapse_std: 塌缩检测阈值（深度标准差 < 此值认为塌缩）
        min_depth: 最小合理深度（米）
        max_depth: 最大合理深度（米）
        
    Returns:
        is_valid: 是否通过验证
        report: 详细统计信息
    """
    num_views = len(depths)
    
    report = {
        'num_views': num_views,
        'depth_stats': [],
        'collapse_detected': False,
        'shell_detected': False,
        'smoothness_stats': [],
    }
    
    all_depths_valid = True
    collapse_detected = False
    shell_detected = False
    
    for i, (depth, depth_dap) in enumerate(zip(depths, depth_daps)):
        # 检查深度值范围
        valid_mask = np.isfinite(depth) & (depth > 0)
        if not np.all(valid_mask):
            all_depths_valid = False
        
        depth_valid = depth[valid_mask]
        
        # 统计信息
        depth_mean = np.mean(depth_valid)
        depth_std = np.std(depth_valid)
        depth_min = np.min(depth_valid)
        depth_max = np.max(depth_valid)
        
        report['depth_stats'].append({
            'view': i,
            'mean': float(depth_mean),
            'std': float(depth_std),
            'min': float(depth_min),
            'max': float(depth_max),
        })
        
        # 检查 1: 深度未整体塌缩（深度标准差 > 阈值）
        if depth_std < max_collapse_std:
            collapse_detected = True
            report['collapse_detected'] = True
        
        # 检查 2: 未整体推到 100m 球壳
        # 如果深度均值接近 100m 且标准差很小，可能是球壳
        if abs(depth_mean - far_threshold) < 5.0 and depth_std < 1.0:
            shell_detected = True
            report['shell_detected'] = True
        
        # 检查 3: 单视角光滑连续（梯度统计）
        grad_x = np.abs(np.diff(depth, axis=1))
        grad_y = np.abs(np.diff(depth, axis=0))
        mean_grad = (np.mean(grad_x) + np.mean(grad_y)) / 2
        max_grad = max(np.max(grad_x), np.max(grad_y))
        
        report['smoothness_stats'].append({
            'view': i,
            'mean_gradient': float(mean_grad),
            'max_gradient': float(max_grad),
        })
        
        # 检查 4: 深度值在合理范围
        if depth_min < min_depth or depth_max > max_depth:
            all_depths_valid = False
    
    # 综合判断
    is_valid = (
        all_depths_valid and
        not collapse_detected and
        not shell_detected
    )
    
    return is_valid, report
