"""
配置加载和验证工具
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List


def load_config(config_path: Path | str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证配置文件的合理性
    
    Args:
        config: 配置字典
        
    Returns:
        is_valid: 是否有效
        errors: 错误信息列表
    """
    errors = []
    
    # 检查必需字段
    required_fields = [
        'paths',
        'deformation',
        'geometry',
        'regularization',
        'optimization',
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必需字段: {field}")
    
    if errors:
        return False, errors
    
    # 检查路径存在性
    if 'paths' in config:
        paths = config['paths']
        if 'colmap_root' in paths:
            colmap_root = Path(paths['colmap_root'])
            if not colmap_root.exists():
                errors.append(f"COLMAP 根目录不存在: {colmap_root}")
        
        if 'stage_root' in paths:
            stage_root = Path(paths['stage_root'])
            if not stage_root.exists():
                errors.append(f"STAGE 根目录不存在: {stage_root}")
        
        if 'scene_name' in paths and 'stage_root' in paths:
            scene_dir = Path(paths['stage_root']) / paths['scene_name']
            if not scene_dir.exists():
                errors.append(f"场景目录不存在: {scene_dir}")
    
    # 检查权重合理性
    if 'geometry' in config:
        geometry = config['geometry']
        if 'point_to_ray' in geometry and 'depth_consistency' in geometry:
            p2r_weight = geometry['point_to_ray'].get('weight', 0)
            depth_weight = geometry['depth_consistency'].get('weight', 0)
            if p2r_weight <= depth_weight:
                errors.append(f"point_to_ray 权重 ({p2r_weight}) 必须大于 depth_consistency 权重 ({depth_weight})")
    
    # 检查正则化权重
    if 'regularization' in config:
        reg = config['regularization']
        if 'prior_anchor' in reg:
            prior_weight = reg['prior_anchor'].get('weight', 0)
            if prior_weight <= 0:
                errors.append("prior_anchor 权重必须大于 0（防止塌缩）")
    
    # 检查学习率
    if 'optimization' in config:
        opt = config['optimization']
        if 'solver' in opt:
            lr = opt['solver'].get('lr', 0)
            if lr <= 0 or lr > 1:
                errors.append(f"学习率应在 (0, 1] 范围内，当前值: {lr}")
    
    # 检查深度范围
    if 'initialization' in config:
        init = config['initialization']
        depth_min = init.get('depth_min', 0)
        depth_max = init.get('depth_max', 0)
        if depth_min >= depth_max or depth_min <= 0:
            errors.append(f"深度范围配置错误: [{depth_min}, {depth_max}]")
    
    return len(errors) == 0, errors


def get_data_paths(config: Dict[str, Any], pano_name: str) -> Dict[str, Path]:
    """
    根据配置获取数据路径
    
    Args:
        config: 配置字典
        pano_name: 全景图名称
        
    Returns:
        paths: 包含各种数据路径的字典
    """
    paths_config = config['paths']
    colmap_root = Path(paths_config['colmap_root'])
    stage_root = Path(paths_config['stage_root'])
    scene_name = paths_config['scene_name']
    
    paths = {
        'colmap_model': colmap_root / scene_name / 'sparse' / '0',
        'rgb': stage_root / scene_name / 'backgrounds' / f'{pano_name}.png',
        'depth_dap': stage_root / scene_name / 'depth_npy' / f'{pano_name}.npy',
        'fused_ply': colmap_root / scene_name / 'fused.ply',
    }
    
    return paths
