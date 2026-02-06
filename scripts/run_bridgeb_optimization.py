#!/usr/bin/env python3
"""
BridgeB 场景多视角深度联合优化脚本
运行 Step 4: 最终联合优化
"""
import sys
from pathlib import Path
import numpy as np
import torch
import pycolmap
from typing import List, Optional
import time
import json
import csv

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def log_time(msg: str):
    """打印带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

from src.utils.config import load_config, get_data_paths
from src.utils.io import load_image, load_depth_npy, save_depth_npy
from src.deformation import DepthReparameterization
from src.solver import JointOptimizationConfig, optimize_multi_view_depth, validate_optimized_depths
from scripts.visualize_depths import visualize_depth_diff_heatmap


def load_colmap_reconstruction(colmap_dir: Path) -> pycolmap.Reconstruction:
    """加载 COLMAP 重建结果"""
    recon = pycolmap.Reconstruction(str(colmap_dir))
    return recon


def get_camera_pose_for_pano(
    recon: pycolmap.Reconstruction,
    pano_name: str,
    camera_name: str = "pano_camera12",
) -> Optional[pycolmap.Rigid3d]:
    """
    获取指定 pano 的相机位姿
    
    Args:
        recon: COLMAP 重建结果
        pano_name: 全景图名称
        camera_name: 相机名称子串
        
    Returns:
        cam_from_world: pycolmap.Rigid3d 相机变换（实际上是 world_from_cam）
    """
    # 查找对应的图像
    for img_id, img in recon.images.items():
        # 图像名称格式: pano_camera{idx}/{pano_name}.png
        if camera_name in img.name and pano_name in img.name:
            # 获取相机位姿
            if hasattr(img, 'cam_from_world'):
                cam_from_world = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                return cam_from_world
    
    # 如果没找到，尝试通过 frame 查找
    for img_id, img in recon.images.items():
        if pano_name in img.name:
            if img.frame_id in recon.frames:
                frame = recon.frames[img.frame_id]
                if frame.has_pose():
                    # 从 frame 获取 rig_from_world
                    rig_from_world = frame.rig_from_world
                    # 查找该 frame 中指定相机的 cam_from_rig
                    for img2_id, img2 in recon.images.items():
                        if img2.frame_id == img.frame_id and camera_name in img2.name:
                            cam_from_world = img2.cam_from_world() if callable(img2.cam_from_world) else img2.cam_from_world
                            return cam_from_world
    
    return None


def build_pano_to_frame_mapping(recon: pycolmap.Reconstruction) -> dict:
    """建立 pano_name 到 frame_id 的映射"""
    pano_to_frame = {}
    for img_id, img in recon.images.items():
        if img.frame_id not in recon.frames:
            continue
        img_name = img.name
        if '/' in img_name:
            pano_name = img_name.split('/')[-1]
            pano_name = Path(pano_name).stem
            if pano_name not in pano_to_frame:
                pano_to_frame[pano_name] = img.frame_id
    return pano_to_frame


def main():
    """主函数"""
    start_time = time.time()
    print("=" * 80)
    print("BridgeB 场景多视角深度联合优化")
    print("=" * 80)
    
    # 加载配置
    config_path = project_root / "configs" / "bridgeb.yaml"
    log_time(f"📖 加载配置: {config_path}")
    config = load_config(config_path)
    log_time(f"✅ 配置加载完成 ({time.time() - start_time:.2f}s)")
    
    # 获取数据路径
    scene_name = config['paths']['scene_name']
    pano_names = config['paths']['pano_names']
    camera_name = config['paths']['camera_name']
    
    print(f"  场景: {scene_name}")
    print(f"  全景图数量: {len(pano_names)}")
    print(f"  全景图列表: {pano_names}")
    
    # COLMAP 路径
    colmap_root = Path(config['paths']['colmap_root'])
    colmap_dir = colmap_root / scene_name / "sparse" / "0"
    
    log_time(f"📖 读取 COLMAP 重建: {colmap_dir}")
    t0 = time.time()
    recon = load_colmap_reconstruction(colmap_dir)
    log_time(f"✅ COLMAP 重建加载完成 ({time.time() - t0:.2f}s)")
    
    # 加载所有 pano 的数据
    log_time(f"📦 开始加载数据...")
    t0 = time.time()
    depths_dap = []
    log_depths_dap = []
    rgbs = []
    cam_poses = []
    heights = []
    widths = []
    
    for idx, pano_name in enumerate(pano_names):
        log_time(f"  处理 {pano_name} ({idx+1}/{len(pano_names)})...")
        paths = get_data_paths(config, pano_name)
        
        # 加载 RGB
        t1 = time.time()
        log_time(f"    📷 加载 RGB...")
        rgb = load_image(paths['rgb'])
        H, W = rgb.shape[:2]
        heights.append(H)
        widths.append(W)
        rgbs.append(torch.from_numpy(rgb.astype(np.float32)))
        log_time(f"    ✅ RGB 加载完成 ({time.time() - t1:.2f}s)")
        
        # 加载 DAP 深度
        t1 = time.time()
        log_time(f"    🧊 加载 DAP 深度...")
        depth_dap = load_depth_npy(paths['depth_dap'])
        if depth_dap.shape != (H, W):
            raise ValueError(f"{pano_name}: DAP depth shape {depth_dap.shape} != RGB shape {(H, W)}")
        
        # DAP 深度缩放（假设 DAP 深度需要缩放 100 倍）
        dap_scale = 100.0
        depth_dap_scaled = depth_dap * dap_scale
        depths_dap.append(torch.from_numpy(depth_dap_scaled.astype(np.float32)))
        
        # log-depth
        log_depth_dap = np.log(depth_dap_scaled + 1e-8)
        log_depths_dap.append(torch.from_numpy(log_depth_dap.astype(np.float32)))
        log_time(f"    ✅ DAP 深度加载完成 ({time.time() - t1:.2f}s)")
        
        # 获取相机位姿
        t1 = time.time()
        log_time(f"    📐 获取相机位姿...")
        cam_pose = get_camera_pose_for_pano(recon, pano_name, camera_name)
        if cam_pose is None:
            raise ValueError(f"无法找到 {pano_name} 的相机位姿")
        cam_poses.append(cam_pose)
        log_time(f"    ✅ 相机位姿获取完成 ({time.time() - t1:.2f}s)")
    
    log_time(f"✅ 数据加载完成 ({time.time() - t0:.2f}s)")
    
    # 检查所有图像尺寸是否一致
    if len(set(heights)) > 1 or len(set(widths)) > 1:
        log_time(f"⚠️  警告: 图像尺寸不一致")
        log_time(f"  高度: {heights}")
        log_time(f"  宽度: {widths}")
        # 使用第一个图像的尺寸
        H, W = heights[0], widths[0]
    else:
        H, W = heights[0], widths[0]
    
    log_time(f"  图像尺寸: {H}x{W}")
    log_time(f"  深度范围: [{np.min([d.min().item() for d in depths_dap]):.2f}, {np.max([d.max().item() for d in depths_dap]):.2f}] 米")
    
    # 创建深度重参数化模块
    log_time(f"🔧 创建深度重参数化模块...")
    t0 = time.time()
    
    # 从配置文件读取深度变形模型配置
    deformation_config = config.get('deformation', {})
    use_directional_bspline = bool(deformation_config.get('use_directional_bspline', True))
    
    if use_directional_bspline:
        log_time(f"  使用新版本：方向 × log-depth B-spline grid")
        # 新版本配置
        bspline_config = deformation_config.get('directional_bspline_grid', {})
        n_alpha = int(bspline_config.get('n_alpha', 12))
        n_depth = int(bspline_config.get('n_depth', 10))
        alpha_method = str(bspline_config.get('alpha_method', 'asin'))
        max_delta_log = float(bspline_config.get('max_delta_log', 0.5))
        spline_order = int(bspline_config.get('spline_order', 3))
        
        # 旧版本 spline 配置（用于 log_depth_min/max）
        monotonic_spline_config = deformation_config.get('monotonic_spline', {})
        log_depth_min = float(monotonic_spline_config.get('log_depth_min', -3.0))
        log_depth_max = float(monotonic_spline_config.get('log_depth_max', 5.0))
        
        log_time(f"    n_alpha={n_alpha}, n_depth={n_depth}, alpha_method={alpha_method}")
        log_time(f"    max_delta_log={max_delta_log}, log_depth_range=[{log_depth_min}, {log_depth_max}]")
    else:
        log_time(f"  使用旧版本：全局 spline + 方向缩放")
        # 旧版本配置
        monotonic_spline_config = deformation_config.get('monotonic_spline', {})
        directional_scale_config = deformation_config.get('directional_scale', {})
        
        spline_type = str(monotonic_spline_config.get('type', 'monotonic_cubic'))
        num_knots = int(monotonic_spline_config.get('num_knots', 10))
        log_depth_min = float(monotonic_spline_config.get('log_depth_min', -3.0))
        log_depth_max = float(monotonic_spline_config.get('log_depth_max', 5.0))
        freeze_reference_point = bool(monotonic_spline_config.get('freeze_reference_point', True))
        reference_log_depth = float(monotonic_spline_config.get('reference_log_depth', 0.0))
        
        scale_method = str(directional_scale_config.get('method', 'spherical_harmonics'))
        sh_max_degree = int(directional_scale_config.get('sh_max_degree', 4))
        max_scale_log = float(directional_scale_config.get('max_scale_log', 0.3))
        
        log_time(f"    spline_type={spline_type}, scale_method={scale_method}")
    
    depth_reparam_modules = []
    for i, pano_name in enumerate(pano_names):
        t1 = time.time()
        log_time(f"  {pano_name}: 创建模块...")
        
        if use_directional_bspline:
            # 新版本：方向 × log-depth B-spline grid
            module = DepthReparameterization(
                height=H,
                width=W,
                use_directional_bspline=True,
                n_alpha=n_alpha,
                n_depth=n_depth,
                alpha_method=alpha_method,
                max_delta_log=max_delta_log,
                log_depth_min=log_depth_min,
                log_depth_max=log_depth_max,
            )
        else:
            # 旧版本：全局 spline + 方向缩放
            module = DepthReparameterization(
                height=H,
                width=W,
                use_directional_bspline=False,
                spline_type=spline_type,
                num_knots=num_knots,
                log_depth_min=log_depth_min,
                log_depth_max=log_depth_max,
                freeze_reference_point=freeze_reference_point,
                reference_log_depth=reference_log_depth,
                scale_method=scale_method,
                sh_max_degree=sh_max_degree,
                max_scale_log=max_scale_log,
            )
        
        depth_reparam_modules.append(module)
        log_time(f"  ✅ {pano_name} 模块创建完成 ({time.time() - t1:.2f}s)")
    log_time(f"✅ 所有模块创建完成 ({time.time() - t0:.2f}s)")
    
    # 配置优化器（从配置文件读取参数）
    log_time(f"⚙️  配置优化器...")
    t0 = time.time()
    
    # 从配置文件读取优化参数
    geometry_config = config.get('geometry', {})
    regularization_config = config.get('regularization', {})
    optimization_config = config.get('optimization', {})
    
    # 几何一致性权重（确保类型正确）
    p2r_config = geometry_config.get('point_to_ray', {})
    p2r_enabled = p2r_config.get('enabled', True)  # 检查是否启用
    if not p2r_enabled:
        log_time(f"⚠️  警告: point_to_ray 已禁用，几何约束将不会生效！")
    lambda_p2r = float(p2r_config.get('weight', 1.0)) if p2r_enabled else 0.0  # 如果禁用则权重为0
    use_robust_p2r = bool(p2r_config.get('use_robust_loss', True))  # 确保转换为布尔值
    huber_delta_p2r = float(p2r_config.get('huber_delta', 0.1))  # 确保转换为浮点数
    
    depth_config = geometry_config.get('depth_consistency', {})
    lambda_depth = float(depth_config.get('weight', 0.1))  # 确保转换为浮点数
    
    far_config = geometry_config.get('far_field', {})
    far_threshold = float(far_config.get('far_threshold', 100.0))  # 确保转换为浮点数
    
    # 正则化权重（确保类型正确）
    prior_config = regularization_config.get('prior_anchor', {})
    lambda_prior = float(prior_config.get('weight', 1.0))  # 确保转换为浮点数
    prior_loss_type = str(prior_config.get('loss_type', 'l2'))  # 确保转换为字符串
    prior_huber_delta = float(prior_config.get('huber_delta', 0.1))  # 确保转换为浮点数
    
    smooth_config = regularization_config.get('smoothness', {})
    lambda_smooth = float(smooth_config.get('weight', 0.01))  # 确保转换为浮点数
    smooth_type = str(smooth_config.get('smooth_type', 'l2'))  # 确保转换为字符串
    edge_aware = bool(smooth_config.get('edge_aware', False))  # 确保转换为布尔值
    rgb_sigma = float(smooth_config.get('rgb_sigma', 10.0))  # 确保转换为浮点数
    
    scale_config = regularization_config.get('scale_constraint', {})
    lambda_scale = float(scale_config.get('weight', 0.01))  # 确保转换为浮点数
    
    # B-spline 约束权重（新版本）
    bspline_constraints_config = deformation_config.get('bspline_constraints', {})
    lambda_mono = float(bspline_constraints_config.get('lambda_mono', 0.1))
    lambda_smooth_bspline = float(bspline_constraints_config.get('lambda_smooth', 0.001))
    lambda_far = float(bspline_constraints_config.get('lambda_far', 0.1))
    
    # 优化器配置
    solver_config = optimization_config.get('solver', {})
    optimizer = solver_config.get('optimizer', 'adam')
    lr_raw = solver_config.get('lr', 1e-3)
    # 处理字符串形式的科学计数法（如 "5e-4"）
    if isinstance(lr_raw, str):
        lr = float(lr_raw)
    else:
        lr = float(lr_raw)  # 确保转换为浮点数
    
    iteration_config = optimization_config.get('iteration', {})
    max_iter = int(iteration_config.get('max_iter', 1000))  # 确保转换为整数
    early_stop_threshold_raw = iteration_config.get('early_stop_threshold', 1e-6)
    # 处理字符串形式的科学计数法（如 "1e-7"）
    if isinstance(early_stop_threshold_raw, str):
        early_stop_threshold = float(early_stop_threshold_raw)
    else:
        early_stop_threshold = float(early_stop_threshold_raw)  # 确保转换为浮点数
    save_interval = int(iteration_config.get('save_interval', 100))  # 确保转换为整数
    print_interval = int(iteration_config.get('print_interval', 10))  # 确保转换为整数
    
    device_config = optimization_config.get('device', {})
    device = device_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    opt_config = JointOptimizationConfig(
        lambda_p2r=lambda_p2r,
        lambda_depth=lambda_depth,
        lambda_prior=lambda_prior,
        lambda_smooth=lambda_smooth,
        lambda_scale=lambda_scale,
        lambda_mono=lambda_mono,
        lambda_smooth_bspline=lambda_smooth_bspline,
        lambda_far=lambda_far,
        optimizer=optimizer,
        lr=lr,
        max_iter=max_iter,
        early_stop_threshold=early_stop_threshold,
        far_threshold=far_threshold,
        use_robust_p2r=use_robust_p2r,
        huber_delta_p2r=huber_delta_p2r,
        prior_loss_type=prior_loss_type,
        prior_huber_delta=prior_huber_delta,
        smooth_type=smooth_type,
        edge_aware=edge_aware,
        rgb_sigma=rgb_sigma,
        device=device,
        save_history=True,
        save_interval=save_interval,
        print_interval=print_interval,
    )
    log_time(f"✅ 优化器配置完成 ({time.time() - t0:.2f}s)")
    
    log_time(f"  设备: {opt_config.device}")
    log_time(f"  最大迭代次数: {opt_config.max_iter}")
    log_time(f"  学习率: {opt_config.lr}")
    
    # 将数据移到设备
    log_time(f"📱 将数据移到设备: {opt_config.device}")
    t0 = time.time()
    device = torch.device(opt_config.device)
    
    log_time(f"  移动 log_depth_daps...")
    log_depths_dap = [d.to(device) for d in log_depths_dap]
    log_time(f"  移动 depths_dap...")
    depths_dap = [d.to(device) for d in depths_dap]
    log_time(f"  移动 rgbs...")
    rgbs = [r.to(device) for r in rgbs]
    log_time(f"  移动深度重参数化模块...")
    for i, module in enumerate(depth_reparam_modules):
        t1 = time.time()
        module.to(device)
        log_time(f"    模块 {i} 已移动到 {device} ({time.time() - t1:.2f}s)")
    
    log_time(f"✅ 数据移动完成 ({time.time() - t0:.2f}s)")
    
    # 检查内存使用
    if torch.cuda.is_available() and opt_config.device == "cuda":
        log_time(f"  GPU 内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # 运行优化
    print(f"\n🚀 开始联合优化...")
    output_dir = project_root / "intermediate" / "bridgeb_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理内存
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        depths_opt, report = optimize_multi_view_depth(
            depth_reparam_modules=depth_reparam_modules,
            log_depth_daps=log_depths_dap,
            depth_daps=depths_dap,
            cam_from_world_list=cam_poses,
            config=opt_config,
            rgbs=rgbs,
            masks=None,
            output_dir=output_dir,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ GPU 内存不足！")
            print(f"  尝试使用 CPU 模式...")
            opt_config.device = "cpu"
            # 将数据移到 CPU
            log_depths_dap = [d.cpu() for d in log_depths_dap]
            depths_dap = [d.cpu() for d in depths_dap]
            rgbs = [r.cpu() for r in rgbs]
            for module in depth_reparam_modules:
                module.to("cpu")
            # 重新运行
            depths_opt, report = optimize_multi_view_depth(
                depth_reparam_modules=depth_reparam_modules,
                log_depth_daps=log_depths_dap,
                depth_daps=depths_dap,
                cam_from_world_list=cam_poses,
                config=opt_config,
                rgbs=rgbs,
                masks=None,
                output_dir=output_dir,
            )
        else:
            raise
    
    print(f"\n✅ 优化完成")
    print(f"  迭代次数: {report['iterations']}")
    print(f"  最终能量: {report['final_energy']:.6f}")
    
    # 保存损失历史
    if report.get('history') is not None:
        log_time(f"📊 保存损失历史...")
        history = report['history']
        config_dict = report.get('config', {})
        
        # 创建损失历史目录
        loss_history_dir = project_root / "logs" / "loss_history" / scene_name
        loss_history_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存为 JSON（包含配置信息）
        json_path = loss_history_dir / f"loss_history_{timestamp}.json"
        history_data = {
            'config': config_dict,
            'iterations': report['iterations'],
            'final_energy': report['final_energy'],
            'history': {k: v for k, v in history.items()}
        }
        with open(json_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        log_time(f"  ✅ JSON 保存: {json_path}")
        
        # 保存为 CSV（便于 Excel/Python 分析）
        csv_path = loss_history_dir / f"loss_history_{timestamp}.csv"
        num_iterations = len(history['total'])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                'iteration',
                'total',
                'p2r_raw', 'depth_raw', 'prior_raw', 'smooth_raw', 'scale_raw',
                'p2r_weighted', 'depth_weighted', 'prior_weighted', 'smooth_weighted', 'scale_weighted'
            ])
            # 写入数据
            for i in range(num_iterations):
                writer.writerow([
                    i + 1,
                    history['total'][i],
                    history['geometry_p2r'][i],
                    history['geometry_depth'][i],
                    history['regularization_prior'][i],
                    history['regularization_smooth'][i],
                    history['regularization_scale'][i],
                    history['weighted_p2r'][i],
                    history['weighted_depth'][i],
                    history['weighted_prior'][i],
                    history['weighted_smooth'][i],
                    history['weighted_scale'][i],
                ])
        log_time(f"  ✅ CSV 保存: {csv_path}")
        
        # 打印损失统计
        print(f"\n📊 损失统计:")
        print(f"  总损失: 初始={history['total'][0]:.6f}, 最终={history['total'][-1]:.6f}, 变化={history['total'][0]-history['total'][-1]:.6f}")
        print(f"  P2R损失: 初始={history['geometry_p2r'][0]:.6f}, 最终={history['geometry_p2r'][-1]:.6f}")
        print(f"  先验损失: 初始={history['regularization_prior'][0]:.6f}, 最终={history['regularization_prior'][-1]:.6f}")
        print(f"  加权P2R: 最终={history['weighted_p2r'][-1]:.6f} (权重={config_dict.get('lambda_p2r', 'N/A')})")
        print(f"  加权先验: 最终={history['weighted_prior'][-1]:.6f} (权重={config_dict.get('lambda_prior', 'N/A')})")
    
    # 验证结果
    print(f"\n🔍 验证优化结果...")
    is_valid, validation_report = validate_optimized_depths(
        depths=depths_opt,
        depth_daps=[d.cpu().numpy() for d in depths_dap],
        far_threshold=100.0,
    )
    
    print(f"  验证结果: {'✅通过' if is_valid else '⚠️未通过'}")
    if not is_valid:
        print(f"  警告: {validation_report}")
    
    # 保存结果
    print(f"\n💾 保存结果...")
    output_depths_dir = project_root / "outputs" / "aligned_depths" / scene_name
    output_depths_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (pano_name, depth_opt) in enumerate(zip(pano_names, depths_opt)):
        output_path = output_depths_dir / f"{pano_name}_aligned.npy"
        save_depth_npy(depth_opt, output_path)
        print(f"  ✅ {pano_name}: {output_path}")
    
    # 可视化深度变化量对比
    print(f"\n📊 生成深度变化量热力图...")
    diff_vis_dir = output_depths_dir / "depth_diff_heatmaps"
    diff_vis_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (pano_name, depth_opt) in enumerate(zip(pano_names, depths_opt)):
        # 获取优化前的深度（DAP缩放后的）
        depth_before = depths_dap[i].cpu().numpy()  # 已经是缩放后的
        
        # 生成三种类型的对比图
        for diff_type in ["log_diff", "absolute", "relative"]:
            heatmap_path = diff_vis_dir / f"{pano_name}_diff_{diff_type}.png"
            try:
                visualize_depth_diff_heatmap(
                    depth_before=depth_before,
                    depth_after=depth_opt,
                    diff_type=diff_type,
                    cmap="RdBu_r",
                    vmax=None,  # 自动计算
                    save_path=heatmap_path,
                )
            except Exception as e:
                print(f"  ⚠️  {pano_name} {diff_type} 可视化失败: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"  ✅ 深度变化量热力图已保存到: {diff_vis_dir}")
    print(f"     包含: log_diff, absolute, relative 三种对比方式")
    
    print(f"\n🎉 全部完成！")
    print(f"  优化深度目录: {output_depths_dir}")
    print(f"  变化量热力图: {diff_vis_dir}")


if __name__ == "__main__":
    main()
