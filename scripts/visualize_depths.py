#!/usr/bin/env python3
"""
可视化优化后的深度图
参考 DAP 的 infer_pics.sh 和 pred_to_vis 函数
"""
import sys
from pathlib import Path
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import cv2

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_depth_npy, load_image


def colorize_depth_fixed(depth_u8: np.ndarray, cmap: str = "Spectral") -> np.ndarray:
    """
    将 uint8 深度图转换为彩色可视化
    
    Args:
        depth_u8: (H, W) uint8, 0~255
        cmap: colormap 名称（如 'Spectral', 'turbo', 'viridis'）
        
    Returns:
        colored: (H, W, 3) uint8 RGB图像
    """
    disp = depth_u8.astype(np.float32) / 255.0
    colored = matplotlib.colormaps[cmap](disp)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    return np.ascontiguousarray(colored)


def depth_to_vis(
    depth: np.ndarray,
    vis_range: str = "100m",
    cmap: str = "Spectral",
    vmin: float = None,
    vmax: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将深度图转换为可视化图像
    
    Args:
        depth: (H, W) float32 深度图（米）
        vis_range: 可视化范围（'100m' 或 '10m'）
        cmap: colormap 名称
        vmin: 最小深度值（米），如果为 None 则自动计算
        vmax: 最大深度值（米），如果为 None 则自动计算
        
    Returns:
        depth_gray: (H, W) uint8 灰度深度图
        depth_color: (H, W, 3) uint8 RGB 彩色深度图
    """
    # 创建有效 mask
    mask = np.isfinite(depth) & (depth > 0)
    
    if mask.sum() == 0:
        # 没有有效像素
        depth_gray = np.zeros_like(depth, dtype=np.uint8)
        depth_color = np.zeros((*depth.shape, 3), dtype=np.uint8)
        return depth_gray, depth_color
    
    # 计算深度范围
    if vmin is None:
        vmin = np.nanmin(depth[mask])
    if vmax is None:
        vmax = np.nanmax(depth[mask])
    
    # 根据 vis_range 处理深度值
    if vis_range == "100m":
        # 将深度范围映射到 [0, 100] 米，然后归一化到 [0, 255]
        depth_clip = np.clip(depth, vmin, min(vmax, 100.0))
        depth_norm = (depth_clip - vmin) / (min(vmax, 100.0) - vmin + 1e-8)
        depth_gray = (depth_norm * 255).astype(np.uint8)
    elif vis_range == "10m":
        # 将深度范围映射到 [0, 10] 米，然后归一化到 [0, 255]
        depth_clip = np.clip(depth, vmin, min(vmax, 10.0))
        depth_norm = (depth_clip - vmin) / (min(vmax, 10.0) - vmin + 1e-8)
        depth_gray = (depth_norm * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown vis_range: {vis_range} (use '100m' or '10m')")
    
    # 无效区域设为 0
    depth_gray[~mask] = 0
    
    # 转换为彩色
    depth_color = colorize_depth_fixed(depth_gray, cmap=cmap)
    
    return depth_gray, depth_color


def visualize_depth_diff_heatmap(
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    diff_type: str = "log_diff",
    cmap: str = "RdBu_r",
    vmax: float = None,
    save_path: Path = None,
) -> np.ndarray:
    """
    可视化深度变化量热力图
    
    Args:
        depth_before: (H, W) float32 优化前的深度图（米）
        depth_after: (H, W) float32 优化后的深度图（米）
        diff_type: 变化量类型
            - "absolute": 绝对变化量 |depth_after - depth_before|
            - "relative": 相对变化量 |depth_after - depth_before| / depth_before
            - "log_diff": log深度差 |log(depth_after) - log(depth_before)|（推荐）
        cmap: colormap 名称（默认：RdBu_r，红蓝对比）
        vmax: 最大变化量（用于归一化），如果为 None 则自动计算（使用95分位数）
        save_path: 保存路径，如果为 None 则不保存
        
    Returns:
        heatmap: (H, W, 3) uint8 RGB 热力图
    """
    # 创建有效 mask
    mask_before = np.isfinite(depth_before) & (depth_before > 0)
    mask_after = np.isfinite(depth_after) & (depth_after > 0)
    mask = mask_before & mask_after
    
    if mask.sum() == 0:
        # 没有有效像素
        heatmap = np.zeros((*depth_before.shape, 3), dtype=np.uint8)
        if save_path is not None:
            cv2.imwrite(str(save_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        return heatmap
    
    # 计算变化量
    if diff_type == "absolute":
        diff = np.abs(depth_after - depth_before)
        diff_label = "Absolute Depth Change (m)"
    elif diff_type == "relative":
        diff = np.abs(depth_after - depth_before) / (depth_before + 1e-8)
        diff_label = "Relative Depth Change"
    elif diff_type == "log_diff":
        log_before = np.log(depth_before + 1e-8)
        log_after = np.log(depth_after + 1e-8)
        diff = np.abs(log_after - log_before)
        diff_label = "Log Depth Change"
    else:
        raise ValueError(f"Unknown diff_type: {diff_type} (use 'absolute', 'relative', or 'log_diff')")
    
    # 无效区域设为 NaN
    diff[~mask] = np.nan
    
    # 计算归一化范围
    if vmax is None:
        vmax = np.nanpercentile(diff[mask], 95)  # 使用95分位数避免异常值
    
    # 归一化到 [0, 1]
    diff_norm = np.clip(diff / (vmax + 1e-8), 0, 1)
    
    # 应用 colormap
    heatmap = matplotlib.colormaps[cmap](diff_norm)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 无效区域设为黑色
    heatmap[~mask] = [0, 0, 0]
    
    # 保存（如果指定）
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # 保存为 BGR（OpenCV 格式）
        cv2.imwrite(str(save_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        
        # 同时保存带颜色条的版本
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(diff, cmap=cmap, vmin=0, vmax=vmax)
        mean_diff = np.nanmean(diff[mask])
        ax.set_title(f"{diff_label}\nMax: {vmax:.4f}, Mean: {mean_diff:.4f}")
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, label=diff_label)
        plt.tight_layout()
        
        colorbar_path = save_path.with_name(save_path.stem + "_with_colorbar.png")
        plt.savefig(colorbar_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return heatmap


def main():
    parser = argparse.ArgumentParser(
        description="可视化优化后的深度图"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="优化后的深度图目录（包含 *.npy 文件）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="输出目录（默认：input_dir 下的 depth_vis_color_{vis_range} 和 depth_vis_gray_{vis_range}）",
    )
    parser.add_argument(
        "--vis_range",
        type=str,
        default="100m",
        choices=["100m", "10m"],
        help="可视化范围（默认：100m）",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Spectral",
        help="Colormap 名称（默认：Spectral，可选：turbo, viridis, plasma 等）",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="最小深度值（米），如果为 None 则自动计算",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="最大深度值（米），如果为 None 则自动计算",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = input_dir.parent
    else:
        output_dir = Path(args.output_dir)
    
    output_color_dir = output_dir / f"depth_vis_color_{args.vis_range}"
    output_gray_dir = output_dir / f"depth_vis_gray_{args.vis_range}"
    output_color_dir.mkdir(parents=True, exist_ok=True)
    output_gray_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有深度图文件
    depth_files = sorted(input_dir.glob("*.npy"))
    if len(depth_files) == 0:
        print(f"⚠️  在 {input_dir} 中未找到 .npy 文件")
        return
    
    print(f"📦 找到 {len(depth_files)} 个深度图文件")
    print(f"   可视化范围: {args.vis_range}")
    print(f"   Colormap: {args.cmap}")
    print(f"   输出目录:")
    print(f"     彩色: {output_color_dir}")
    print(f"     灰度: {output_gray_dir}")
    
    # 处理每个深度图
    for depth_file in depth_files:
        pano_name = depth_file.stem.replace("_aligned", "")
        print(f"\n处理: {depth_file.name}")
        
        # 加载深度图
        depth = load_depth_npy(depth_file)
        
        # 转换为可视化图像
        depth_gray, depth_color = depth_to_vis(
            depth,
            vis_range=args.vis_range,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        
        # 保存灰度图
        gray_path = output_gray_dir / f"{pano_name}.png"
        cv2.imwrite(str(gray_path), depth_gray)
        print(f"  ✅ 灰度图: {gray_path}")
        
        # 保存彩色图
        color_path = output_color_dir / f"{pano_name}.png"
        cv2.imwrite(str(color_path), cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))
        print(f"  ✅ 彩色图: {color_path}")
    
    print(f"\n✅ 全部完成！")
    print(f"   处理了 {len(depth_files)} 个深度图")


if __name__ == "__main__":
    main()
