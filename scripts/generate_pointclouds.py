#!/usr/bin/env python3
"""
从优化后的深度图生成二进制 PLY 点云
参考 DAP 的 batch_generate_pointclouds.py
"""
import sys
from pathlib import Path
import numpy as np
import argparse
import cv2
from plyfile import PlyData, PlyElement

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import load_depth_npy, load_image


def spherical_uv_to_directions(uv: np.ndarray):
    """
    将 UV 坐标转换为球面方向向量（DAP 约定）
    
    Args:
        uv: (H, W, 2) UV 坐标，u 和 v 都在 [0, 1]
        
    Returns:
        directions: (H, W, 3) 单位方向向量
    """
    u, v = uv[..., 0], uv[..., 1]
    
    # DAP 约定：theta/phi
    theta = (1.0 - u) * (2.0 * np.pi)  # [0, 2*pi]
    phi = v * np.pi                     # [0, pi]
    
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # 单位方向向量
    directions = np.stack([
        sin_phi * cos_theta,  # x
        sin_phi * sin_theta,  # y
        cos_phi               # z
    ], axis=-1)
    
    return directions


def image_uv(width: int, height: int) -> np.ndarray:
    """
    生成等轴柱状图的 UV 坐标网格
    
    Args:
        width: 图像宽度
        height: 图像高度
        
    Returns:
        uv: (H, W, 2) UV 坐标，u 和 v 都在 [0, 1]
    """
    u = np.linspace(0, 1, width, dtype=np.float32)   # [W]
    v = np.linspace(0, 1, height, dtype=np.float32)  # [H]
    u_grid, v_grid = np.meshgrid(u, v)  # [H, W]
    uv = np.stack([u_grid, v_grid], axis=-1)  # [H, W, 2]
    return uv


def save_3d_points_binary(
    points: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    filename: str,
):
    """
    保存3D点云到二进制 PLY 文件
    
    Args:
        points: 3D点 (H, W, 3) 或 (N, 3)
        colors: 颜色 (H, W, 3) 或 (N, 3)，RGB uint8
        mask: 有效点mask (H, W) 或 (N,)
        filename: 输出PLY路径
    """
    # 重塑为 (N, 3)
    if points.ndim == 3:
        points = points.reshape(-1, 3)
    if colors.ndim == 3:
        colors = colors.reshape(-1, 3)
    if mask.ndim == 2:
        mask = mask.reshape(-1)
    
    # 只处理有效点
    valid_points = points[mask]
    valid_colors = colors[mask]
    
    # 创建结构化数组
    vertex_data = np.empty(len(valid_points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = valid_points[:, 0]
    vertex_data['y'] = valid_points[:, 1]
    vertex_data['z'] = valid_points[:, 2]
    vertex_data['red'] = valid_colors[:, 0]
    vertex_data['green'] = valid_colors[:, 1]
    vertex_data['blue'] = valid_colors[:, 2]
    
    # 保存为二进制 PLY
    vertex_element = PlyElement.describe(vertex_data, 'vertex', comments=['point cloud'])
    PlyData([vertex_element], text=False).write(filename)


def depth_to_pointcloud_ply(
    depth_path: Path,
    image_path: Path,
    out_ply: Path,
    verbose: bool = True,
):
    """
    将等轴柱状深度图 + RGB 转换为点云，并保存为二进制 PLY
    
    使用与 DAP 相同的 equirect 参数化（theta/phi）：
        theta = (1 - u) * 2*pi  # [0, 2*pi]
        phi   = v * pi          # [0, pi]
    方向向量：
        x = sin(phi) * cos(theta)
        y = sin(phi) * sin(theta)
        z = cos(phi)
    点云：
        p = depth * dir
    
    Args:
        depth_path: 深度图 .npy 文件路径（float32，单位：米）
        image_path: RGB 图像路径
        out_ply: 输出 PLY 文件路径
        verbose: 是否输出详细日志
    """
    # 读取深度图
    if verbose:
        print(f"  📖 读取深度图: {depth_path}")
    depth = load_depth_npy(depth_path)
    H, W = depth.shape
    
    if verbose:
        print(f"     深度图形状: {H}x{W} (总点数: {H*W:,})")
        print(f"     深度范围: [{depth.min():.2f}, {depth.max():.2f}] 米")
    
    # 读取 RGB 图像
    if verbose:
        print(f"  🖼️  读取 RGB 图像: {image_path}")
    rgb = load_image(image_path)
    
    if rgb.shape[0] != H or rgb.shape[1] != W:
        if verbose:
            print(f"     ⚠️  RGB 尺寸不匹配，调整 RGB: {rgb.shape[:2]} -> {(H, W)}")
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # 有效像素 mask
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        print("     ⚠️  深度图中没有有效像素，跳过点云导出")
        return
    
    if verbose:
        print(f"     有效像素: {valid.sum():,} / {H*W:,} ({100*valid.sum()/(H*W):.1f}%)")
    
    # 生成 UV 坐标
    if verbose:
        print(f"  🔄 生成 UV 坐标...")
    uv = image_uv(width=W, height=H)  # (H, W, 2)
    
    # 计算方向向量
    if verbose:
        print(f"  📐 计算方向向量...")
    dirs = spherical_uv_to_directions(uv)  # (H, W, 3)
    
    # 计算 3D 点云
    if verbose:
        print(f"  💎 计算 3D 点云...")
    points = depth[..., None] * dirs  # (H, W, 3)
    
    # 确保颜色是 uint8
    colors = rgb.astype(np.uint8)
    
    # 保存为二进制 PLY
    if verbose:
        print(f"  💾 保存点云: {out_ply}")
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    save_3d_points_binary(points, colors, valid, str(out_ply))
    
    if verbose:
        valid_count = valid.sum()
        file_size = out_ply.stat().st_size / (1024**2)  # MB
        print(f"     ✅ 完成！点数: {valid_count:,}, 文件大小: {file_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="从优化后的深度图生成二进制 PLY 点云"
    )
    parser.add_argument(
        "--depth_dir",
        type=Path,
        required=True,
        help="优化后的深度图目录（包含 *.npy 文件）",
    )
    parser.add_argument(
        "--rgb_dir",
        type=Path,
        required=True,
        help="RGB 图像目录",
    )
    parser.add_argument(
        "--output_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="输出目录列表（可以指定多个）",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default="BridgeB",
        help="场景名称（用于查找 RGB 图像）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细日志",
    )
    
    args = parser.parse_args()
    
    depth_dir = Path(args.depth_dir)
    rgb_dir = Path(args.rgb_dir)
    
    if not depth_dir.exists():
        print(f"❌ 深度图目录不存在: {depth_dir}")
        return
    
    if not rgb_dir.exists():
        print(f"❌ RGB 图像目录不存在: {rgb_dir}")
        return
    
    # 查找所有深度图文件
    depth_files = sorted(depth_dir.glob("*.npy"))
    if len(depth_files) == 0:
        print(f"⚠️  在 {depth_dir} 中未找到 .npy 文件")
        return
    
    print(f"📦 找到 {len(depth_files)} 个深度图文件")
    print(f"   深度图目录: {depth_dir}")
    print(f"   RGB 图像目录: {rgb_dir}")
    print(f"   输出目录: {args.output_dirs}")
    
    # 处理每个深度图
    success_count = 0
    fail_count = 0
    
    for depth_file in depth_files:
        # 从文件名提取 pano_name（去掉 _aligned 后缀）
        pano_name = depth_file.stem.replace("_aligned", "")
        
        # 查找对应的 RGB 图像
        rgb_candidates = [
            rgb_dir / f"{pano_name}.png",
            rgb_dir / f"{pano_name}.jpg",
        ]
        rgb_path = None
        for candidate in rgb_candidates:
            if candidate.exists():
                rgb_path = candidate
                break
        
        if rgb_path is None:
            print(f"⚠️  跳过 {depth_file.name}: 未找到对应的 RGB 图像")
            fail_count += 1
            continue
        
        print(f"\n处理: {depth_file.name}")
        
        # 为每个输出目录生成点云
        for output_dir in args.output_dirs:
            output_dir = Path(output_dir)
            out_ply = output_dir / f"{pano_name}.ply"
            
            try:
                depth_to_pointcloud_ply(
                    depth_path=depth_file,
                    image_path=rgb_path,
                    out_ply=out_ply,
                    verbose=args.verbose,
                )
                success_count += 1
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1
    
    print(f"\n✅ 全部完成！")
    print(f"   成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    main()
