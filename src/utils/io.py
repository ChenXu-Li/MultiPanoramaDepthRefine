"""
I/O 工具函数
"""
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional


def load_image(image_path: Path | str) -> np.ndarray:
    """
    加载 RGB 图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        rgb: (H, W, 3) uint8 RGB 图像
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    img = Image.open(image_path)
    rgb = np.array(img, dtype=np.uint8)
    
    # 如果是灰度图，转换为 RGB
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        # RGBA -> RGB
        rgb = rgb[:, :, :3]
    
    return rgb


def load_depth_npy(depth_path: Path | str) -> np.ndarray:
    """
    加载深度图（.npy 格式）
    
    Args:
        depth_path: 深度图路径
        
    Returns:
        depth: (H, W) float32 深度图
    """
    depth_path = Path(depth_path)
    if not depth_path.exists():
        raise FileNotFoundError(f"深度文件不存在: {depth_path}")
    
    depth = np.load(depth_path).astype(np.float32)
    
    # 确保是 2D
    if depth.ndim != 2:
        raise ValueError(f"深度图应为 2D，当前形状: {depth.shape}")
    
    return depth


def save_depth_npy(depth: np.ndarray, output_path: Path | str):
    """
    保存深度图（.npy 格式）
    
    Args:
        depth: (H, W) 深度图
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    depth_float32 = depth.astype(np.float32)
    np.save(output_path, depth_float32)


def save_image(image: np.ndarray, output_path: Path | str):
    """
    保存图像
    
    Args:
        image: (H, W, 3) uint8 图像或 (H, W) 灰度图
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    img.save(output_path)
