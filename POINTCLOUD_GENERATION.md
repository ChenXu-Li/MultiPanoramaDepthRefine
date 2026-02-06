# 点云生成指南

## 概述

本指南介绍如何从优化后的深度图生成二进制 PLY 点云文件。

## 快速开始

### 方法 1: 使用便捷脚本（推荐）

```bash
cd /root/autodl-tmp/code/MultiPanoramaDepthRefine
bash scripts/generate_bridgeb_pointclouds.sh
```

### 方法 2: 使用 Python 脚本

```bash
cd /root/autodl-tmp/code/MultiPanoramaDepthRefine

python scripts/generate_pointclouds.py \
  --depth_dir outputs/aligned_depths/BridgeB \
  --rgb_dir /root/autodl-tmp/data/STAGE1_4x/BridgeB/backgrounds \
  --output_dirs \
    outputs/pointclouds \
    /root/autodl-tmp/data/STAGE1_4x/BridgeB/pointclouds_mutil_opt \
  --verbose
```

## 参数说明

### `--depth_dir`
优化后的深度图目录（包含 `*.npy` 文件）

### `--rgb_dir`
RGB 图像目录（用于点云颜色）

### `--output_dirs`
输出目录列表（可以指定多个目录，点云会保存到所有目录）

### `--scene_name`（可选）
场景名称，默认 "BridgeB"

### `--verbose`（可选）
输出详细日志

## 输出文件

### BridgeB 场景

**输出目录 1**: `/root/autodl-tmp/code/MultiPanoramaDepthRefine/outputs/pointclouds/`
- `point2_median.ply` (27MB)
- `point3_median.ply` (27MB)
- `point4_median.ply` (27MB)
- `point5_median.ply` (27MB)

**输出目录 2**: `/root/autodl-tmp/data/STAGE1_4x/BridgeB/pointclouds_mutil_opt/`
- `point2_median.ply` (27MB)
- `point3_median.ply` (27MB)
- `point4_median.ply` (27MB)
- `point5_median.ply` (27MB)

## 点云规格

- **格式**: 二进制 PLY（binary PLY）
- **点数**: 1,843,200 点/文件（960×1920 像素）
- **属性**: x, y, z, red, green, blue
- **坐标系**: 相机坐标系（与 DAP 一致）
- **单位**: 米

## 技术细节

### Equirectangular 参数化

使用与 DAP 相同的球面坐标约定：

```
theta = (1 - u) * 2*pi  # [0, 2*pi]
phi   = v * pi          # [0, pi]

方向向量：
  x = sin(phi) * cos(theta)
  y = sin(phi) * sin(theta)
  z = cos(phi)

点云：
  p = depth * dir
```

### 点云生成流程

1. 读取深度图（.npy，float32，单位：米）
2. 读取 RGB 图像（.png 或 .jpg）
3. 生成 UV 坐标网格
4. 计算球面方向向量
5. 计算 3D 点云坐标
6. 保存为二进制 PLY 格式

## 示例

### 示例 1: 生成 BridgeB 场景点云

```bash
python scripts/generate_pointclouds.py \
  --depth_dir outputs/aligned_depths/BridgeB \
  --rgb_dir /root/autodl-tmp/data/STAGE1_4x/BridgeB/backgrounds \
  --output_dirs outputs/pointclouds \
  --verbose
```

### 示例 2: 生成多个场景的点云

```bash
# BridgeB
python scripts/generate_pointclouds.py \
  --depth_dir outputs/aligned_depths/BridgeB \
  --rgb_dir /root/autodl-tmp/data/STAGE1_4x/BridgeB/backgrounds \
  --output_dirs outputs/pointclouds

# OtherScene
python scripts/generate_pointclouds.py \
  --depth_dir outputs/aligned_depths/OtherScene \
  --rgb_dir /root/autodl-tmp/data/STAGE1_4x/OtherScene/backgrounds \
  --output_dirs outputs/pointclouds
```

## 验证点云

可以使用以下 Python 代码验证点云文件：

```python
from plyfile import PlyData
import numpy as np

ply = PlyData.read('outputs/pointclouds/point2_median.ply')
print(f"格式: {'binary' if not ply.text else 'ascii'}")
print(f"点数: {len(ply['vertex'])}")
print(f"属性: {ply['vertex'].data.dtype.names}")

# 获取点云数据
points = np.array([
    [v['x'], v['y'], v['z']] 
    for v in ply['vertex']
])
colors = np.array([
    [v['red'], v['green'], v['blue']] 
    for v in ply['vertex']
])

print(f"点云形状: {points.shape}")
print(f"颜色形状: {colors.shape}")
print(f"点云范围: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
      f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
      f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
```

## 参考

- DAP 项目: `code/DAP/batch_generate_pointclouds.py`
- PanoramaDepthRefine: `code/PanoramaDepthRefine/single_opt.py` (depth_to_pointcloud_ply)
