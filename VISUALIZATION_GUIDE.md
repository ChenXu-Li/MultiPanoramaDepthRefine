# 深度图可视化指南

## 概述

本指南介绍如何可视化优化后的深度图，参考 DAP 项目的可视化方法。

## 快速开始

### 方法 1: 使用便捷脚本（推荐）

```bash
cd /root/autodl-tmp/code/MultiPanoramaDepthRefine
bash scripts/visualize_bridgeb.sh
```

这会生成：
- 100m 范围的可视化（适合查看整体深度分布）
- 10m 范围的可视化（适合查看近景细节）

### 方法 2: 使用 Python 脚本

```bash
cd /root/autodl-tmp/code/MultiPanoramaDepthRefine

# 生成 100m 范围的可视化
python scripts/visualize_depths.py \
  --input_dir outputs/aligned_depths/BridgeB \
  --vis_range 100m \
  --cmap Spectral

# 生成 10m 范围的可视化
python scripts/visualize_depths.py \
  --input_dir outputs/aligned_depths/BridgeB \
  --vis_range 10m \
  --cmap Spectral
```

## 参数说明

### `--input_dir`
优化后的深度图目录（包含 `*.npy` 文件）

### `--output_dir`（可选）
输出目录，默认为 `input_dir` 的父目录

### `--vis_range`
可视化范围：
- `100m`: 将深度范围映射到 [0, 100] 米（适合查看整体场景）
- `10m`: 将深度范围映射到 [0, 10] 米（适合查看近景细节）

### `--cmap`
Colormap 名称，可选：
- `Spectral`（默认）：彩虹色，从蓝到红
- `turbo`: 现代彩虹色，更好的感知均匀性
- `viridis`: 蓝绿色，适合科学可视化
- `plasma`: 紫红色，高对比度
- 其他 matplotlib colormap 名称

### `--vmin` / `--vmax`（可选）
手动指定深度范围（米），如果不指定则自动计算

## 输出文件

脚本会在输出目录下创建以下目录结构：

```
outputs/aligned_depths/
├── depth_vis_color_100m/    # 100m 范围彩色可视化
│   ├── point2_median.png
│   ├── point3_median.png
│   └── ...
├── depth_vis_gray_100m/     # 100m 范围灰度可视化
│   ├── point2_median.png
│   └── ...
├── depth_vis_color_10m/     # 10m 范围彩色可视化
│   └── ...
└── depth_vis_gray_10m/      # 10m 范围灰度可视化
    └── ...
```

## 示例

### 示例 1: 可视化 BridgeB 场景

```bash
python scripts/visualize_depths.py \
  --input_dir outputs/aligned_depths/BridgeB \
  --vis_range 100m \
  --cmap turbo
```

### 示例 2: 自定义深度范围

```bash
python scripts/visualize_depths.py \
  --input_dir outputs/aligned_depths/BridgeB \
  --vis_range 10m \
  --cmap Spectral \
  --vmin 0.5 \
  --vmax 50.0
```

### 示例 3: 可视化其他场景

```bash
python scripts/visualize_depths.py \
  --input_dir outputs/aligned_depths/OtherScene \
  --output_dir outputs/aligned_depths \
  --vis_range 100m \
  --cmap viridis
```

## 可视化效果说明

### 100m 范围
- **用途**: 查看整体场景的深度分布
- **特点**: 覆盖 0-100 米的深度范围
- **适用**: 大场景、远景分析

### 10m 范围
- **用途**: 查看近景细节
- **特点**: 覆盖 0-10 米的深度范围，细节更清晰
- **适用**: 前景物体、细节分析

### Colormap 选择建议
- **Spectral**: 通用选择，对比度高
- **turbo**: 现代风格，感知均匀性好
- **viridis**: 适合科学可视化，色盲友好
- **plasma**: 高对比度，适合打印

## 参考

- DAP 项目: `code/DAP/test/infer_pics.py`
- DAP 脚本: `code/DAP/infer_pics.sh`
