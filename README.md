# MultiPanoramaDepthRefine

## Panoramic Multi-view Depth Alignment with Smooth Direction-aware Deformation

多视角全景深度对齐项目，通过结构化深度变形模型实现多视角几何一致性。

## 项目结构

```
MultiPanoramaDepthRefine/
├── configs/              # 配置文件（YAML）
├── src/                  # 源代码
├── intermediate/         # 中间结果（项目内）
├── outputs/              # 最终输出（项目内）
├── logs/                 # 日志文件（项目内）
├── tests/                # 单元测试
└── scripts/              # 辅助脚本
```

**重要**：
- 数据在项目外，通过配置文件指定绝对路径
- 中间文件和输出保存在项目内，由代码自动管理

详细目录结构请参考 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 快速开始

### 1. 准备配置文件

复制并修改场景特定配置：

```bash
cp configs/default.yaml configs/my_scene.yaml
vim configs/my_scene.yaml  # 修改数据路径和场景名称
```

### 2. 运行多视角优化

```bash
python src/run_multi_view.py --config configs/my_scene.yaml
```

### 3. 查看结果

- 对齐后的深度图：`outputs/aligned_depths/`
- 可视化结果：`outputs/visualizations/`
- 中间结果：`intermediate/`

## 配置文件说明

详细配置说明请参考 [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

### 关键配置项

```yaml
paths:
  colmap_root: /path/to/colmap_STAGE1_4x    # COLMAP 重建根目录
  stage_root: /path/to/STAGE1_4x             # STAGE 数据集根目录
  scene_name: BridgeB                         # 场景名称
  pano_names:                                 # 参与优化的全景图列表
    - point2_median
    - point3_median
    - point4_median
    - point5_median
```

## 核心算法

本项目采用结构化深度变形模型：

1. **全局单调 Spline**：修正整体尺度和非线性压缩
2. **方向相关缩放**：捕捉全景方向相关的系统性畸变
3. **多视角几何一致性**：Point-to-Ray 距离 + Ray-space 深度一致性
4. **防退化护栏**：log-depth 先验、平滑正则、尺度约束

详细算法说明请参考 [README.md](README.md) 和 [CURSORREAD.md](CURSORREAD.md)

## 设计原则

- **不优化像素，只优化结构**：使用结构化变形模型而非 per-pixel 优化
- **不相信远景几何**：远景（≥100m）不参与几何对齐
- **方向一致性优先**：Point-to-Ray 权重 >> 深度一致性权重
- **防退化必须存在**：先验锚点、平滑正则、尺度约束缺一不可

## 开发指南

实现指南请参考 [CURSORREAD.md](CURSORREAD.md)

## 许可证

[待定]
