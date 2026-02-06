# 配置文件使用指南

## 配置文件结构

本项目使用 YAML 格式的配置文件，所有配置文件位于 `configs/` 目录下。

## 配置文件层次

1. **`default.yaml`**: 默认配置模板，包含所有可配置项及其说明
2. **场景特定配置**: 如 `bridgeb.yaml`, `roottop.yaml`，针对特定场景的配置

## 配置项说明

### 1. 数据路径配置 (`paths`)

**重要**：所有路径必须是**绝对路径**，数据在项目外。

```yaml
paths:
  colmap_root: /root/autodl-tmp/data/colmap_STAGE1_4x  # COLMAP 重建根目录
  stage_root: /root/autodl-tmp/data/STAGE1_4x          # STAGE 数据集根目录
  scene_name: BridgeB                                   # 场景名称
  pano_names:                                           # 参与优化的全景图列表
    - point2_median
    - point3_median
    - point4_median
    - point5_median
  camera_name: pano_camera12                            # 相机名称子串
```

**路径推断规则**：
- RGB 图像：`{stage_root}/{scene_name}/backgrounds/{pano_name}.png`
- DAP 深度：`{stage_root}/{scene_name}/depth_npy/{pano_name}.npy`
- COLMAP 模型：`{colmap_root}/{scene_name}/sparse/0`

### 2. 输出配置 (`output`)

中间文件和输出文件保存在**项目内**，由代码自动管理。

```yaml
output:
  project_root: null  # null 表示使用配置文件所在目录的父目录
  save_intermediate: true      # 是否保存中间结果
  save_aligned_depths: true    # 是否保存对齐后的深度图
  generate_pointcloud: false   # 是否生成一致的点云
  save_visualizations: true     # 是否保存可视化结果
  viz_format: png               # 可视化格式
```

**输出目录结构**（自动创建）：
- `intermediate/`: 中间结果
- `outputs/aligned_depths/`: 对齐后的深度图
- `outputs/pointclouds/`: 一致的点云（如果启用）
- `outputs/visualizations/`: 可视化结果

### 3. 深度变形模型配置 (`deformation`)

#### 全局单调 Spline

```yaml
deformation:
  monotonic_spline:
    type: monotonic_cubic        # 'monotonic_cubic' 或 'linear'
    num_knots: 10                # 控制点数量
    log_depth_min: -3.0          # log-depth 最小值
    log_depth_max: 5.0           # log-depth 最大值
    freeze_reference_point: true # 是否冻结参考点（防尺度漂移）
    reference_log_depth: 0.0     # 参考点位置（log(1.0) = 0）
```

#### 方向相关缩放

```yaml
  directional_scale:
    method: spherical_harmonics  # 'spherical_harmonics' 或 'bspline_grid'
    sh_max_degree: 4             # 球谐最大阶数（仅 SH）
    bspline_grid_resolution: [16, 8]  # B-spline 网格分辨率（仅 B-spline）
    max_scale_log: 0.3            # 缩放幅度限制（log 空间）
    scale_regularization_weight: 0.01  # L2 正则化权重
```

### 4. 几何一致性配置 (`geometry`)

#### Point-to-Ray 距离（主约束）

```yaml
geometry:
  point_to_ray:
    enabled: true
    weight: 1.0                   # 权重（必须 >> depth_consistency）
    sampling: all_pixels          # 'all_pixels' 或 'sparse_sample'
    sparse_sample_rate: 0.1       # 稀疏采样率（仅当 sampling=sparse_sample）
    use_robust_loss: true         # 是否使用 Huber loss
    huber_delta: 0.1              # Huber delta（米）
```

#### Ray-space 深度一致性（弱约束）

```yaml
  depth_consistency:
    enabled: true
    weight: 0.1                   # 权重（必须 << point_to_ray）
    use_robust_loss: false
```

#### 远景处理

```yaml
  far_field:
    far_threshold: 100.0          # 远景阈值（米）
    apply_smoothness_to_far: true  # 远景是否参与平滑项
```

### 5. 防退化护栏配置 (`regularization`)

#### log-depth 先验锚点

```yaml
regularization:
  prior_anchor:
    enabled: true                 # 必须启用，防止整体塌缩
    weight: 1.0                  # 权重必须非零
    loss_type: l2                # 'l2' 或 'huber'
    huber_delta: 0.1
```

#### 球面平滑正则

```yaml
  smoothness:
    enabled: true
    weight: 0.01
    smooth_type: l2              # 'l2' 或 'l1'
    edge_aware: false            # 是否使用边缘感知平滑
    rgb_sigma: 10.0              # RGB 边缘敏感度（仅当 edge_aware=true）
```

#### 方向变形约束

```yaml
  scale_constraint:
    enabled: true
    weight: 0.01                 # L2 正则化权重
```

### 6. 优化配置 (`optimization`)

#### 优化器配置

```yaml
optimization:
  solver:
    optimizer: adam              # 'adam' 或 'sgd'
    lr: 1e-3                    # 学习率
    adam_beta1: 0.9             # Adam 参数
    adam_beta2: 0.999
    adam_eps: 1e-8
    sgd_momentum: 0.9           # SGD 动量（仅当 optimizer=sgd）
```

#### 迭代配置

```yaml
  iteration:
    max_iter: 1000               # 最大迭代次数
    early_stop_threshold: 1e-6   # 提前停止阈值
    save_interval: 100           # 保存间隔
    print_interval: 10           # 打印间隔
```

#### 设备配置

```yaml
  device:
    device: cuda                 # 'cuda' 或 'cpu'
    gpu_id: 0                    # GPU 设备 ID（仅当 device=cuda）
```

### 7. 验证配置 (`validation`)

```yaml
validation:
  validate_before_optimization: true
  validate_after_optimization: true
  checks:
    check_monotonicity: true           # 检查单调性
    check_continuity: true             # 检查连续性
    check_scale_collapse: true          # 检查尺度退化
    check_far_field_stability: true    # 检查远景稳定性
  thresholds:
    monotonicity_violation_rate: 0.01
    continuity_gradient_threshold: 1.0
    scale_collapse_std_threshold: 0.1
```

## 使用示例

### 1. 创建场景特定配置

```bash
# 复制默认配置
cp configs/default.yaml configs/my_scene.yaml

# 编辑场景特定配置
vim configs/my_scene.yaml
```

### 2. 在代码中加载配置

```python
from pathlib import Path
from src.utils.config import load_config, validate_config

# 加载配置
config_path = Path("configs/bridgeb.yaml")
config = load_config(config_path)

# 验证配置
is_valid, errors = validate_config(config)
if not is_valid:
    print("配置错误：", errors)
    exit(1)
```

### 3. 命令行使用

```bash
# 使用指定配置文件运行
python src/run_multi_view.py --config configs/bridgeb.yaml

# 或使用默认配置
python src/run_multi_view.py --config configs/default.yaml
```

## 配置验证规则

配置文件加载时会自动验证以下内容：

1. **路径存在性**：检查所有数据路径是否存在
2. **权重合理性**：
   - `lambda_p2r > lambda_depth`（几何一致性）
   - `lambda_prior > 0`（防塌缩）
3. **数值范围**：
   - 学习率在 `(0, 1]` 范围内
   - 深度范围 `depth_min < depth_max` 且 `depth_min > 0`
4. **配置一致性**：
   - 如果 `method=spherical_harmonics`，则 `sh_max_degree` 必须设置
   - 如果 `method=bspline_grid`，则 `bspline_grid_resolution` 必须设置

## 常见问题

### Q: 如何添加新的全景图？

A: 在 `paths.pano_names` 列表中添加新的 pano 名称：

```yaml
paths:
  pano_names:
    - point2_median
    - point3_median
    - point4_median
    - point5_median
    - point6_median  # 新增
```

### Q: 如何调整优化权重？

A: 修改 `geometry` 和 `regularization` 部分的权重：

```yaml
geometry:
  point_to_ray:
    weight: 2.0  # 增大主约束权重

regularization:
  prior_anchor:
    weight: 1.5  # 增大防塌缩权重
```

### Q: 如何禁用某个功能？

A: 将对应项的 `enabled` 设置为 `false`：

```yaml
geometry:
  depth_consistency:
    enabled: false  # 禁用深度一致性约束
```

### Q: 如何保存更多中间结果？

A: 启用调试模式：

```yaml
debug:
  enabled: true
  save_iterations: true  # 保存每次迭代的结果
```

## 最佳实践

1. **为每个场景创建独立配置**：不要修改 `default.yaml`，而是创建场景特定配置
2. **使用版本控制**：将配置文件提交到 Git，但忽略 `intermediate/` 和 `outputs/`
3. **文档化修改**：在配置文件中添加注释说明为什么修改某个参数
4. **逐步调参**：先使用默认配置，然后逐步调整权重和参数
