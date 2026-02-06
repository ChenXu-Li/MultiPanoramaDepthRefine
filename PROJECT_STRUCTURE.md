# MultiPanoramaDepthRefine 项目目录结构

## 设计原则

1. **数据在项目外**：所有输入数据（RGB、深度、COLMAP）通过配置文件指定绝对路径
2. **中间文件在项目内**：所有中间结果、输出、日志保存在项目目录下
3. **模块化设计**：每个功能模块独立，便于测试和维护

## 目录结构

```
MultiPanoramaDepthRefine/
├── README.md                          # 项目说明文档
├── CURSORREAD.md                      # Cursor 实现指南
├── PROJECT_STRUCTURE.md               # 本文件：目录结构说明
│
├── configs/                            # 配置文件目录
│   ├── default.yaml                   # 默认配置（示例）
│   ├── bridgeb.yaml                   # BridgeB 场景配置示例
│   └── roottop.yaml                   # RoofTop 场景配置示例
│
├── src/                                # 源代码目录
│   ├── __init__.py
│   │
│   ├── camera/                         # 相机模型
│   │   ├── __init__.py
│   │   ├── spherical_camera.py         # 球面相机模型（equirectangular）
│   │   └── coordinate_transform.py     # 坐标变换工具
│   │
│   ├── deformation/                    # 深度变形模型（核心）
│   │   ├── __init__.py
│   │   ├── monotonic_spline.py        # 全局单调 spline 映射
│   │   ├── directional_scale.py       # 方向相关缩放（SH/B-spline）
│   │   └── depth_reparam.py           # 深度重参数化主模块
│   │
│   ├── geometry/                       # 几何一致性约束
│   │   ├── __init__.py
│   │   ├── point_to_ray.py            # Point-to-Ray 距离
│   │   ├── ray_space_consistency.py   # Ray-space 深度一致性
│   │   └── multi_view_loss.py         # 多视角几何 loss 组合
│   │
│   ├── regularization/                 # 防退化护栏
│   │   ├── __init__.py
│   │   ├── prior_anchor.py            # log-depth 先验锚点
│   │   ├── smoothness.py               # 球面平滑正则
│   │   ├── far_field_mask.py          # 远景处理（≥100m）
│   │   └── scale_guard.py             # 尺度冻结/约束
│   │
│   ├── solver/                         # 优化求解器
│   │   ├── __init__.py
│   │   ├── optimizer.py               # PyTorch 优化器封装
│   │   ├── energy_function.py         # 总能量函数
│   │   └── convergence.py             # 收敛检测
│   │
│   ├── utils/                          # 工具函数
│   │   ├── __init__.py
│   │   ├── io.py                      # 文件 I/O（加载 RGB、深度、COLMAP）
│   │   ├── visualize.py               # 可视化工具
│   │   ├── geometry.py                 # 几何工具函数
│   │   └── config.py                   # 配置加载和验证
│   │
│   └── run_multi_view.py               # 主入口脚本
│
├── intermediate/                       # 中间结果目录（项目内）
│   ├── step0_initialization/           # Step 0: 初始化和验证
│   │   ├── {pano_name}_initial_depth.npy
│   │   └── {pano_name}_validation_report.json
│   │
│   ├── step1_deformation/             # Step 1: 深度变形模型初始化
│   │   ├── {pano_name}_spline_params.npy
│   │   ├── {pano_name}_scale_params.npy
│   │   └── {pano_name}_deformed_depth.npy
│   │
│   ├── step2_geometry/                 # Step 2: 几何一致性优化
│   │   ├── {pano_name}_geometry_loss_history.json
│   │   └── {pano_name}_geometry_iter_{iter}.npy
│   │
│   ├── step3_regularization/           # Step 3: 防退化护栏
│   │   ├── {pano_name}_prior_loss_history.json
│   │   └── {pano_name}_smooth_loss_history.json
│   │
│   └── step4_joint_optimization/        # Step 4: 联合优化
│       ├── {pano_name}_final_depth.npy
│       ├── {pano_name}_energy_history.json
│       └── {pano_name}_convergence_report.json
│
├── outputs/                            # 最终输出目录（项目内）
│   ├── aligned_depths/                 # 对齐后的深度图
│   │   └── {pano_name}_aligned_depth.npy
│   │
│   ├── pointclouds/                    # 一致的点云（可选）
│   │   └── {scene_name}_aligned.ply
│   │
│   └── visualizations/                 # 可视化结果
│       ├── {pano_name}_depth_colormap.png
│       ├── {pano_name}_deformation_vis.png
│       └── {scene_name}_multi_view_comparison.png
│
├── logs/                               # 日志目录（项目内）
│   ├── {scene_name}_{timestamp}.log
│   └── {scene_name}_error.log
│
├── tests/                              # 单元测试
│   ├── __init__.py
│   ├── test_deformation.py             # 变形模型测试
│   ├── test_geometry.py                 # 几何一致性测试
│   ├── test_regularization.py          # 防退化测试
│   └── test_integration.py              # 集成测试
│
└── scripts/                            # 辅助脚本
    ├── batch_process.py                # 批量处理脚本
    ├── visualize_results.py            # 结果可视化脚本
    └── validate_outputs.py              # 输出验证脚本
```

## 目录说明

### `configs/`
- 存放所有配置文件（YAML 格式）
- 每个场景可以有自己的配置文件
- 配置文件指定**绝对路径**指向项目外的数据目录

### `src/`
- 源代码目录，按功能模块组织
- `camera/`: 相机模型和坐标变换
- `deformation/`: 深度变形模型（核心算法）
- `geometry/`: 多视角几何一致性约束
- `regularization/`: 防退化护栏
- `solver/`: 优化求解器
- `utils/`: 工具函数

### `intermediate/`
- **项目内目录**，保存所有中间结果
- 按步骤组织，便于调试和检查
- 每个步骤的结果都有时间戳或版本号

### `outputs/`
- **项目内目录**，保存最终输出
- `aligned_depths/`: 对齐后的深度图
- `pointclouds/`: 一致的点云（可选）
- `visualizations/`: 可视化结果

### `logs/`
- **项目内目录**，保存运行日志
- 按场景和时间戳组织

### `tests/`
- 单元测试和集成测试
- 每个模块都有对应的测试文件

### `scripts/`
- 辅助脚本，用于批量处理、可视化等

## 数据路径约定

### 输入数据（项目外，通过配置指定）

```yaml
paths:
  # COLMAP 重建目录（包含各场景的 sparse/0）
  colmap_root: /root/autodl-tmp/data/colmap_STAGE1_4x
  
  # STAGE 数据集根目录（包含各场景）
  stage_root: /root/autodl-tmp/data/STAGE1_4x
  
  # 场景名称（如 BridgeB, RoofTop）
  scene_name: BridgeB
  
  # 全景图名称列表（如 ["point2_median", "point3_median", ...]）
  pano_names:
    - point2_median
    - point3_median
    - point4_median
    - point5_median
```

### 中间文件（项目内）

所有中间文件保存在 `intermediate/` 目录下，路径由代码自动生成，不依赖配置。

### 输出文件（项目内）

所有输出文件保存在 `outputs/` 目录下，路径由代码自动生成，不依赖配置。

## 文件命名约定

### 中间文件
- 格式：`{pano_name}_{step}_{description}.{ext}`
- 示例：`point2_median_step1_deformed_depth.npy`

### 输出文件
- 格式：`{pano_name}_{description}.{ext}`
- 示例：`point2_median_aligned_depth.npy`

### 日志文件
- 格式：`{scene_name}_{timestamp}.log`
- 示例：`BridgeB_20240101_120000.log`

## 注意事项

1. **绝对路径**：配置文件中所有数据路径必须是绝对路径
2. **项目内目录**：`intermediate/`, `outputs/`, `logs/` 都在项目内，由代码自动创建
3. **版本控制**：`.gitignore` 应忽略 `intermediate/`, `outputs/`, `logs/` 目录
4. **可移植性**：代码不依赖硬编码路径，所有路径通过配置读取
