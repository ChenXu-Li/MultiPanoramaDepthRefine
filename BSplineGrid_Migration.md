# B-spline Grid 迁移完成报告

## 概述

已成功将球谐函数（SH）方向相关修正模型替换为**方向 × log-depth 的 B-spline grid 表示**。

## 核心改动

### 1. 新增模块

#### `src/deformation/directional_bspline_grid.py`
- **`DirectionalBSplineGrid`**: 方向 × log-depth B-spline grid 模块
- **方向变量**: `alpha = asin(ray_dir.y)` 或 `ray_dir.y`（不是完整球面）
- **Grid 维度**: `axis 0 = alpha`, `axis 1 = log_depth`
- **B-spline 阶数**: cubic (order=3)
- **前向计算**: `d' = exp(log(d) + Δ(alpha, log(d)))`

#### `src/regularization/bspline_constraints.py`
- **单调性约束**: 沿 log-depth 轴确保 `∂d'/∂d > 0`
- **方向平滑正则**: 防止相邻方向抖动
- **远景渐近约束**: `lim_{d→∞} Δ(alpha, d) → 0`

### 2. 修改模块

#### `src/deformation/depth_reparam.py`
- 添加 `use_directional_bspline` 参数（默认 `True`）
- 新版本：直接使用 B-spline grid 修正，无需全局 spline
- 旧版本：保留兼容性（全局 spline + 方向缩放）

### 3. 配置文件更新

#### `configs/bridgeb.yaml`
```yaml
deformation:
  use_directional_bspline: true  # 启用新版本
  directional_bspline_grid:
    n_alpha: 12
    n_depth: 10
    alpha_method: asin
    max_delta_log: 0.5
    spline_order: 3
  bspline_constraints:
    lambda_mono: 0.1
    lambda_smooth: 0.001
    lambda_far: 0.1
```

## 技术细节

### 方向变量设计

**不使用完整球面 (θ, φ)**，而是使用：
- `alpha = asin(ray_dir.y)` ∈ [-π/2, π/2]（推荐）
- 或 `alpha = ray_dir.y` ∈ [-1, 1]

**原因**：
- 全景深度误差主要随仰角变化
- 方位角误差弱，可忽略
- 极区稳定性更好

### B-spline Grid 参数化

- **Grid 维度**: 2D (alpha × log_depth)
- **推荐分辨率**: 
  - `N_alpha = 8 ~ 16`
  - `N_depth = 8 ~ 12`
- **Spline 阶数**: 3 (cubic)
- **Local support**: 每个查询点影响 4×4 个控制点

### 前向计算流程

```python
for each pixel / ray:
    1. 计算 ray_dir
    2. alpha = direction_to_alpha(ray_dir)
    3. d = raw_depth
    4. x = log(d)
    5. delta = bspline_interp(alpha, x)
    6. d_corrected = exp(x + delta)
```

### 损失函数

1. **几何一致性**: Point-to-Ray / Ray-Depth（已有）
2. **单调性约束**: `L_mono = sum max(0, -(c_{i,j+1} - c_{i,j}))`
3. **方向平滑正则**: `L_smooth = sum ||c_{i+1,j} - c_{i,j}||^2`
4. **远景渐近约束**: 最远列控制点 L2 约束

### 初始化策略

- **控制点初始化**: `control_points[:] = 0.0`
- **含义**: 初始等价于 identity mapping，不破坏原始深度
- **稳定 warm-up**: 从零开始优化

## 使用方式

### 创建模块

```python
from src.deformation import DepthReparameterization

module = DepthReparameterization(
    height=H,
    width=W,
    use_directional_bspline=True,  # 启用新版本
    n_alpha=12,
    n_depth=10,
    alpha_method="asin",
    max_delta_log=0.5,
    log_depth_min=-3.0,
    log_depth_max=5.0,
)
```

### 计算约束损失

```python
from src.regularization import compute_bspline_constraints_loss_multi_view

# 获取控制点
control_points_list = [
    module.get_directional_bspline().get_control_points()
    for module in depth_reparam_modules
]

# 计算约束损失
constraint_losses = compute_bspline_constraints_loss_multi_view(
    control_points_list=control_points_list,
    lambda_mono=0.1,
    lambda_smooth=0.001,
    lambda_far=0.1,
)
```

## 待完成工作

1. **更新 `scripts/run_bridgeb_optimization.py`**
   - 从配置文件读取 `use_directional_bspline` 和相关参数
   - 在损失函数中集成 B-spline 约束损失

2. **更新 `src/solver/energy_function.py`**
   - 添加 B-spline 约束损失到总能量函数

3. **测试**
   - 单元测试：B-spline 插值正确性
   - 集成测试：端到端优化流程
   - 验证：单调性、平滑性、远景渐近性

## 禁止事项（已遵守）

✅ **不再使用球谐函数**（新版本中）
✅ **不在 (θ, φ) 上建 2D spline**（使用 alpha × log_depth）
✅ **不用 MLP 替代 spline**（使用 B-spline grid）
✅ **不让 spline 同时解释方向和深度而不解耦**（明确分离 alpha 和 log_depth）

## 优势

1. **局部修正**: B-spline grid 支持局部方向修正，不受全局耦合影响
2. **稳定训练**: 单调性约束保证深度顺序，远景渐近约束防止发散
3. **方向解耦**: 使用 alpha（仰角）而非完整球面，更符合误差特性
4. **可解释性**: 控制点网格直观，易于可视化和调试
