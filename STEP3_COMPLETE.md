# Step 3 实现完成总结

## 已完成的功能

### 1. log-depth 先验锚点 (`src/regularization/prior_anchor.py`)
- ✅ 实现：`L_prior = |log D' - log D^DAP|^2`
- ✅ 防止整体塌缩到常数深度
- ✅ 支持 L2 和 Huber loss
- ✅ 支持多视角计算

### 2. 球面平滑正则 (`src/regularization/smoothness.py`)
- ✅ 实现：`L_smooth = |∇_θ D'|^2 + |∇_φ D'|^2`
- ✅ 在 equirectangular 空间中计算梯度（u/v 方向）
- ✅ 支持 L1 和 L2 平滑
- ✅ 支持边缘感知平滑（基于 RGB）
- ✅ 支持多视角计算

### 3. 方向变形约束 (`src/regularization/scale_guard.py`)
- ✅ 实现：`L_g = |g|^2`
- ✅ 防止方向缩放参数爆炸
- ✅ 支持球谐函数和 B-spline grid

### 4. 远景处理 (`src/regularization/far_field_mask.py`)
- ✅ 创建远景 mask：`D^{DAP} >= 100m`
- ✅ 远景不参与几何对齐（在 Step 2 中处理）
- ✅ 远景仅参与平滑项

### 5. 防退化护栏组合 (`src/regularization/regularization_loss.py`)
- ✅ `RegularizationLoss`：组合所有正则化项
- ✅ 权重控制
- ✅ 支持多视角

## 测试结果

所有测试通过：
- ✅ 先验锚点：零变形损失 = 0，塌缩损失显著增大
- ✅ 平滑正则：平滑深度损失 = 0，噪声深度损失增大
- ✅ 远景 mask：正确识别远景像素
- ✅ 防退化护栏：塌缩情况损失显著大于正常情况

## 关键特性

1. **先验锚点必须存在**：`lambda_prior > 0`（防止整体塌缩）
2. **平滑正则**：保持深度场连续性
3. **方向变形约束**：防止 scale 爆炸
4. **远景处理**：远景不参与几何对齐，仅参与平滑

## 数学公式

### log-depth 先验锚点

$$
L_{prior} = \sum_p |\log D' - \log D^{DAP}|^2
$$

### 球面平滑正则

$$
L_{smooth} = |\nabla_\theta D'|^2 + |\nabla_\phi D'|^2
$$

在 equirectangular 空间中：
- $\nabla_\theta$ 对应水平方向（u 方向）
- $\nabla_\phi$ 对应垂直方向（v 方向）

### 方向变形约束

$$
L_g = |g|^2
$$

其中 $g$ 是方向相关缩放的参数（球谐系数或 B-spline grid 值）

### 总正则化损失

$$
L_{regularization} = \lambda_{prior} L_{prior} + \lambda_{smooth} L_{smooth} + \lambda_{scale} L_g
$$

## 使用示例

```python
from src.regularization import RegularizationLoss
from src.deformation import SphericalHarmonicsScale
import torch

# 创建损失函数
loss_fn = RegularizationLoss(
    lambda_prior=1.0,      # 必须 > 0
    lambda_smooth=0.01,
    lambda_scale=0.01,
    far_threshold=100.0,
    prior_loss_type="l2",
    smooth_type="l2",
    edge_aware=False,
)

# 计算损失
loss_dict = loss_fn.compute_loss(
    log_depths=[log_depth_1, log_depth_2],  # List[(H, W)]
    log_depth_daps=[log_dap_1, log_dap_2],  # List[(H, W)]
    depth_daps=[depth_dap_1, depth_dap_2],  # List[(H, W)] 用于远景 mask
    scale_modules=[scale_1, scale_2],       # List[ScaleModule] 可选
    rgbs=[rgb_1, rgb_2],                    # List[(H, W, 3)] 可选
)

total_loss = loss_dict['total']
prior_loss = loss_dict['prior']
smooth_loss = loss_dict['smooth']
scale_loss = loss_dict['scale']
```

## 注意事项

1. **先验权重必须非零**：`lambda_prior > 0`（防止塌缩的关键）
2. **远景处理**：使用 DAP 深度判断远景（`D^{DAP} >= 100m`）
3. **权重平衡**：`lambda_prior >> lambda_smooth, lambda_scale`
4. **边缘感知平滑**：需要提供 RGB 图像

## 防退化机制

### 1. 防止整体塌缩
- **机制**：先验锚点强制深度接近 DAP
- **测试**：关闭 prior → 观察塌缩，打开 prior → 塌缩消失

### 2. 防止尺度退化
- **机制**：先验锚点 + 平滑正则
- **测试**：所有深度为常数 → loss 显著增大

### 3. 防止方向 scale 爆炸
- **机制**：方向变形约束 `L_g = |g|^2`
- **测试**：scale 系数过大 → loss 增大

### 4. 远景稳定性
- **机制**：远景不参与几何对齐，仅参与平滑
- **测试**：远景深度不随迭代剧烈变化

## 下一步

Step 3 已完成，可以继续实现：
- Step 4: 最终联合优化（组合 Step 1 + Step 2 + Step 3）
