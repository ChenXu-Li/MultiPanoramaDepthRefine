# Step 4: 最终联合优化 - 完成报告

## 概述

Step 4 实现了多视角深度图的最终联合优化，组合了 Step 1（深度重参数化）、Step 2（多视角几何一致性）和 Step 3（防退化护栏）的所有组件。

## 实现内容

### 1. 总能量函数 (`src/solver/energy_function.py`)

实现了 `TotalEnergyFunction` 类，组合：

- **几何一致性损失**（Step 2）：
  - Point-to-Ray 距离损失（`λ_p2r`）
  - Ray-space 深度一致性损失（`λ_depth`）

- **正则化损失**（Step 3）：
  - log-depth 先验锚点损失（`λ_prior`）
  - 球面平滑正则损失（`λ_smooth`）
  - 方向变形约束损失（`λ_scale`）

总损失形式：
```
L = λ_p2r * L_p2r + λ_depth * L_depth + λ_prior * L_prior + λ_smooth * L_smooth + λ_scale * L_scale
```

### 2. 优化器封装 (`src/solver/optimizer.py`)

实现了 `MultiViewOptimizer` 类：

- 自动收集所有深度重参数化模块的可学习参数
- 支持 Adam 和 SGD 优化器
- 在每次优化步骤后确保单调性约束

### 3. 联合优化主循环 (`src/solver/joint_optimization.py`)

实现了 `optimize_multi_view_depth` 函数：

- 主优化循环
- 梯度裁剪（防止梯度爆炸）
- 收敛检测和早停
- 历史记录（可选）
- 中间结果保存（可选）

### 4. 输出验证 (`src/solver/validation.py`)

实现了 `validate_optimized_depths` 函数，检查：

- ✅ 深度未整体塌缩（深度标准差 > 阈值）
- ✅ 未整体推到 100m 球壳（深度均值不在 100m 附近）
- ✅ 单视角仍光滑连续（梯度统计）
- ✅ 深度值在合理范围

## 推荐配置

根据 `CURSORREAD.md`，推荐起步权重：

```python
λ_p2r    = 1.0   # Point-to-Ray 损失（主约束）
λ_depth  = 0.1   # 深度一致性损失（弱约束）
λ_prior  = 1.0   # 先验锚点损失（防止塌缩）
λ_smooth = 0.01  # 平滑正则损失
λ_scale  = 0.01  # 方向变形约束损失
```

## 使用方法

### Python API

```python
from src.solver import JointOptimizationConfig, optimize_multi_view_depth
from src.deformation import DepthReparameterization

# 创建深度重参数化模块（每个视角一个）
depth_reparam_modules = [
    DepthReparameterization(height=H, width=W, ...),
    DepthReparameterization(height=H, width=W, ...),
]

# DAP 深度
log_depth_daps = [log_depth_dap_1, log_depth_dap_2, ...]
depth_daps = [depth_dap_1, depth_dap_2, ...]

# 相机位姿
cam_from_world_list = [cam_pose_1, cam_pose_2, ...]

# 配置
config = JointOptimizationConfig(
    lambda_p2r=1.0,
    lambda_depth=0.1,
    lambda_prior=1.0,
    lambda_smooth=0.01,
    lambda_scale=0.01,
    max_iter=1000,
    lr=1e-3,
    device="cuda",
)

# 优化
depths_opt, report = optimize_multi_view_depth(
    depth_reparam_modules=depth_reparam_modules,
    log_depth_daps=log_depth_daps,
    depth_daps=depth_daps,
    cam_from_world_list=cam_from_world_list,
    config=config,
)
```

## 测试

### 基本测试

运行 `tests/test_step4_simple.py`：

```bash
cd /root/autodl-tmp/code/MultiPanoramaDepthRefine
python tests/test_step4_simple.py
```

### 完整测试

运行 `tests/test_step4.py`：

```bash
pytest tests/test_step4.py -v
```

## 最终检查清单

根据 `CURSORREAD.md`，最终检查项：

- [ ] 深度未整体塌缩
- [ ] 未整体推到 100m 球壳
- [ ] 单视角仍光滑连续
- [ ] 多视角结构一致性明显提升

使用 `validate_optimized_depths` 函数进行自动验证。

## 文件结构

```
src/solver/
├── __init__.py              # 模块导出
├── energy_function.py       # 总能量函数
├── optimizer.py             # 优化器封装
├── joint_optimization.py    # 联合优化主循环
└── validation.py            # 输出验证

tests/
├── test_step4.py            # 完整单元测试
└── test_step4_simple.py     # 简化测试
```

## 注意事项

1. **权重配置**：`λ_prior` 必须 > 0，防止深度整体塌缩
2. **学习率**：建议从 `1e-3` 开始，根据收敛情况调整
3. **迭代次数**：建议 500-1000 次迭代，可通过早停提前结束
4. **设备**：支持 CPU 和 CUDA，建议使用 GPU 加速

## 下一步

完成 Step 4 后，可以：

1. 使用优化后的深度图生成点云
2. 进行可视化验证
3. 评估多视角一致性提升
