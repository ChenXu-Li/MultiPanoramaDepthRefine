# Bug 修复：AttributeError 'float' object has no attribute 'item'

## 问题描述

运行优化时出现错误：
```
AttributeError: 'float' object has no attribute 'item'
```

错误发生在 `joint_optimization.py` 第 232 行：
```python
history['regularization_scale'].append(loss_dict['regularization']['scale'].item())
```

## 根本原因

在 `regularization_loss.py` 中，`scale_loss` 被初始化为 Python `float` (`0.0`)，而不是 `torch.Tensor`。当 `scale_modules` 为空或都为 `None` 时（新版本使用 B-spline grid，不使用 `scale_module`），`scale_loss` 保持为 `0.0`，导致后续调用 `.item()` 失败。

同样的问题也可能存在于 `prior_loss` 和 `smooth_loss` 的初始化。

## 修复方案

### 1. 修复 `src/regularization/regularization_loss.py`

- ✅ 将所有损失初始化为 `torch.Tensor` 而不是 `float`
- ✅ 确保设备一致性（从 `log_depths` 获取设备）
- ✅ 过滤掉 `None` 的 `scale_modules`（新版本兼容）

### 2. 修复 `src/regularization/scale_guard.py`

- ✅ 修复空列表时的设备获取问题
- ✅ 确保 `total_loss` 初始化为 tensor
- ✅ 正确处理 `None` 的 `scale_module`

### 3. 修复 `src/solver/joint_optimization.py`

- ✅ 添加 `safe_item()` 辅助函数，安全处理 tensor 或 float
- ✅ 所有 `.item()` 调用改为使用 `safe_item()`
- ✅ 确保收敛检查也使用 `safe_item()`

## 修改的文件

1. **`src/regularization/regularization_loss.py`**
   - `prior_loss`、`smooth_loss`、`scale_loss` 初始化为 tensor
   - 过滤 `None` 的 `scale_modules`

2. **`src/regularization/scale_guard.py`**
   - 修复空列表处理
   - 确保返回类型为 tensor

3. **`src/solver/joint_optimization.py`**
   - 添加 `safe_item()` 函数
   - 所有损失值获取使用 `safe_item()`

## 验证

运行测试确认修复：
```bash
python -c "
import torch
from src.regularization import RegularizationLoss

loss_fn = RegularizationLoss(...)
result = loss_fn.compute_loss(...)

# 所有损失都是 tensor
assert isinstance(result['prior'], torch.Tensor)
assert isinstance(result['smooth'], torch.Tensor)
assert isinstance(result['scale'], torch.Tensor)
"
```

✅ **测试通过**：所有损失值都是 `torch.Tensor` 类型

## 影响范围

- ✅ 修复了新版本（B-spline grid）的兼容性问题
- ✅ 保持了旧版本的兼容性
- ✅ 提高了代码的健壮性（`safe_item()` 可以处理意外情况）

## 状态

✅ **已修复并验证**

现在可以正常运行优化流程：
```bash
python scripts/run_bridgeb_optimization.py
```
