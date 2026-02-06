# 损失历史记录与分析指南

## 概述

优化过程会自动记录所有损失项的历史，包括：
- **原始损失**：各项损失的原始值
- **加权损失**：各项损失乘以对应权重后的值（用于分析权重影响）

## 自动记录

优化脚本会自动保存损失历史到：
```
logs/loss_history/{scene_name}/loss_history_{timestamp}.json
logs/loss_history/{scene_name}/loss_history_{timestamp}.csv
```

### JSON 格式
包含完整的配置信息和损失历史：
```json
{
  "config": {
    "lambda_p2r": 2.0,
    "lambda_prior": 0.5,
    ...
  },
  "iterations": 485,
  "final_energy": 0.074567,
  "history": {
    "total": [0.082, 0.080, ...],
    "geometry_p2r": [0.001, 0.0008, ...],
    ...
  }
}
```

### CSV 格式
便于在 Excel 或 Python 中分析：
```csv
iteration,total,p2r_raw,depth_raw,prior_raw,smooth_raw,scale_raw,p2r_weighted,depth_weighted,prior_weighted,smooth_weighted,scale_weighted
1,0.082,0.001,0.036,0.014,0.001,0.0001,0.002,0.0036,0.007,0.00001,0.000001
2,0.080,0.0008,0.035,0.014,0.001,0.0001,0.0016,0.0035,0.007,0.00001,0.000001
...
```

## 损失项说明

### 原始损失（Raw Loss）
- `geometry_p2r`: Point-to-Ray 距离损失（几何一致性主约束）
- `geometry_depth`: Ray-space 深度一致性损失（几何一致性弱约束）
- `regularization_prior`: 先验锚点损失（防止过度偏离 DAP）
- `regularization_smooth`: 平滑正则损失（保持深度场连续）
- `regularization_scale`: 方向变形约束损失（防止方向过度变形）

### 加权损失（Weighted Loss）
- `weighted_p2r` = `lambda_p2r` × `geometry_p2r`
- `weighted_depth` = `lambda_depth` × `geometry_depth`
- `weighted_prior` = `lambda_prior` × `regularization_prior`
- `weighted_smooth` = `lambda_smooth` × `regularization_smooth`
- `weighted_scale` = `lambda_scale` × `regularization_scale`

**总损失** = 所有加权损失之和

## 可视化损失曲线

### 使用脚本绘制

```bash
python scripts/plot_loss_history.py \
    --loss_file logs/loss_history/BridgeB/loss_history_20240101_120000.json \
    --output loss_curves.png
```

或使用 CSV：
```bash
python scripts/plot_loss_history.py \
    --loss_file logs/loss_history/BridgeB/loss_history_20240101_120000.csv
```

### 输出内容

脚本会生成包含4个子图的损失曲线：
1. **总损失**：整体优化进度
2. **几何损失（原始）**：P2R 和深度一致性损失
3. **正则化损失（原始）**：先验、平滑、缩放约束损失
4. **加权损失**：用于分析权重影响

## 权重调整建议

### 分析步骤

1. **查看损失曲线**
   ```bash
   python scripts/plot_loss_history.py --loss_file logs/loss_history/BridgeB/loss_history_*.json
   ```

2. **检查收敛情况**
   - 如果 `total` 损失还在下降 → 增加 `max_iter`
   - 如果 `total` 损失已收敛但 `p2r` 仍较大 → 增加 `lambda_p2r`

3. **分析权重平衡**
   - 比较 `weighted_p2r` 和 `weighted_prior` 的最终值
   - 如果 `weighted_p2r < weighted_prior` → 几何约束不足，需要增加 `lambda_p2r`

### 常见问题与调整

#### 问题 1: P2R 损失未收敛（> 0.001）

**症状**：
- `geometry_p2r` 最终值 > 0.001
- 点云未完全重合

**解决方案**：
```python
lambda_p2r = 5.0  # 从 2.0 增加到 5.0
lambda_prior = 0.1  # 从 0.5 降低到 0.1（减少对 DAP 的依赖）
```

#### 问题 2: 提前收敛但未完全重合

**症状**：
- 能量变化率 < 阈值，提前停止
- 但 `geometry_p2r` 仍较大

**解决方案**：
```python
early_stop_threshold = 1e-8  # 从 1e-7 降低到 1e-8（更严格）
# 或
max_iter = 2000  # 增加最大迭代次数
```

#### 问题 3: 损失震荡

**症状**：
- 损失曲线上下波动
- 不收敛

**解决方案**：
```python
lr = 1e-4  # 从 5e-4 降低到 1e-4（更小的学习率）
```

#### 问题 4: 先验损失过大

**症状**：
- `weighted_prior` 远大于 `weighted_p2r`
- 优化结果过于接近 DAP，几何一致性不足

**解决方案**：
```python
lambda_prior = 0.1  # 从 0.5 降低到 0.1
lambda_p2r = 5.0  # 从 2.0 增加到 5.0
```

## Python 分析示例

### 读取并分析损失历史

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 加载损失历史
with open('logs/loss_history/BridgeB/loss_history_*.json', 'r') as f:
    data = json.load(f)

history = data['history']
config = data['config']

# 分析收敛速度
iterations = np.arange(1, len(history['total']) + 1)
total_loss = np.array(history['total'])

# 计算损失下降速度
loss_reduction = total_loss[0] - total_loss[-1]
reduction_rate = loss_reduction / len(iterations)

print(f"总损失下降: {loss_reduction:.6f}")
print(f"平均每迭代下降: {reduction_rate:.6f}")

# 检查 P2R 损失
p2r_final = history['geometry_p2r'][-1]
if p2r_final > 0.001:
    print(f"⚠️  P2R 损失未收敛: {p2r_final:.6f}")
    print(f"   建议增加 lambda_p2r 到 {config['lambda_p2r'] * 2:.1f}")
else:
    print(f"✅ P2R 损失已收敛: {p2r_final:.6f}")

# 分析权重平衡
weighted_p2r_final = history['weighted_p2r'][-1]
weighted_prior_final = history['weighted_prior'][-1]

if weighted_p2r_final < weighted_prior_final:
    print(f"⚠️  几何约束不足")
    print(f"   加权 P2R: {weighted_p2r_final:.6f}")
    print(f"   加权先验: {weighted_prior_final:.6f}")
    print(f"   建议: lambda_p2r = {config['lambda_p2r'] * 2:.1f}")
```

## 输出文件位置

损失历史文件保存在：
```
logs/loss_history/{scene_name}/
├── loss_history_{timestamp}.json  # JSON 格式（包含配置）
├── loss_history_{timestamp}.csv   # CSV 格式（便于分析）
└── loss_history_{timestamp}.png   # 损失曲线图（如果使用可视化脚本）
```

## 注意事项

1. **总是记录**：损失历史总是被记录，不依赖 `save_history` 配置
2. **文件大小**：每次迭代记录约 11 个浮点数，1000 次迭代约 88KB（JSON）
3. **时间戳**：每次运行生成新的时间戳，不会覆盖旧文件
4. **配置信息**：JSON 文件包含完整的优化配置，便于复现

## 快速检查命令

```bash
# 查看最新的损失历史文件
ls -lt logs/loss_history/BridgeB/ | head -5

# 绘制最新的损失曲线
LATEST=$(ls -t logs/loss_history/BridgeB/*.json | head -1)
python scripts/plot_loss_history.py --loss_file "$LATEST"

# 查看损失统计
python -c "
import json
with open('$LATEST') as f:
    data = json.load(f)
h = data['history']
print(f'迭代: {data[\"iterations\"]}')
print(f'总损失: {h[\"total\"][0]:.6f} -> {h[\"total\"][-1]:.6f}')
print(f'P2R: {h[\"geometry_p2r\"][0]:.6f} -> {h[\"geometry_p2r\"][-1]:.6f}')
"
```
