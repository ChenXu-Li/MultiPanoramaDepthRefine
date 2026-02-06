# Step 2 实现完成总结

## 已完成的功能

### 1. 坐标变换 (`src/geometry/coordinate_transform.py`)
- ✅ 深度 → 世界点：`P_i = T_i ( D'_i \cdot r(\theta,\phi) )`
- ✅ 获取相机中心在世界坐标系中的位置
- ✅ 获取相机在世界坐标系中的射线方向
- ✅ 支持批量处理

### 2. Point-to-Ray 距离 (`src/geometry/point_to_ray.py`)
- ✅ 点到射线距离计算：`d = |(P - C) - ((P - C) · r) r|`
- ✅ 支持批量计算
- ✅ 支持 Huber loss（robust loss）
- ✅ 支持 mask 过滤

### 3. Ray-space 深度一致性 (`src/geometry/ray_space_consistency.py`)
- ✅ 世界点投影到相机坐标系
- ✅ 计算投影深度一致性：`|log D'_j - log D_{i→j}|`
- ✅ 支持 robust loss

### 4. 多视角几何损失 (`src/geometry/multi_view_loss.py`)
- ✅ `MultiViewGeometryLoss`：组合 Point-to-Ray 和深度一致性
- ✅ 权重控制：`lambda_p2r >> lambda_depth`
- ✅ 远景处理：`D^{DAP} >= 100m` 的像素不参与几何对齐
- ✅ 支持多视角对之间的损失计算

## 测试结果

所有测试通过：
- ✅ 坐标变换：深度 → 世界点转换正确
- ✅ Point-to-Ray 距离：距离计算正确
- ✅ 多视角损失（Identity）：所有视角深度相同 → loss ≈ 0
- ✅ 远景排除：>= 100m 的像素被正确排除

## 关键特性

1. **Point-to-Ray 优先**：`lambda_p2r >> lambda_depth`（方向一致性优先于径向一致性）
2. **远景处理**：远景（>= 100m）不参与几何对齐，仅参与平滑项
3. **Robust Loss**：Point-to-Ray 支持 Huber loss，对异常值鲁棒
4. **位姿冻结**：相机位姿不参与优化（梯度不回传到 T_i）

## 数学公式

### Point-to-Ray 距离（主约束）

$$
L_{p2r} = \left| (P_i - C_j) - ((P_i - C_j) \cdot r_j) r_j \right|
$$

其中：
- $P_i$：视角 i 的世界点
- $C_j$：视角 j 的相机中心
- $r_j$：视角 j 的射线方向（单位向量）

### Ray-space 深度一致性（弱约束）

$$
L_{depth} = \left| \log D'_j - \log D_{i\rightarrow j} \right|
$$

其中：
- $D'_j$：视角 j 的优化后深度
- $D_{i\rightarrow j}$：视角 i 的点投影到视角 j 的深度

### 总损失

$$
L_{geometry} = \lambda_{p2r} L_{p2r} + \lambda_{depth} L_{depth}
$$

其中：$\lambda_{p2r} \gg \lambda_{depth}$

## 使用示例

```python
from src.geometry import MultiViewGeometryLoss
import torch
import pycolmap

# 创建损失函数
loss_fn = MultiViewGeometryLoss(
    lambda_p2r=1.0,
    lambda_depth=0.1,
    far_threshold=100.0,
    use_robust_p2r=True,
    huber_delta_p2r=0.1,
    device="cpu",
)

# 计算损失
loss_dict = loss_fn.compute_loss(
    depths=[depth_1, depth_2],  # List[(H, W)]
    cam_from_world_list=[pose_1, pose_2],  # List[pycolmap.Rigid3d]
    depth_dap_list=[dap_1, dap_2],  # List[(H, W)] 用于远景 mask
    height=height,
    width=width,
)

total_loss = loss_dict['total']
p2r_loss = loss_dict['p2r']
depth_loss = loss_dict['depth']
```

## 注意事项

1. **权重配置**：必须确保 `lambda_p2r > lambda_depth`
2. **远景处理**：使用 DAP 深度判断远景（`D^{DAP} >= 100m`）
3. **对应关系**：当前实现使用简化版本，实际应用中应通过投影建立精确对应关系
4. **位姿冻结**：相机位姿不参与优化，只用于坐标变换

## 下一步

Step 2 已完成，可以继续实现：
- Step 3: 防退化护栏
- Step 4: 联合优化
