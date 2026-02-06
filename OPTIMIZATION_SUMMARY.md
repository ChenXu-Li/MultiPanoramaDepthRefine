# BridgeB 场景多视角深度联合优化 - 完成报告

## 优化结果

### 基本信息
- **场景**: BridgeB
- **全景图数量**: 4
- **全景图列表**: point2_median, point3_median, point4_median, point5_median
- **图像尺寸**: 960x1920
- **优化设备**: CUDA

### 优化过程
- **总迭代次数**: 485（提前收敛）
- **最终能量**: 0.074567
- **收敛状态**: ✅ 提前收敛（能量变化率 < 阈值）
- **验证结果**: ✅ 通过

### 输出文件
所有优化后的深度图已保存到：
`/root/autodl-tmp/code/MultiPanoramaDepthRefine/outputs/aligned_depths/BridgeB/`

文件列表：
- `point2_median_aligned.npy` (7.1MB)
- `point3_median_aligned.npy` (7.1MB)
- `point4_median_aligned.npy` (7.1MB)
- `point5_median_aligned.npy` (7.1MB)

### 深度图统计
- **形状**: (960, 1920)
- **深度范围**: [0.57, 107.06] 米
- **有效像素**: 100% (1,843,200/1,843,200)

### 性能指标
- **每次迭代耗时**: ~0.48s
- **总优化时间**: ~4 分钟（485 次迭代）
- **GPU 内存使用**: ~0.25 GB

### 损失值变化
- **初始总损失**: ~0.082
- **最终总损失**: 0.074567
- **Point-to-Ray 损失**: 0.000000（已收敛）
- **Prior 损失**: ~0.0105（稳定）

## 下一步
优化后的深度图可用于：
1. 生成一致的点云
2. 进行可视化验证
3. 评估多视角一致性提升

