#!/bin/bash
# 生成 BridgeB 场景的点云
# 从优化后的深度图生成二进制 PLY 点云

cd /root/autodl-tmp/code/MultiPanoramaDepthRefine

python scripts/generate_pointclouds.py \
  --depth_dir outputs/aligned_depths/BridgeB \
  --rgb_dir /root/autodl-tmp/data/STAGE1_4x/BridgeB/backgrounds \
  --output_dirs \
    outputs/pointclouds \
    /root/autodl-tmp/data/STAGE1_4x/BridgeB/pointclouds_mutil_opt \
  --verbose

echo ""
echo "✅ 点云生成完成！"
echo "   输出目录 1: outputs/pointclouds"
echo "   输出目录 2: /root/autodl-tmp/data/STAGE1_4x/BridgeB/pointclouds_mutil_opt"
