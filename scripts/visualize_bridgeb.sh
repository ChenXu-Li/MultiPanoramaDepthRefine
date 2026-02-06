#!/bin/bash
# 可视化 BridgeB 场景优化后的深度图
# 参考 DAP 的 infer_pics.sh

INPUT_DIR="/root/autodl-tmp/code/MultiPanoramaDepthRefine/outputs/aligned_depths/BridgeB"
OUTPUT_DIR="/root/autodl-tmp/code/MultiPanoramaDepthRefine/outputs/aligned_depths"

cd /root/autodl-tmp/code/MultiPanoramaDepthRefine

# 生成 100m 范围的可视化（默认）
echo "生成 100m 范围的可视化..."
python scripts/visualize_depths.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --vis_range 100m \
  --cmap Spectral

# 生成 10m 范围的可视化（用于查看近景细节）
echo ""
echo "生成 10m 范围的可视化..."
python scripts/visualize_depths.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --vis_range 10m \
  --cmap Spectral

echo ""
echo "✅ 可视化完成！"
echo "   输出目录:"
echo "     100m 彩色: ${OUTPUT_DIR}/depth_vis_color_100m"
echo "     100m 灰度: ${OUTPUT_DIR}/depth_vis_gray_100m"
echo "     10m 彩色: ${OUTPUT_DIR}/depth_vis_color_10m"
echo "     10m 灰度: ${OUTPUT_DIR}/depth_vis_gray_10m"
