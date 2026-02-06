#!/usr/bin/env python3
"""
绘制损失历史曲线，用于分析权重设置
"""
import sys
from pathlib import Path
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

def load_loss_history(file_path: Path):
    """加载损失历史（支持 JSON 和 CSV）"""
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        history = data['history']
        config = data.get('config', {})
        iterations = data.get('iterations', len(history['total']))
        return history, config, iterations
    elif file_path.suffix == '.csv':
        history = {
            'total': [],
            'geometry_p2r': [],
            'geometry_depth': [],
            'regularization_prior': [],
            'regularization_smooth': [],
            'regularization_scale': [],
            'weighted_p2r': [],
            'weighted_depth': [],
            'weighted_prior': [],
            'weighted_smooth': [],
            'weighted_scale': [],
        }
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history['total'].append(float(row['total']))
                history['geometry_p2r'].append(float(row['p2r_raw']))
                history['geometry_depth'].append(float(row['depth_raw']))
                history['regularization_prior'].append(float(row['prior_raw']))
                history['regularization_smooth'].append(float(row['smooth_raw']))
                history['regularization_scale'].append(float(row['scale_raw']))
                history['weighted_p2r'].append(float(row['p2r_weighted']))
                history['weighted_depth'].append(float(row['depth_weighted']))
                history['weighted_prior'].append(float(row['prior_weighted']))
                history['weighted_smooth'].append(float(row['smooth_weighted']))
                history['weighted_scale'].append(float(row['scale_weighted']))
        return history, {}, len(history['total'])
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

def plot_loss_history(history, config, output_path: Path):
    """绘制损失曲线"""
    iterations = np.arange(1, len(history['total']) + 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Loss History Analysis', fontsize=16)
    
    # 1. 总损失
    ax = axes[0, 0]
    ax.plot(iterations, history['total'], 'b-', linewidth=2, label='Total Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. 原始损失（几何一致性）
    ax = axes[0, 1]
    ax.plot(iterations, history['geometry_p2r'], 'r-', label='P2R (raw)', linewidth=1.5)
    ax.plot(iterations, history['geometry_depth'], 'g-', label='Depth (raw)', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Geometry Loss (Raw)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')  # 使用对数刻度
    
    # 3. 原始损失（正则化）
    ax = axes[1, 0]
    ax.plot(iterations, history['regularization_prior'], 'm-', label='Prior (raw)', linewidth=1.5)
    ax.plot(iterations, history['regularization_smooth'], 'c-', label='Smooth (raw)', linewidth=1.5)
    ax.plot(iterations, history['regularization_scale'], 'y-', label='Scale (raw)', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Regularization Loss (Raw)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # 4. 加权损失（用于分析权重影响）
    ax = axes[1, 1]
    ax.plot(iterations, history['weighted_p2r'], 'r-', label=f'P2R (λ={config.get("lambda_p2r", "?")})', linewidth=2)
    ax.plot(iterations, history['weighted_prior'], 'm-', label=f'Prior (λ={config.get("lambda_prior", "?")})', linewidth=2)
    ax.plot(iterations, history['weighted_depth'], 'g--', label=f'Depth (λ={config.get("lambda_depth", "?")})', linewidth=1.5)
    ax.plot(iterations, history['weighted_smooth'], 'c--', label=f'Smooth (λ={config.get("lambda_smooth", "?")})', linewidth=1.5)
    ax.plot(iterations, history['weighted_scale'], 'y--', label=f'Scale (λ={config.get("lambda_scale", "?")})', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weighted Loss')
    ax.set_title('Weighted Losses (for Weight Analysis)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 损失曲线已保存: {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="绘制损失历史曲线")
    parser.add_argument(
        "--loss_file",
        type=Path,
        required=True,
        help="损失历史文件路径（JSON 或 CSV）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出图像路径（默认：与输入文件同目录，扩展名为.png）"
    )
    
    args = parser.parse_args()
    
    if not args.loss_file.exists():
        print(f"❌ 文件不存在: {args.loss_file}")
        sys.exit(1)
    
    # 加载损失历史
    print(f"📖 加载损失历史: {args.loss_file}")
    history, config, iterations = load_loss_history(args.loss_file)
    print(f"  迭代次数: {iterations}")
    print(f"  配置: {config}")
    
    # 确定输出路径
    if args.output is None:
        output_path = args.loss_file.with_suffix('.png')
    else:
        output_path = args.output
    
    # 绘制
    print(f"📊 绘制损失曲线...")
    plot_loss_history(history, config, output_path)
    
    # 打印统计信息
    print(f"\n📊 损失统计:")
    print(f"  总损失: 初始={history['total'][0]:.6f}, 最终={history['total'][-1]:.6f}")
    print(f"  P2R损失: 初始={history['geometry_p2r'][0]:.6f}, 最终={history['geometry_p2r'][-1]:.6f}")
    print(f"  先验损失: 初始={history['regularization_prior'][0]:.6f}, 最终={history['regularization_prior'][-1]:.6f}")
    
    if config:
        print(f"\n⚙️  当前权重配置:")
        print(f"  lambda_p2r = {config.get('lambda_p2r', 'N/A')}")
        print(f"  lambda_prior = {config.get('lambda_prior', 'N/A')}")
        print(f"  lambda_depth = {config.get('lambda_depth', 'N/A')}")
        print(f"  lambda_smooth = {config.get('lambda_smooth', 'N/A')}")
        print(f"  lambda_scale = {config.get('lambda_scale', 'N/A')}")
        
        # 分析权重建议
        print(f"\n💡 权重分析建议:")
        final_p2r = history['geometry_p2r'][-1]
        final_prior = history['regularization_prior'][-1]
        
        if final_p2r > 0.001:
            print(f"  ⚠️  P2R损失仍较大 ({final_p2r:.6f})，建议增加 lambda_p2r")
        else:
            print(f"  ✅ P2R损失已收敛 ({final_p2r:.6f})")
        
        weighted_p2r_final = history['weighted_p2r'][-1]
        weighted_prior_final = history['weighted_prior'][-1]
        
        if weighted_p2r_final < weighted_prior_final:
            print(f"  ⚠️  加权P2R ({weighted_p2r_final:.6f}) < 加权先验 ({weighted_prior_final:.6f})，几何约束可能不足")
            print(f"     建议: 增加 lambda_p2r 或降低 lambda_prior")
        else:
            print(f"  ✅ 加权P2R ({weighted_p2r_final:.6f}) >= 加权先验 ({weighted_prior_final:.6f})，权重配置合理")

if __name__ == "__main__":
    main()
