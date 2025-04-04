import time
import numpy as np
import matplotlib.pyplot as plt
from run_simulation import run_nerf_accelerator_simulation

def compare_memory_access_strategies():
    """比较不同内存访问策略的性能"""
    strategies = ["baseline", "spatial_locality"]
    results = {}
    
    for strategy in strategies:
        print(f"\n\n===== 开始使用 {strategy} 策略运行仿真 =====\n")
        result = run_nerf_accelerator_simulation(memory_access_strategy=strategy)
        results[strategy] = result
        print(f"\n===== {strategy} 策略运行结束 =====\n")
    
    # 打印比较结果
    print("\n\n===== 策略比较结果 =====")
    metrics = [
        "total_mb_read", 
        "total_memory_accesses", 
        "total_memory_cycles", 
        "row_hit_rate",
        "blocks_processed"
    ]
    
    metrics_display_names = {
        "total_mb_read": "总读取量 (MB)",
        "total_memory_accesses": "内存访问次数",
        "total_memory_cycles": "内存访问周期",
        "row_hit_rate": "行命中率 (%)",
        "blocks_processed": "处理的内存块数"
    }
    
    print(f"{'指标':<20} {'Baseline':<15} {'空间局部性':<15} {'改进百分比':<15}")
    print("-" * 65)
    
    for metric in metrics:
        baseline_value = results["baseline"].get(metric, 0)
        spatial_value = results["spatial_locality"].get(metric, 0)
        
        if metric == "row_hit_rate":
            # 对于命中率，高是好的
            improvement = ((spatial_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0
            improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
        else:
            # 对于其他指标，低是好的
            improvement = ((baseline_value - spatial_value) / baseline_value) * 100 if baseline_value > 0 else 0
            improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
        
        # 格式化值
        if metric == "total_mb_read":
            baseline_str = f"{baseline_value:.2f}"
            spatial_str = f"{spatial_value:.2f}"
        elif metric in ["row_hit_rate"]:
            baseline_str = f"{baseline_value:.2f}%"
            spatial_str = f"{spatial_value:.2f}%"
        else:
            baseline_str = f"{int(baseline_value)}"
            spatial_str = f"{int(spatial_value)}"
        
        print(f"{metrics_display_names[metric]:<20} {baseline_str:<15} {spatial_str:<15} {improvement_str:<15}")
    
    # 可视化比较结果
    try:
        plt.figure(figsize=(12, 8))
        
        # 内存读取量比较
        plt.subplot(2, 2, 1)
        plt.bar(['Baseline', 'Spatial Locality'], 
                [results["baseline"]["total_mb_read"], results["spatial_locality"]["total_mb_read"]])
        plt.title('总读取量 (MB)')
        plt.ylabel('MB')
        
        # 内存访问次数比较
        plt.subplot(2, 2, 2)
        plt.bar(['Baseline', 'Spatial Locality'], 
                [results["baseline"]["total_memory_accesses"], results["spatial_locality"]["total_memory_accesses"]])
        plt.title('内存访问次数')
        
        # 行命中率比较
        plt.subplot(2, 2, 3)
        plt.bar(['Baseline', 'Spatial Locality'], 
                [results["baseline"]["row_hit_rate"], results["spatial_locality"]["row_hit_rate"]])
        plt.title('行命中率 (%)')
        plt.ylabel('%')
        
        # 处理的内存块数比较
        plt.subplot(2, 2, 4)
        plt.bar(['Baseline', 'Spatial Locality'], 
                [results["baseline"]["blocks_processed"], results["spatial_locality"]["blocks_processed"]])
        plt.title('处理的内存块数')
        
        plt.tight_layout()
        plt.savefig('strategy_comparison.png')
        print("\n比较结果图表已保存为 strategy_comparison.png")
    except Exception as e:
        print(f"生成比较图表时出错: {e}")

if __name__ == "__main__":
    compare_memory_access_strategies()