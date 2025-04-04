import time
import numpy as np
import matplotlib.pyplot as plt
import os
from run_simulation import run_nerf_accelerator_simulation

def compare_memory_access_strategies(sample_locations_file=None, source_view_poses_file=None, target_view_pose_file=None):
    """
    Compare all three memory access strategies
    
    Args:
        sample_locations_file: Path to sample_locations_valid.npy
        source_view_poses_file: Path to source_view_poses.npy
        target_view_pose_file: Path to target_view_pose.npy
    """
    # Create results directory
    os.makedirs("comparison_results", exist_ok=True)
    
    # Strategies to compare
    strategies = ["baseline", "spatial_locality", "epipolar"]
    results = {}
    
    for strategy in strategies:
        print(f"\n\n===== Running simulation with {strategy} strategy =====\n")
        
        # Skip epipolar if required files are missing
        if strategy == "epipolar" and (not sample_locations_file or not source_view_poses_file or not target_view_pose_file):
            print("Skipping epipolar strategy as 3D data files are missing")
            continue
        
        # Run simulation for this strategy
        result = run_nerf_accelerator_simulation(
            memory_access_strategy=strategy,
            sample_locations_file=sample_locations_file,
            source_view_poses_file=source_view_poses_file,
            target_view_pose_file=target_view_pose_file
        )
        
        results[strategy] = result
        print(f"\n===== {strategy} strategy simulation complete =====\n")
    
    # Print comparison results
    print("\n\n===== Strategy Comparison Results =====")
    metrics = [
        "total_mb_read", 
        "total_memory_accesses", 
        "total_memory_cycles", 
        "row_hit_rate",
        "blocks_processed",
        "simulation_runtime_s"
    ]
    
    metrics_display_names = {
        "total_mb_read": "Total Read (MB)",
        "total_memory_accesses": "Memory Accesses",
        "total_memory_cycles": "Memory Cycles",
        "row_hit_rate": "Row Hit Rate (%)",
        "blocks_processed": "Blocks Processed",
        "simulation_runtime_s": "Simulation Runtime (s)"
    }
    
    # Calculate baseline for normalization
    baseline_values = {}
    for metric in metrics:
        if "baseline" in results:
            baseline_values[metric] = results["baseline"].get(metric, 0)
        else:
            baseline_values[metric] = 1.0  # Avoid division by zero
    
    # Print header with available strategies
    available_strategies = [s for s in strategies if s in results]
    header = f"{'Metric':<20}"
    for s in available_strategies:
        header += f" {s.capitalize():<15}"
    print(header)
    print("-" * (20 + 15 * len(available_strategies)))
    
    # Print metrics
    for metric in metrics:
        line = f"{metrics_display_names[metric]:<20}"
        
        for strategy in available_strategies:
            value = results[strategy].get(metric, 0)
            
            # Format based on metric type
            if metric == "total_mb_read":
                line += f" {value:.2f}{"":>10}"
            elif metric == "row_hit_rate":
                line += f" {value:.2f}%{"":>9}"
            elif metric == "simulation_runtime_s":
                line += f" {value:.2f}{"":>10}"
            else:
                line += f" {int(value)}{"":>10}"
        
        print(line)
    
    # Print relative improvement
    print("\n===== Improvement Relative to Baseline =====")
    for metric in metrics:
        if "baseline" in results:
            line = f"{metrics_display_names[metric]:<20}"
            baseline_val = baseline_values[metric]
            
            for strategy in available_strategies:
                if strategy == "baseline":
                    line += f" --{"":>13}"
                    continue
                    
                value = results[strategy].get(metric, 0)
                
                # Calculate improvement percentage
                if metric in ["row_hit_rate"]:
                    # For metrics where higher is better
                    if baseline_val > 0:
                        improvement = ((value - baseline_val) / baseline_val) * 100
                    else:
                        improvement = float('inf')
                    improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
                else:
                    # For metrics where lower is better
                    if baseline_val > 0:
                        improvement = ((baseline_val - value) / baseline_val) * 100
                    else:
                        improvement = float('inf') if value < baseline_val else float('-inf')
                    improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
                
                line += f" {improvement_str:<15}"
            
            print(line)
    
    # Generate visualization
    try:
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.flatten()
        
        # Plot data for each metric
        for i, metric in enumerate(metrics):
            values = [results[s].get(metric, 0) for s in available_strategies]
            axs[i].bar(available_strategies, values)
            axs[i].set_title(metrics_display_names[metric])
            
            # Add values on top of bars
            for j, v in enumerate(values):
                if metric == "total_mb_read" or metric == "simulation_runtime_s":
                    value_text = f"{v:.2f}"
                elif metric == "row_hit_rate":
                    value_text = f"{v:.2f}%"
                else:
                    value_text = f"{int(v)}"
                axs[i].text(j, v, value_text, ha='center', va='bottom')
            
            # Add y-label
            if metric == "total_mb_read":
                axs[i].set_ylabel("MB")
            elif metric == "row_hit_rate":
                axs[i].set_ylabel("Percentage")
            elif metric == "simulation_runtime_s":
                axs[i].set_ylabel("Seconds")
        
        plt.tight_layout()
        plt.savefig("comparison_results/strategy_comparison.png")
        print("\nComparison visualization saved to comparison_results/strategy_comparison.png")
    except Exception as e:
        print(f"Error generating comparison visualization: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare memory access strategies")
    parser.add_argument("--sample-locations", help="Path to sample_locations_valid.npy")
    parser.add_argument("--source-poses", help="Path to source_view_poses.npy")
    parser.add_argument("--target-pose", help="Path to target_view_pose.npy")
    
    args = parser.parse_args()
    
    compare_memory_access_strategies(
        sample_locations_file=args.sample_locations,
        source_view_poses_file=args.source_poses,
        target_view_pose_file=args.target_pose
    )