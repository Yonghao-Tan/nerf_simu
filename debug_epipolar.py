"""
Epipolar调试脚本
为Yonghao-Tan制作 - 2025-04-04
"""
import os
import argparse
from run_simulation import run_nerf_accelerator_simulation

def debug_epipolar_strategy(
    sample_locations_file=None, 
    source_view_poses_file=None, 
    target_view_pose_file=None,
    frustum_size=(16, 16),
    depth_layers=8,
    max_frames=10
):
    """
    运行epipolar策略的3D可视化调试
    
    参数:
        sample_locations_file: 3D采样点位置文件
        source_view_poses_file: 源视图姿态文件
        target_view_pose_file: 目标视图姿态文件
        frustum_size: 视锥体大小 (高度,宽度)
        depth_layers: 深度层数
        max_frames: 最大调试帧数
    """
    print("=== Epipolar策略调试模式 ===")
    
    # 检查所需文件
    files_ok = True
    
    if not sample_locations_file or not os.path.exists(sample_locations_file):
        print(f"错误：未提供样本位置文件或文件不存在: {sample_locations_file}")
        files_ok = False
    
    if not source_view_poses_file or not os.path.exists(source_view_poses_file):
        print(f"错误：未提供源视图姿态文件或文件不存在: {source_view_poses_file}")
        files_ok = False
    
    if not target_view_pose_file or not os.path.exists(target_view_pose_file):
        print(f"错误：未提供目标视图姿态文件或文件不存在: {target_view_pose_file}")
        files_ok = False
    
    if not files_ok:
        print("\nEpipolar策略调试需要提供以下文件:")
        print("  1. sample_locations_valid.npy - 3D采样点位置")
        print("  2. source_view_poses.npy - 源视图相机姿态")
        print("  3. target_view_pose.npy - 目标视图相机姿态")
        print("\n请提供所有必需文件并重试。")
        return
    
    print(f"视锥体大小: {frustum_size}")
    print(f"深度层数: {depth_layers}")
    print(f"最大可视化帧数: {max_frames}")
    print(f"样本位置文件: {sample_locations_file}")
    print(f"源视图姿态文件: {source_view_poses_file}")
    print(f"目标视图姿态文件: {target_view_pose_file}")
    
    # 运行仿真
    run_nerf_accelerator_simulation(
        memory_access_strategy="epipolar",
        sample_locations_file=sample_locations_file,
        source_view_poses_file=source_view_poses_file,
        target_view_pose_file=target_view_pose_file,
        frustum_size=frustum_size,
        depth_layers=depth_layers,
        epipolar_debug=True,
        max_epipolar_frames=max_frames
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行Epipolar策略的3D可视化调试")
    parser.add_argument("--sample-locations", required=True, help="sample_locations_valid.npy文件路径")
    parser.add_argument("--source-poses", required=True, help="source_view_poses.npy文件路径")
    parser.add_argument("--target-pose", required=True, help="target_view_pose.npy文件路径")
    parser.add_argument("--frustum-size", type=int, nargs=2, default=[16, 16], help="视锥体大小 (高度 宽度)")
    parser.add_argument("--depth-layers", type=int, default=8, help="深度层数")
    parser.add_argument("--max-frames", type=int, default=10, help="最大调试帧数")
    
    args = parser.parse_args()
    
    debug_epipolar_strategy(
        sample_locations_file=args.sample_locations,
        source_view_poses_file=args.source_poses,
        target_view_pose_file=args.target_pose,
        frustum_size=tuple(args.frustum_size),
        depth_layers=args.depth_layers,
        max_frames=args.max_frames
    )