import numpy as np
import time
from nerf_accelerator_sim import Ray, SamplePoint, SourceView, NeRFAcceleratorSimulator
from ddr3_memory_model import DDR3Config, SourceViewStorageConfig
from memory_scheduler import MemoryAccessStrategy

def run_nerf_accelerator_simulation(memory_access_strategy="baseline", enable_debug=False, debug_threshold=10, debug_start_after=5,
                                  sample_locations_file=None, source_view_poses_file=None, target_view_pose_file=None,
                                  frustum_size=(16, 16), depth_layers=8, epipolar_debug=False, max_epipolar_frames=10):
    # Ray grid layout parameters
    ray_grid_height = 32
    ray_grid_width = 1008
    
    # Load actual projection coordinate data
    # Shape is [number_of_source_views, number_of_rays, number_of_samples, 2]
    try:
        print("Loading projection coordinate data...")
        projection_coords = np.load('../eval/pixel_locations_valid_hr.npy')
        print(f"Coordinate data loaded successfully, shape: {projection_coords.shape}")
        
        num_source_views = projection_coords.shape[0]  # First dimension is source views
        num_rays = projection_coords.shape[1]  # Second dimension is rays (flattened H*W)
        num_samples_per_ray = projection_coords.shape[2]  # Third dimension is samples
        
        print(f"Number of source views: {num_source_views}")
        print(f"Total rays: {num_rays} (grid: {ray_grid_height}×{ray_grid_width})")
        print(f"Samples per ray: {num_samples_per_ray}")
        
        # Verify ray count matches grid size
        if ray_grid_height * ray_grid_width != num_rays:
            print(f"Warning: Ray count ({num_rays}) doesn't match grid size ({ray_grid_height}×{ray_grid_width}={ray_grid_height*ray_grid_width})")
            # Adjust grid size to match actual ray count
            ray_grid_height = int(np.sqrt(num_rays))
            ray_grid_width = int(np.ceil(num_rays / ray_grid_height))
            print(f"Adjusted grid size to: {ray_grid_height}×{ray_grid_width}")
    except Exception as e:
        print(f"Error loading coordinate data: {e}")
        print("Continuing with simulated data...")
        # Use default values
        num_source_views = 10
        num_rays = ray_grid_height * ray_grid_width
        num_samples_per_ray = 64
        projection_coords = None
    
    # Source view image dimensions
    # After 4x downsampling: 1008/4 = 252, 756/4 = 189
    image_width = 252   # Width after downsampling (original 1008)
    image_height = 189  # Height after downsampling (original 756)
    feature_dim = 32
    downsample_factor = 4  # 4x downsampling factor
    
    print(f"Source view dimensions: {image_width}×{image_height}×{feature_dim}")
    print(f"Original dimensions (before {downsample_factor}x downsampling): {image_width*downsample_factor}×{image_height*downsample_factor}")
    
    # DDR3 configuration
    ddr_config = DDR3Config(
        num_banks=8,
        num_rows=65536,
        num_columns=1024,
        device_width=64,
        burst_length=8,
        tRCD=13,  # 13ns
        tRP=13,   # 13ns
        tCL=13,   # 13ns
        clock_period=1.25  # DDR3-1600
    )
    
    # Storage configuration
    storage_config = SourceViewStorageConfig(
        block_size=(4, 4),        # Now using 4×4 blocks instead of 16×16
        feature_dim=feature_dim,
        data_width=8,
        downsample_factor=downsample_factor
    )
    
    print("Creating simulation data structures...")
    
    # Create source views
    source_views = []
    for i in range(num_source_views):
        source_view = SourceView(
            id=i,
            height=image_height,        # Downsampled height
            width=image_width,          # Downsampled width
            feature_dim=feature_dim,
            camera_params={
                "position": np.random.rand(3),
                "rotation": np.random.rand(3, 3)
            }
        )
        source_views.append(source_view)
    
    # Create rays and sample points - maintain grid layout
    rays = []
    ray_grid = []  # Store rays in 2D grid
    
    # Initialize ray grid
    for h in range(ray_grid_height):
        ray_row = []
        for w in range(ray_grid_width):
            ray_idx = h * ray_grid_width + w
            if ray_idx < num_rays:
                # Create ray
                ray = Ray(
                    id=ray_idx,
                    origin=np.random.rand(3),  # Virtual origin (doesn't affect simulation)
                    direction=np.random.rand(3),  # Virtual direction (doesn't affect simulation)
                    # Add grid position information
                    grid_y=h,
                    grid_x=w
                )
                ray_row.append(ray)
                rays.append(ray)
            else:
                ray_row.append(None)
        ray_grid.append(ray_row)
    
    # Add sample points to each ray
    for ray in rays:
        i = ray.id  # Ray index
        
        # Add sample points
        for j in range(num_samples_per_ray):
            # Create sample point
            sample_point = SamplePoint(
                id=j,
                position=np.random.rand(3)  # Virtual position (doesn't affect simulation)
            )
            
            # Set projection coordinates
            for k in range(num_source_views):
                if projection_coords is not None:
                    # Use actual projection coordinates, note dimension order is [source_views, rays, samples, 2]
                    # Coordinates are in original space (before downsampling)
                    sample_point.coordinates[k] = projection_coords[k, i, j]
                else:
                    # Use random projection coordinates (for testing only)
                    # Generate in original space
                    px = np.random.randint(0, image_width * downsample_factor)
                    py = np.random.randint(0, image_height * downsample_factor)
                    sample_point.coordinates[k] = np.array([px, py])
                
            ray.sample_points.append(sample_point)
    
    # 创建仿真器
    print("初始化仿真器...")
    print(f"使用内存访问策略: {memory_access_strategy}")
    simulator = NeRFAcceleratorSimulator(
        rays=rays,
        source_views=source_views,
        ddr_config=ddr_config,
        storage_config=storage_config,
        ray_grid_dims=(ray_grid_height, ray_grid_width),
        memory_access_strategy=memory_access_strategy
    )
    
    # 如果使用epipolar策略则加载3D数据
    if memory_access_strategy == "epipolar":
        if all([sample_locations_file, source_view_poses_file, target_view_pose_file]):
            print("正在为epipolar策略加载3D数据")
            success = simulator.load_3d_data(
                sample_locations_file=sample_locations_file,
                source_view_poses_file=source_view_poses_file,
                target_view_pose_file=target_view_pose_file
            )
            if not success:
                print("加载3D数据失败。退回到baseline策略。")
                memory_access_strategy = "baseline"
                simulator.strategy = MemoryAccessStrategy.BASELINE
        else:
            print("Epipolar策略需要3D数据文件。退回到baseline策略。")
            memory_access_strategy = "baseline"
            simulator.strategy = MemoryAccessStrategy.BASELINE
    
    # 启用调试模式
    if enable_debug:
        print(f"调试模式已启用: 阈值={debug_threshold}, 在处理{debug_start_after}个子块后开始")
        simulator.read_scheduler.enable_debug(debug_threshold, debug_start_after)
    
    # 如果启用了epipolar调试，配置它
    if memory_access_strategy == "epipolar" and epipolar_debug:
        print(f"Epipolar调试模式已启用: 最多生成{max_epipolar_frames}帧")
        simulator.read_scheduler.enable_epipolar_debug(True, max_epipolar_frames)
    
    # Set parameters based on strategy
    if memory_access_strategy == "baseline":
        print("Note: Baseline strategy ignores sub_batch_size and samples_per_batch parameters")
    
    # Run simulation with parameters appropriate for the strategy
    frustum_size = (16, 16)
    depth_layers = 8
    
    if memory_access_strategy == "epipolar":
        print(f"Epipolar strategy parameters: frustum_size={frustum_size}, depth_layers={depth_layers}")
    
    # Run simulation
    print("Starting simulation...")
    start_time = time.time()
    results = simulator.run_simulation(
        ray_batch_size=(16, 16),  # Main block size
        sub_batch_size=(8, 8),    # Sub-block size (for spatial locality)
        samples_per_batch=8       # Samples per batch (for spatial locality)
    )
    simulation_time = time.time() - start_time
    
    # 打印结果
    print("\n仿真结果:")
    print(f"内存访问次数: {results['total_memory_accesses']}")
    print(f"内存访问周期: {results['total_memory_cycles']}")
    print(f"处理的内存块数: {results['blocks_processed']}")
    print(f"DDR总读取量: {results['total_mb_read']:.2f} MB")
    print(f"行命中率: {results['row_hit_rate']:.2f}%")
    print(f"总仿真时间: {simulation_time:.2f} 秒")
    
    # DDR3统计
    print("\nDDR3内存统计:")
    print(f"总读取次数: {results['ddr_stats']['total_reads']}")
    print(f"行激活次数: {results['ddr_stats']['total_row_activations']}")
    print(f"预充电次数: {results['ddr_stats']['total_precharges']}")
    print(f"读取burst数: {results['ddr_stats']['total_bursts']}")
    print(f"平均每次读取延迟: {results['ddr_stats']['average_latency_per_read']:.2f} cycles")
    
    if enable_debug:
        print("\n调试信息已保存到 debug_output 目录")
    
    return results

# 修改命令行参数解析
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="运行NeRF加速器仿真")
    parser.add_argument("--strategy", default="baseline", choices=["baseline", "spatial", "epipolar"],
                      help="使用的内存访问策略")
    parser.add_argument("--debug", action="store_true", help="启用普通调试模式")
    parser.add_argument("--threshold", type=int, default=10, help="调试读取计数阈值")
    parser.add_argument("--start-after", type=int, default=5, help="在处理N个子块后开始调试")
    parser.add_argument("--sample-locations", help="sample_locations_valid.npy文件路径")
    parser.add_argument("--source-poses", help="source_view_poses.npy文件路径")
    parser.add_argument("--target-pose", help="target_view_pose.npy文件路径")
    parser.add_argument("--frustum-size", type=int, nargs=2, default=[16, 16], help="视锥体大小 (高度 宽度)")
    parser.add_argument("--depth-layers", type=int, default=8, help="深度层数")
    parser.add_argument("--epipolar-debug", action="store_true", help="启用epipolar调试可视化")
    parser.add_argument("--max-frames", type=int, default=10, help="最大epipolar调试帧数")
    
    args = parser.parse_args()
    
    # 如果策略是"spatial"，则转换为内部使用的"spatial_locality"
    strategy = args.strategy
    if strategy == "spatial":
        strategy = "spatial_locality"
    
    run_nerf_accelerator_simulation(
        memory_access_strategy=strategy,
        enable_debug=args.debug,
        debug_threshold=args.threshold,
        debug_start_after=args.start_after,
        sample_locations_file=args.sample_locations,
        source_view_poses_file=args.source_poses,
        target_view_pose_file=args.target_pose,
        frustum_size=tuple(args.frustum_size),
        depth_layers=args.depth_layers,
        epipolar_debug=args.epipolar_debug,
        max_epipolar_frames=args.max_frames
    )