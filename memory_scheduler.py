import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
from camera_utils import Camera, point_in_frustum

from ddr3_memory_model import DDR3Memory, coordinate_to_block, SourceViewStorageConfig
from epipolar_debug_utils import create_debug_directory, visualize_frustum_and_points, create_summary_visualization

# 定义内存访问策略枚举
class MemoryAccessStrategy(Enum):
    BASELINE = "baseline"             # Process each point individually
    SPATIAL_LOCALITY = "spatial_locality"  # Use spatial locality of coordinates
    EPIPOLAR = "epipolar"             # Use epipolar geometry and 3D frustums

@dataclass
class RayBatch:
    """光线批处理组"""
    rays: List                                      # 组内光线
    samples: List[int]                              # 处理的sample ID列表
    source_views: List[int]                         # 处理的source view ID列表
    block_access_map: Dict[Tuple[int, int, int], Set[Tuple[int, int]]] = field(default_factory=dict)  
    # {(source_view_id, block_y, block_x): {(ray_id, sample_id)}}

@dataclass
class Block:
    """内存块"""
    source_view_id: int                            # 源视图ID
    block_x: int                                   # X方向块坐标
    block_y: int                                   # Y方向块坐标
    data: np.ndarray                               # 块数据

@dataclass
class MemoryAccessInfo:
    """内存访问信息"""
    block: Block                                   # 访问的块
    ray_sample_tuples: Set[Tuple[int, int]]        # 使用该块的(ray_id, sample_id)集合
    latency_cycles: int                            # 访问延迟(时钟周期)

@dataclass
class SimulationResult:
    """仿真结果"""
    total_memory_accesses: int = 0                 # 总内存访问次数
    total_memory_cycles: int = 0                   # 总内存访问周期
    total_bytes_read: int = 0                      # 总读取字节数
    row_hit_rate: float = 0.0                      # 行命中率
    simulation_runtime_s: float = 0.0              # 仿真运行时间(秒)
    blocks_processed: int = 0                      # 处理的块数量
    
    def __add__(self, other):
        """实现结果累加"""
        if not isinstance(other, SimulationResult):
            return self
            
        result = SimulationResult()
        result.total_memory_accesses = self.total_memory_accesses + other.total_memory_accesses
        result.total_memory_cycles = self.total_memory_cycles + other.total_memory_cycles
        result.total_bytes_read = self.total_bytes_read + other.total_bytes_read
        # 行命中率需要加权平均
        if self.total_memory_accesses + other.total_memory_accesses > 0:
            result.row_hit_rate = (self.row_hit_rate * self.total_memory_accesses + 
                                  other.row_hit_rate * other.total_memory_accesses) / (
                                  self.total_memory_accesses + other.total_memory_accesses)
        result.simulation_runtime_s = self.simulation_runtime_s + other.simulation_runtime_s
        result.blocks_processed = self.blocks_processed + other.blocks_processed
        return result

class ReadScheduler:
    """内存读取调度器"""
    def __init__(self, rays, source_views, ddr_memory, storage_config, strategy=MemoryAccessStrategy.BASELINE):
        self.rays = rays
        self.source_views = source_views
        self.ddr_memory = ddr_memory
        self.storage_config = storage_config
        self.strategy = strategy
        
        # 调试设置
        self.debug_mode = False
        self.debug_threshold = 10  # 读取次数阈值，超过此值启用调试
        self.debug_subblocks_processed = 0  # 已处理子块数，用于延迟启用调试
        self.debug_start_after = 5  # 处理多少子块后启用调试
        self.debug_enabled = False
        self.debug_dir = "debug_output"
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        # 计算视场大小和块数
        self.max_height = max(sv.height for sv in source_views)
        self.max_width = max(sv.width for sv in source_views)
        self.num_blocks_y = (self.max_height + storage_config.block_size[0] - 1) // storage_config.block_size[0]
        self.num_blocks_x = (self.max_width + storage_config.block_size[1] - 1) // storage_config.block_size[1]
        
        # 初始化DDR3内存映射
        self.ddr_memory.initialize_mapping(source_views, (self.num_blocks_x, self.num_blocks_y))
        
        # 构建光线网格
        grid_height = getattr(self, 'ray_grid_height', 32) 
        grid_width = getattr(self, 'ray_grid_width', 504)
        self.ray_grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
        
        for ray in rays:
            if hasattr(ray, 'grid_y') and hasattr(ray, 'grid_x'):
                y, x = ray.grid_y, ray.grid_x
                if 0 <= y < grid_height and 0 <= x < grid_width:
                    self.ray_grid[y][x] = ray
    
    def enable_debug(self, threshold=10, start_after=5):
        """启用调试模式"""
        self.debug_mode = True
        self.debug_threshold = threshold
        self.debug_start_after = start_after
        print(f"调试模式已启用: 阈值={threshold}, 在处理{start_after}个子块后激活")

    def enable_epipolar_debug(self, enable=True, max_frames=10):
        """
        启用epipolar策略的调试模式
        
        参数:
            enable: 是否启用调试
            max_frames: 最大保存的帧数 (视锥体+点云)
        """
        self.epipolar_debug = enable
        self.max_epipolar_debug_frames = max_frames
        self.epipolar_debug_count = 0
        self.epipolar_debug_stats = []
        self.epipolar_debug_frustum_coords = []
        
        if enable:
            self.epipolar_debug_dir = create_debug_directory()
            print(f"Epipolar调试模式已启用，最多保存{max_frames}帧，输出目录: {self.epipolar_debug_dir}")
        
    def set_camera_info(self, source_cameras, target_camera):
        """
        Set camera information for epipolar strategy
        
        Args:
            source_cameras: List of Camera objects for source views
            target_camera: Camera object for target view
        """
        self.source_cameras = source_cameras
        self.target_camera = target_camera
        print(f"Camera information set: {len(source_cameras)} source cameras and 1 target camera")
    
    def process_all_rays(self, ray_batch_size=(16, 16), sub_batch_size=(8, 8), samples_per_batch=8) -> SimulationResult:
        """
        Process all rays based on the selected memory access strategy
        
        Parameters:
            ray_batch_size: (height, width) Main block size
            sub_batch_size: (height, width) Sub-block size (for spatial locality)
            samples_per_batch: Samples per batch
                
        Returns:
            result: Simulation result
        """
        start_time = time.time()
        result = SimulationResult()
        
        if self.strategy == MemoryAccessStrategy.EPIPOLAR:
            # For epipolar strategy, use the 3D frustum-based approach
            if not hasattr(self, 'target_camera') or not hasattr(self, 'source_cameras'):
                print("Warning: Epipolar strategy requires camera information. Falling back to baseline.")
                self.strategy = MemoryAccessStrategy.BASELINE
            else:
                result = self._process_epipolar(ray_batch_size, samples_per_batch)
                result.simulation_runtime_s = time.time() - start_time
                return result
        
        # Original grid-based processing for baseline and spatial locality strategies
        grid_height = len(self.ray_grid)
        grid_width = len(self.ray_grid[0]) if grid_height > 0 else 0
        
        # Process each main block
        for i in range(0, grid_height, ray_batch_size[0]):
            for j in range(0, grid_width, ray_batch_size[1]):
                # Extract rays from current main block
                block_rays = []
                for di in range(min(ray_batch_size[0], grid_height - i)):
                    for dj in range(min(ray_batch_size[1], grid_width - j)):
                        ray = self.ray_grid[i+di][j+dj]
                        if ray is not None:
                            block_rays.append(ray)
                
                if not block_rays:
                    continue
                
                # Process main block
                print(f"Processing main block ({i},{j}) -> ({i+ray_batch_size[0]-1},{j+ray_batch_size[1]-1}), {len(block_rays)} rays")
                block_result = self._process_block(block_rays, sub_batch_size, samples_per_batch)
                
                # Update statistics
                result = result + block_result
                
                # Print current block memory usage
                print(f"  Block processing complete - Memory read: {block_result.total_bytes_read / (1024*1024):.2f} MB, "
                    f"Row hit rate: {block_result.row_hit_rate*100:.2f}%, "
                    f"Memory accesses: {block_result.total_memory_accesses}")
        
        # Add total runtime
        result.simulation_runtime_s = time.time() - start_time
        
        return result

    def _process_epipolar(self, frustum_size=(16, 16), depth_layers=8) -> SimulationResult:
        """
        使用epipolar策略处理光线，基于3D视锥体
        
        参数:
            frustum_size: (height, width) 目标视图上每个视锥体块的大小
            depth_layers: 将深度范围划分为多少层
                
        返回:
            result: 仿真结果
        """
        result = SimulationResult()
        
        # 使用光线网格尺寸而不是假设的目标视图尺寸
        grid_height = len(self.ray_grid)
        grid_width = len(self.ray_grid[0]) if grid_height > 0 else 0
        
        if grid_height == 0 or grid_width == 0:
            print("错误：光线网格为空，无法进行epipolar策略处理。")
            return result
        
        # 检查是否启用了epipolar调试
        epipolar_debug_enabled = hasattr(self, 'epipolar_debug') and self.epipolar_debug
        
        # 计算所有采样点的最小/最大深度（在目标视图的相机空间中）
        all_points = []
        point_to_ray_sample = {}  # 用于快速查找点所属的(ray_id, sample_id)
        
        # 构建ray_id的快速查找表
        ray_id_to_index = {ray.id: i for i, ray in enumerate(self.rays)}
        
        # 从所有光线中收集3D采样点
        for ray in self.rays:
            for sample_idx, sample in enumerate(ray.sample_points):
                if hasattr(sample, 'position') and sample.position is not None:
                    all_points.append(sample.position)
                    # 存储每个点对应的光线和样本索引
                    point_tuple = tuple(sample.position.tolist())
                    point_to_ray_sample[point_tuple] = (ray.id, sample_idx)
        
        if not all_points:
            print("没有找到有效的3D采样点，无法进行epipolar策略处理。")
            return result
        
        all_points = np.array(all_points)
        
        # 将点投影到目标相机并获取深度
        _, target_depths = self.target_camera.project_points(all_points)
        
        min_depth = np.min(target_depths)
        max_depth = np.max(target_depths)
        depth_range = max_depth - min_depth
        
        print(f"深度范围: {min_depth:.2f} 到 {max_depth:.2f} (范围: {depth_range:.2f})")
        
        # 计算视锥体块网格尺寸 - 基于光线网格而不是目标视图尺寸
        frustum_height, frustum_width = frustum_size
        num_frustum_y = (grid_height + frustum_height - 1) // frustum_height
        num_frustum_x = (grid_width + frustum_width - 1) // frustum_width
        
        print(f"将光线网格 ({grid_height}x{grid_width}) 划分为 {num_frustum_y}x{num_frustum_x} 个视锥体块")
        print(f"每个视锥体块大小为 {frustum_height}x{frustum_width}，共有 {depth_layers} 个深度层")
        
        # 处理每个视锥体块
        for fy in range(num_frustum_y):
            for fx in range(num_frustum_x):
                # 计算此视锥体在光线网格中的范围
                grid_start_y = fy * frustum_height
                grid_start_x = fx * frustum_width
                grid_end_y = min(grid_start_y + frustum_height, grid_height)
                grid_end_x = min(grid_start_x + frustum_width, grid_width)
                
                # 检查块大小是否有效
                block_h = grid_end_y - grid_start_y
                block_w = grid_end_x - grid_start_x
                
                if block_h <= 0 or block_w <= 0:
                    continue
                
                print(f"处理视锥体块 ({fx},{fy}) 对应网格范围 ({grid_start_y},{grid_start_x}) -> "
                    f"({grid_end_y-1},{grid_end_x-1})，大小 {block_h}x{block_w}")
                
                # 收集此块中的所有光线
                block_rays = []
                for y in range(grid_start_y, grid_end_y):
                    for x in range(grid_start_x, grid_end_x):
                        ray = self.ray_grid[y][x]
                        if ray is not None:
                            block_rays.append(ray)
                
                # 如果块中没有光线则跳过
                if not block_rays:
                    continue
                
                # 收集所有块内光线的采样点（用于调试）
                if epipolar_debug_enabled:
                    block_sample_points = []
                    for ray in block_rays:
                        for sample in ray.sample_points:
                            if hasattr(sample, 'position') and sample.position is not None:
                                block_sample_points.append(sample.position)
                    block_sample_points = np.array(block_sample_points) if block_sample_points else np.empty((0, 3))
                
                # 处理每个深度层
                for layer in range(depth_layers):
                    # 计算此层的深度范围
                    layer_depth_min = min_depth + layer * (depth_range / depth_layers)
                    layer_depth_max = min_depth + (layer + 1) * (depth_range / depth_layers)
                    
                    # 对于调试：计算视锥体角点
                    if epipolar_debug_enabled:
                        # 为调试目的，我们需要计算视锥体角点
                        # 确保角点按照正确的顺序生成
                        
                        # 获取相机位置
                        camera_position = self.target_camera.position
                        
                        # 找到定义视锥体的四个角光线
                        corner_rays = []
                        corner_positions = []
                        
                        # 按照特定顺序获取四个角点: 左下, 右下, 右上, 左上
                        corner_coords = [
                            (grid_start_x, grid_start_y),              # 左下
                            (grid_end_x - 1, grid_start_y),            # 右下
                            (grid_end_x - 1, grid_end_y - 1),          # 右上
                            (grid_start_x, grid_end_y - 1)             # 左上
                        ]
                        
                        for x, y in corner_coords:
                            if 0 <= y < grid_height and 0 <= x < grid_width:
                                ray = self.ray_grid[y][x]
                                if ray is not None:
                                    corner_rays.append(ray)
                                    corner_positions.append((x, y))
                        
                        frustum_corners = None
                        
                        # 如果我们能获得足够的角点射线，创建视锥体
                        if len(corner_rays) >= 4:
                            # 计算射线方向
                            ray_dirs = []
                            for ray in corner_rays:
                                ray_dir = self.target_camera.ray_direction(ray.grid_x, ray.grid_y)
                                ray_dirs.append(ray_dir)
                            
                            # 创建视锥体角点 - 确保正确的顺序
                            frustum_corners = []
                            
                            # 先添加近平面四个点
                            for direction in ray_dirs:
                                near_point = camera_position + direction * layer_depth_min
                                frustum_corners.append(near_point)
                            
                            # 再添加远平面四个点 (保持相同顺序)
                            for direction in ray_dirs:
                                far_point = camera_position + direction * layer_depth_max
                                frustum_corners.append(far_point)
                                
                            frustum_corners = np.array(frustum_corners)
                    
                    # 查找此视锥体+深度范围内的所有采样点
                    # 优化：只从当前块的光线中查找
                    points_in_frustum = []
                    ray_sample_tuples = []
                    
                    for ray in block_rays:
                        for sample_idx, sample in enumerate(ray.sample_points):
                            if hasattr(sample, 'position') and sample.position is not None:
                                # 将点投影到目标相机中检查深度
                                cam_point = self.target_camera.world_to_camera(sample.position.reshape(1, 3))
                                depth = cam_point[0, 2]
                                
                                # 检查点是否在当前深度层范围内
                                if layer_depth_min <= depth <= layer_depth_max:
                                    points_in_frustum.append(sample.position)
                                    ray_sample_tuples.append((ray.id, sample_idx))
                    
                    if not points_in_frustum:
                        # 此视锥体+深度层中没有点，跳过
                        continue
                    
                    # 转换为numpy数组进行批处理
                    points_in_frustum = np.array(points_in_frustum)
                    
                    # 调试可视化
                    if epipolar_debug_enabled and self.epipolar_debug_count < self.max_epipolar_debug_frames:
                        layer_info = f"Block({fx},{fy}) Depth{layer+1}/{depth_layers} ({layer_depth_min:.2f}-{layer_depth_max:.2f})"
                        output_file = f"{self.epipolar_debug_dir}/epipolar_debug_{self.epipolar_debug_count:03d}.png"
                        
                        # 查找不在视锥体内的其他点
                        if len(block_sample_points) > 0:
                            if len(points_in_frustum) > 0:
                                # 找出不在视锥体内的点
                                in_frustum_list = points_in_frustum.tolist()
                                outside_points = []
                                for point in block_sample_points:
                                    if point.tolist() not in in_frustum_list:
                                        outside_points.append(point)
                                outside_points = np.array(outside_points) if outside_points else np.empty((0, 3))
                            else:
                                outside_points = block_sample_points
                        else:
                            outside_points = np.empty((0, 3))
                        
                        # 保存调试统计信息
                        debug_stats = {
                            'frustum_pos': (fx, fy),
                            'layer': layer,
                            'depth_range': (layer_depth_min, layer_depth_max),
                            'points_in_frustum': len(points_in_frustum),
                            'points_outside': len(outside_points)
                        }
                        self.epipolar_debug_stats.append(debug_stats)
                        self.epipolar_debug_frustum_coords.append((fx, fy))
                        
                        # 生成可视化
                        if frustum_corners is not None:
                            visualize_frustum_and_points(
                                frustum_corners, 
                                points_in_frustum, 
                                points_outside=outside_points,
                                camera_position=camera_position,
                                layer_info=layer_info,
                                output_file=output_file
                            )
                            self.epipolar_debug_count += 1
                    
                    print(f"  深度层 {layer+1}/{depth_layers} ({layer_depth_min:.2f}-{layer_depth_max:.2f}): "
                        f"{len(points_in_frustum)} 个采样点")
                    
                    # 处理每个源视图
                    for sv_idx, source_view_id in enumerate(range(len(self.source_views))):
                        if sv_idx >= len(self.source_cameras):
                            continue
                        
                        source_camera = self.source_cameras[sv_idx]
                        source_view = self.source_views[sv_idx]
                        
                        # 将点投影到此源视图
                        pixels, _ = source_camera.project_points(points_in_frustum)
                        
                        # 过滤掉投影到源视图外部的点
                        valid_mask = ((0 <= pixels[:, 0]) & (pixels[:, 0] < source_view.width) &
                                    (0 <= pixels[:, 1]) & (pixels[:, 1] < source_view.height))
                        
                        if not np.any(valid_mask):
                            continue
                        
                        valid_pixels = pixels[valid_mask]
                        valid_tuples = [ray_sample_tuples[i] for i in np.where(valid_mask)[0]]
                        
                        # 查找投影点的边界矩形
                        min_x = np.min(valid_pixels[:, 0])
                        max_x = np.max(valid_pixels[:, 0])
                        min_y = np.min(valid_pixels[:, 1])
                        max_y = np.max(valid_pixels[:, 1])
                        
                        # 转换为块坐标（应用降采样）
                        downsample = self.storage_config.downsample_factor
                        min_block_x, min_block_y = coordinate_to_block(
                            [min_x * downsample, min_y * downsample], 
                            self.storage_config.block_size, 
                            downsample)
                        
                        max_block_x, max_block_y = coordinate_to_block(
                            [max_x * downsample, max_y * downsample], 
                            self.storage_config.block_size,
                            downsample)
                        
                        # 创建用于内存访问的批次
                        batch = RayBatch(
                            rays=[self.rays[ray_id_to_index[ray_id]] for ray_id, _ in valid_tuples if ray_id in ray_id_to_index],
                            samples=list(range(max(sample_idx for _, sample_idx in valid_tuples) + 1)),
                            source_views=[source_view_id]
                        )
                        
                        # 准备块访问映射
                        batch.block_access_map = {}
                        
                        # 添加块到访问映射
                        for block_y in range(min_block_y, max_block_y + 1):
                            for block_x in range(min_block_x, max_block_x + 1):
                                block_key = (source_view_id, block_y, block_x)
                                batch.block_access_map[block_key] = set(valid_tuples)
                        
                        # 执行内存访问
                        if batch.block_access_map:
                            pre_reads = self.ddr_memory.total_reads
                            access_infos = self._execute_batch_memory_access(batch)
                            batch_reads = self.ddr_memory.total_reads - pre_reads
                            
                            print(f"    源视图 {source_view_id}: 访问了 {len(batch.block_access_map)} 个块， "
                                f"覆盖 {len(valid_tuples)} 个采样点，共 {batch_reads} 次读取")
                            
                            # 更新统计信息
                            result.total_memory_accesses += len(access_infos)
                            result.total_memory_cycles += sum(info.latency_cycles for info in access_infos)
                            result.blocks_processed += len(access_infos)
                            
                            # 计算此批次读取的字节数
                            batch_bytes_read = 0
                            for access_info in access_infos:
                                block = access_info.block
                                if block.data is not None:
                                    batch_bytes_read += block.data.nbytes
                            
                            result.total_bytes_read += batch_bytes_read
        
        # 如果启用了调试，创建摘要可视化
        if epipolar_debug_enabled and self.epipolar_debug_stats:
            summary_file = f"{self.epipolar_debug_dir}/epipolar_summary.png"
            create_summary_visualization(
                self.epipolar_debug_stats, 
                self.epipolar_debug_frustum_coords, 
                summary_file
            )
            print(f"Epipolar调试摘要已保存至: {summary_file}")
            
            # 输出统计信息
            total_points = sum(stats['points_in_frustum'] for stats in self.epipolar_debug_stats)
            print(f"\nEpipolar调试统计:")
            print(f"  已生成 {self.epipolar_debug_count} 个视锥体可视化")
            print(f"  总计 {total_points} 个采样点在所有可视化视锥体内")
            print(f"  调试输出目录: {self.epipolar_debug_dir}")
        
        # 获取DDR统计
        ddr_stats = self.ddr_memory.get_statistics()
        result.row_hit_rate = ddr_stats["row_hit_rate"]
        
        return result
        
    def _process_block(self, block_rays, sub_batch_size, samples_per_batch) -> SimulationResult:
        """
        处理一个主块中的光线
        
        参数:
            block_rays: 块内的光线
            sub_batch_size: 子批次大小(用于spatial策略)
            samples_per_batch: 每批采样点数
            
        返回:
            result: 块处理结果
        """
        if self.strategy == MemoryAccessStrategy.BASELINE:
            return self._process_block_baseline(block_rays, samples_per_batch)
        else:
            return self._process_block_spatial(block_rays, sub_batch_size, samples_per_batch)
    
    def _process_block_baseline(self, block_rays, samples_per_batch) -> SimulationResult:
        """
        基线策略：直接处理每个光线的所有样本点、源视图组合
        
        参数:
            block_rays: 块内的光线
            samples_per_batch: 这个参数对baseline不使用，保留是为了接口一致性
            
        返回:
            result: 块处理结果
        """
        result = SimulationResult()
        
        # 获取样本点总数
        num_samples = max([len(ray.sample_points) for ray in block_rays]) if block_rays else 0
        
        # 对于baseline策略，一次性处理所有样本点
        # 创建一个包含所有样本点的批次
        batch = RayBatch(
            rays=block_rays,
            samples=list(range(num_samples)),  # 所有样本点
            source_views=[sv.id for sv in self.source_views]
        )
        
        # 收集此批次中所有坐标对应的块
        self._collect_block_access_baseline(batch)
        
        # 执行内存访问
        access_infos = self._execute_batch_memory_access(batch)
        
        # 更新统计信息
        result.total_memory_accesses += len(access_infos)
        result.total_memory_cycles += sum(info.latency_cycles for info in access_infos)
        result.blocks_processed += len(access_infos)
        
        # 计算本批次读取的字节数
        batch_bytes_read = 0
        for access_info in access_infos:
            block = access_info.block
            if block.data is not None:
                batch_bytes_read += block.data.nbytes
        
        result.total_bytes_read += batch_bytes_read
        
        # 获取DDR统计
        ddr_stats = self.ddr_memory.get_statistics()
        result.row_hit_rate = ddr_stats["row_hit_rate"]
        
        return result
    
    def _process_block_spatial(self, block_rays, sub_batch_size, samples_per_batch) -> SimulationResult:
        """
        空间局部性策略：按子块和样本点批次处理
        
        参数:
            block_rays: 块内的光线
            sub_batch_size: 子批次大小
            samples_per_batch: 每批采样点数
            
        返回:
            result: 块处理结果
        """
        result = SimulationResult()
        
        # 创建射线ID映射，用于快速查找
        ray_dict = {ray.id: ray for ray in block_rays}
        
        # 如果block_rays没有网格位置信息，则按照ID排序
        if not (hasattr(block_rays[0], 'grid_y') and hasattr(block_rays[0], 'grid_x')):
            # 简单地按ID组织成网格
            ray_ids = sorted([ray.id for ray in block_rays])
            grid_size = int(np.ceil(np.sqrt(len(ray_ids))))
            ray_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
            
            for i, ray_id in enumerate(ray_ids):
                row = i // grid_size
                col = i % grid_size
                if row < grid_size and col < grid_size:
                    ray_grid[row][col] = ray_dict[ray_id]
        else:
            # 使用已有的网格位置
            min_grid_y = min(ray.grid_y for ray in block_rays)
            min_grid_x = min(ray.grid_x for ray in block_rays)
            max_grid_y = max(ray.grid_y for ray in block_rays)
            max_grid_x = max(ray.grid_x for ray in block_rays)
            
            grid_height = max_grid_y - min_grid_y + 1
            grid_width = max_grid_x - min_grid_x + 1
            
            ray_grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
            
            for ray in block_rays:
                rel_y = ray.grid_y - min_grid_y
                rel_x = ray.grid_x - min_grid_x
                ray_grid[rel_y][rel_x] = ray
        
        # 获取样本点总数
        num_samples = max([len(ray.sample_points) for ray in block_rays]) if block_rays else 0
        
        # 子块统计计数器 - 真正的子块是8×8 rays + 8 samples
        total_sub_blocks = 0
        total_initial_reads = 0  # 平均坐标读取
        total_extra_reads = 0    # 未覆盖坐标额外读取
        
        # 按子块和样本批次处理
        grid_height = len(ray_grid)
        grid_width = len(ray_grid[0]) if grid_height > 0 else 0
        
        # 处理每个子块 - 先按ray子网格迭代
        for i in range(0, grid_height, sub_batch_size[0]):
            for j in range(0, grid_width, sub_batch_size[1]):
                # 提取当前ray子网格中的射线
                sub_block_rays = []
                for di in range(min(sub_batch_size[0], grid_height - i)):
                    for dj in range(min(sub_batch_size[1], grid_width - j)):
                        ray = ray_grid[i+di][j+dj]
                        if ray is not None:
                            sub_block_rays.append(ray)
            
                if not sub_block_rays:
                    continue
            
                # 再按样本批次迭代 - 每8个samples是一个true subblock
                for s_start in range(0, num_samples, samples_per_batch):
                    s_end = min(s_start + samples_per_batch, num_samples)
                    samples = list(range(s_start, s_end))
                    
                    # 这里的一个完整子块 = 8×8 rays + 8 samples
                    total_sub_blocks += 1
                    self.debug_subblocks_processed += 1
                    
                    # 检查是否应该启用调试
                    if self.debug_mode and self.debug_subblocks_processed >= self.debug_start_after:
                        self.debug_enabled = True
                    
                    # 创建批次
                    batch = RayBatch(
                        rays=sub_block_rays,
                        samples=samples,
                        source_views=[sv.id for sv in self.source_views]
                    )
                    
                    # 子块处理前的DDR访问次数
                    pre_reads = self.ddr_memory.total_reads
                    
                    # 利用空间局部性收集块访问
                    if self.debug_enabled:
                        initial_reads, extra_reads = self._collect_block_access_spatial_debug(
                            batch, total_sub_blocks, i, j, s_start
                        )
                    else:
                        initial_reads, extra_reads = self._collect_block_access_spatial_with_stats(batch)
                    
                    total_initial_reads += initial_reads
                    total_extra_reads += extra_reads
                    
                    # 执行内存访问
                    access_infos = self._execute_batch_memory_access(batch)
                    
                    # 子块的内存访问次数
                    sub_block_reads = self.ddr_memory.total_reads - pre_reads
                    
                    # 输出每个精确子块(8×8 rays × 8 samples)的统计
                    ray_count = len(sub_block_rays)
                    sample_count = len(samples)
                    print(f"    子块({ray_count}rays×{sample_count}samples) 在 ({i},{j}) 坐标和样本 {s_start}-{s_end-1}: "
                          f"读取 {sub_block_reads} 次, 初次 {initial_reads}, 额外 {extra_reads}")
                    
                    # 更新统计信息
                    result.total_memory_accesses += len(access_infos)
                    result.total_memory_cycles += sum(info.latency_cycles for info in access_infos)
                    result.blocks_processed += len(access_infos)
                    
                    # 计算本批次读取的字节数
                    batch_bytes_read = 0
                    for access_info in access_infos:
                        block = access_info.block
                        if block.data is not None:
                            batch_bytes_read += block.data.nbytes
                    
                    result.total_bytes_read += batch_bytes_read
        
        # 计算平均值并输出统计信息
        if total_sub_blocks > 0:
            avg_initial_reads = total_initial_reads / total_sub_blocks
            avg_extra_reads = total_extra_reads / total_sub_blocks
            avg_total_reads = (total_initial_reads + total_extra_reads) / total_sub_blocks
            
            # Update output messages in _process_block_spatial
            print(f"    Subblock({ray_count}rays×{sample_count}samples) at ({i},{j}), samples {s_start}-{s_end-1}: "
                f"Reads: {sub_block_reads}, Initial: {initial_reads}, Extra: {extra_reads}")

            # Later in the method, replace the statistics output
            print(f"  Subblock statistics ({total_sub_blocks} subblocks - each 8×8 rays × 8 samples):")
            print(f"    - Average initial (avg coord) reads: {avg_initial_reads:.2f} per subblock")
            print(f"    - Average extra (uncovered coords) reads: {avg_extra_reads:.2f} per subblock")
            print(f"    - Average total reads: {avg_total_reads:.2f} per subblock")
            print(f"    - Initial read ratio: {(total_initial_reads / max(1, total_initial_reads + total_extra_reads) * 100):.2f}%")
        
        # 获取DDR统计
        ddr_stats = self.ddr_memory.get_statistics()
        result.row_hit_rate = ddr_stats["row_hit_rate"]
        
        return result
    
    def _collect_block_access_baseline(self, batch: RayBatch):
        """
        Baseline strategy: Collect block accesses for each coordinate in the batch
        
        Parameters:
            batch: Ray batch
        """
        # Clear previous mapping
        batch.block_access_map = {}
        downsample = self.storage_config.downsample_factor
        
        # Iterate through all rays, samples, and source views
        for ray in batch.rays:
            for sample_id in batch.samples:
                if sample_id >= len(ray.sample_points):
                    continue
                    
                sample = ray.sample_points[sample_id]
                
                for source_view_id in batch.source_views:
                    # Get projection coordinate
                    if source_view_id not in sample.coordinates:
                        continue
                        
                    coord = sample.coordinates[source_view_id]
                    
                    # Find corresponding source view
                    source_view = next((sv for sv in self.source_views if sv.id == source_view_id), None)
                    if not source_view:
                        continue
                    
                    # Check if coordinate is within image bounds
                    x, y = coord
                    if not (0 <= x < source_view.width * downsample and 0 <= y < source_view.height * downsample):
                        # Coordinate is out of image bounds, skip
                        continue
                    
                    # Convert to block coordinates
                    block_x, block_y = coordinate_to_block(
                        coord, self.storage_config.block_size, downsample)
                    
                    # Record block access
                    block_key = (source_view_id, block_y, block_x)
                    if block_key not in batch.block_access_map:
                        batch.block_access_map[block_key] = set()
                    
                    batch.block_access_map[block_key].add((ray.id, sample_id))
    
    def _collect_block_access_spatial_with_stats(self, batch: RayBatch):
        """
        Spatial locality strategy: Smart block access merging with statistics
        Accurately counts for each subblock (8×8 rays × 8 samples)
        
        Parameters:
            batch: Ray batch
            
        Returns:
            (initial_reads, extra_reads): Initial and extra read counts
        """
        # Clear previous mapping
        batch.block_access_map = {}
        downsample = self.storage_config.downsample_factor
        
        initial_reads = 0  # Initial (average coordinate) read count
        extra_reads = 0    # Extra read count
        
        # Process each source view separately
        for source_view_id in batch.source_views:
            source_view = next((sv for sv in self.source_views if sv.id == source_view_id), None)
            if not source_view:
                continue
            
            # Collect all valid projection coordinates in this batch (within image bounds)
            valid_coords = []
            ray_sample_map = {}  # Map from coordinates to (ray_id, sample_id)
            
            for ray in batch.rays:
                for sample_id in batch.samples:
                    if sample_id >= len(ray.sample_points):
                        continue
                        
                    sample = ray.sample_points[sample_id]
                    if source_view_id not in sample.coordinates:
                        continue
                        
                    coord = sample.coordinates[source_view_id]
                    x, y = coord
                    
                    # Check if coordinate is within image bounds (in original space)
                    if not (0 <= x < source_view.width * downsample and 0 <= y < source_view.height * downsample):
                        continue
                        
                    valid_coords.append(coord)
                    coord_tuple = tuple(map(float, coord))  # Convert to hashable type
                    ray_sample_map[coord_tuple] = (ray.id, sample_id)
            
            # If no valid coordinates, continue to next source view
            if not valid_coords:
                continue
            
            # Calculate average coordinate
            avg_coord = np.mean(valid_coords, axis=0)
            
            # Ensure average coordinate is within image bounds (in original space)
            avg_x, avg_y = avg_coord
            avg_x = max(0, min(avg_x, source_view.width * downsample - 1))
            avg_y = max(0, min(avg_y, source_view.height * downsample - 1))
            
            # Get block for average coordinate
            avg_block_x, avg_block_y = coordinate_to_block(
                [avg_x, avg_y], self.storage_config.block_size, downsample)
            
            # Track uncovered coordinates
            uncovered_coords = set(map(tuple, map(lambda c: map(float, c), valid_coords)))  # Convert to set for easy lookup
            
            # Process initial block (average coordinate)
            block_key = (source_view_id, avg_block_y, avg_block_x)
            covered_coords = self._process_block_with_coords_and_track(
                batch, source_view_id, avg_block_y, avg_block_x, 
                list(map(np.array, uncovered_coords)), ray_sample_map)
            
            # Only count as an initial read if coordinates were covered
            if covered_coords:
                initial_reads += 1
            
            # Remove covered coordinates from uncovered set
            for coord in covered_coords:
                coord_tuple = tuple(map(float, coord))
                if coord_tuple in uncovered_coords:
                    uncovered_coords.remove(coord_tuple)
            
            # Process remaining uncovered coordinates - extra reads
            processed_blocks = {(avg_block_y, avg_block_x)}  # Track already processed blocks
            
            while uncovered_coords:
                # Select an uncovered coordinate
                coord = np.array(next(iter(uncovered_coords)))
                
                # Get its corresponding block
                block_x, block_y = coordinate_to_block(coord, self.storage_config.block_size, downsample)
                
                # Check if already processed this block
                if (block_y, block_x) in processed_blocks:
                    # Remove from uncovered set to avoid infinite loop
                    uncovered_coords.remove(tuple(map(float, coord)))
                    continue
                    
                # Mark as processed
                processed_blocks.add((block_y, block_x))
                
                # Process this block
                covered_coords = self._process_block_with_coords_and_track(
                    batch, source_view_id, block_y, block_x,
                    list(map(np.array, uncovered_coords)), ray_sample_map)
                
                # Only count as an extra read if coordinates were covered
                if covered_coords:
                    extra_reads += 1
                    
                # Update uncovered set
                for coord in covered_coords:
                    coord_tuple = tuple(map(float, coord))
                    if coord_tuple in uncovered_coords:
                        uncovered_coords.remove(coord_tuple)
                        
        return initial_reads, extra_reads
    
    def _collect_block_access_spatial_debug(self, batch, subblock_id, ray_y, ray_x, sample_start):
        """
        带调试功能的空间局部性策略：分析坐标分布和块覆盖情况
        
        参数:
            batch: 光线批处理组
            subblock_id: 子块ID
            ray_y, ray_x: 子块在光线网格中的坐标
            sample_start: 样本起始索引
            
        返回:
            (initial_reads, extra_reads): 初始读取次数和额外读取次数
        """
        # 清空之前的映射
        batch.block_access_map = {}
        
        initial_reads = 0  # 平均坐标读取计数
        extra_reads = 0    # 额外读取计数
        
        # 为每个源视图单独处理
        for source_view_id in batch.source_views:
            source_view = next((sv for sv in self.source_views if sv.id == source_view_id), None)
            if not source_view:
                continue
            
            # 收集此批次中所有有效的投影坐标（在图像范围内的）
            valid_coords = []
            ray_sample_map = {}  # 存储每个坐标对应的(ray_id, sample_id)
            
            for ray in batch.rays:
                for sample_id in batch.samples:
                    if sample_id >= len(ray.sample_points):
                        continue
                        
                    sample = ray.sample_points[sample_id]
                    if source_view_id not in sample.coordinates:
                        continue
                        
                    coord = sample.coordinates[source_view_id]
                    x, y = coord
                    
                    # 检查坐标是否在图像范围内
                    if not (0 <= x < source_view.width and 0 <= y < source_view.height):
                        continue
                        
                    valid_coords.append(coord)
                    coord_tuple = tuple(map(float, coord))  # 转换为可哈希类型
                    ray_sample_map[coord_tuple] = (ray.id, sample_id)
            
            # 如果没有有效坐标，继续下一个源视图
            if not valid_coords:
                continue
            
            # 创建坐标数组用于调试
            coord_array = np.array(valid_coords)
            
            # 计算平均坐标
            avg_coord = np.mean(coord_array, axis=0)
            
            # 确保平均坐标在图像范围内
            avg_x, avg_y = avg_coord
            avg_x = max(0, min(avg_x, source_view.width - 1))
            avg_y = max(0, min(avg_y, source_view.height - 1))
            
            # 获取平均坐标对应的块
            avg_block_x, avg_block_y = coordinate_to_block(
                [avg_x, avg_y], self.storage_config.block_size)
            
            # 标记所有已被覆盖的坐标
            uncovered_coords = set(map(tuple, map(lambda c: map(float, c), valid_coords)))  # 转为集合以方便查询
            uncovered_array = coord_array.copy()  # 用于可视化
            
            # 记录每次读取的信息
            read_infos = []
            
            # 处理平均块 - 初始读取
            block_key = (source_view_id, avg_block_y, avg_block_x)
            block_start_x = avg_block_x * self.storage_config.block_size[1]
            block_start_y = avg_block_y * self.storage_config.block_size[0]
            block_end_x = block_start_x + self.storage_config.block_size[1]
            block_end_y = block_start_y + self.storage_config.block_size[0]
            
            covered_coords = self._process_block_with_coords_and_track(
                batch, source_view_id, avg_block_y, avg_block_x, 
                list(map(np.array, uncovered_coords)), ray_sample_map)
            
            # 记录初始读取信息
            if covered_coords:
                initial_reads += 1
                read_infos.append({
                    "read_type": "initial",
                    "block": (avg_block_x, avg_block_y),
                    "block_bounds": (block_start_x, block_start_y, block_end_x, block_end_y),
                    "covered_coords": covered_coords,
                    "target_coord": avg_coord
                })
            
            # 更新未覆盖集合
            covered_set = set()
            for coord in covered_coords:
                coord_tuple = tuple(map(float, coord))
                if coord_tuple in uncovered_coords:
                    uncovered_coords.remove(coord_tuple)
                    covered_set.add(coord_tuple)
            
            # 可视化未覆盖坐标
            if covered_coords:
                # 从uncovered_array中移除已覆盖的坐标
                mask = np.ones(uncovered_array.shape[0], dtype=bool)
                for i, coord in enumerate(uncovered_array):
                    if tuple(map(float, coord)) in covered_set:
                        mask[i] = False
                uncovered_array = uncovered_array[mask]
            
            # 处理剩余未覆盖的坐标 - 额外读取
            processed_blocks = {(avg_block_y, avg_block_x)}  # 记录已处理过的块坐标
            
            extra_read_count = 0
            while uncovered_coords and extra_read_count < 100:  # 防止无限循环
                extra_read_count += 1
                
                # 选择一个未覆盖的坐标
                first_coord = next(iter(uncovered_coords))
                coord = np.array(first_coord)
                
                # 获取其对应的块
                block_x, block_y = coordinate_to_block(coord, self.storage_config.block_size)
                
                # 检查是否已经处理过这个块
                if (block_y, block_x) in processed_blocks:
                    # 在集合中移除此坐标，避免无限循环
                    uncovered_coords.remove(first_coord)
                    continue
                    
                # 标记为已处理
                processed_blocks.add((block_y, block_x))
                
                # 计算块边界
                block_start_x = block_x * self.storage_config.block_size[1]
                block_start_y = block_y * self.storage_config.block_size[0]
                block_end_x = block_start_x + self.storage_config.block_size[1]
                block_end_y = block_start_y + self.storage_config.block_size[0]
                
                # 处理这个块
                covered_coords = self._process_block_with_coords_and_track(
                    batch, source_view_id, block_y, block_x,
                    list(map(np.array, uncovered_coords)), ray_sample_map)
                
                # 记录额外读取信息
                if covered_coords:
                    extra_reads += 1
                    read_infos.append({
                        "read_type": "extra",
                        "block": (block_x, block_y),
                        "block_bounds": (block_start_x, block_start_y, block_end_x, block_end_y),
                        "covered_coords": covered_coords,
                        "target_coord": coord
                    })
                
                # 更新未覆盖集合和数组
                covered_set = set()
                for coord in covered_coords:
                    coord_tuple = tuple(map(float, coord))
                    if coord_tuple in uncovered_coords:
                        uncovered_coords.remove(coord_tuple)
                        covered_set.add(coord_tuple)
                
                # 更新未覆盖数组
                if covered_coords:
                    # 从uncovered_array中移除已覆盖的坐标
                    mask = np.ones(uncovered_array.shape[0], dtype=bool)
                    for i, coord in enumerate(uncovered_array):
                        if tuple(map(float, coord)) in covered_set:
                            mask[i] = False
                    uncovered_array = uncovered_array[mask]
            
            # 判断是否需要详细调试
            total_reads = initial_reads + extra_reads
            if total_reads >= self.debug_threshold:
                # 生成调试可视化
                self._generate_debug_visualization(
                    subblock_id, ray_y, ray_x, sample_start, source_view_id, 
                    coord_array, read_infos, source_view.width, source_view.height
                )
                
        return initial_reads, extra_reads
    
    def _process_block_with_coords_and_track(self, batch, source_view_id, block_y, block_x, 
                                      coords, ray_sample_map):
        """
        Process a block and check which coordinates it covers, updating block access mapping
        
        Parameters:
            batch: Ray batch
            source_view_id: Source view ID
            block_y, block_x: Block coordinates
            coords: List of coordinates to check
            ray_sample_map: Mapping from coordinates to (ray_id, sample_id)
        
        Returns:
            covered_coords: List of coordinates covered by this block
        """
        # Apply downsampling factor to calculate block boundaries
        downsample = self.storage_config.downsample_factor
        block_start_y = block_y * self.storage_config.block_size[0] 
        block_start_x = block_x * self.storage_config.block_size[1]
        block_end_y = block_start_y + self.storage_config.block_size[0]
        block_end_x = block_start_x + self.storage_config.block_size[1]
        
        # Check which coordinates fall within this block after downsampling
        covered_ray_samples = set()
        covered_coords = []
        
        for coord in coords:
            x, y = coord
            # Apply downsampling to the coordinates for comparison with block boundaries
            x_ds = x / downsample
            y_ds = y / downsample
            
            if (block_start_x <= x_ds < block_end_x and block_start_y <= y_ds < block_end_y):
                # This coordinate is covered by the current block
                coord_tuple = tuple(map(float, coord))
                if coord_tuple in ray_sample_map:
                    ray_sample = ray_sample_map[coord_tuple]
                    covered_ray_samples.add(ray_sample)
                    covered_coords.append(coord)
        
        # If any coordinates are covered, add block access
        if covered_ray_samples:
            block_key = (source_view_id, block_y, block_x)
            if block_key not in batch.block_access_map:
                batch.block_access_map[block_key] = set()
            
            batch.block_access_map[block_key].update(covered_ray_samples)
        
        return covered_coords
    
    def _generate_debug_visualization(self, subblock_id, ray_y, ray_x, sample_start, 
                                source_view_id, all_coords, read_infos, img_width, img_height):
        """Generate debug visualizations with English labels only"""
        prefix = f"subblock_{subblock_id}_ray_{ray_y}_{ray_x}_sample_{sample_start}_sv_{source_view_id}"
        downsample = self.storage_config.downsample_factor
        
        # Include both original and downsampled coordinate spaces
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original coordinate space
        ax1.scatter(all_coords[:, 0], all_coords[:, 1], alpha=0.5, s=10)
        
        # Calculate bounds
        if len(all_coords) > 0:
            min_x = np.min(all_coords[:, 0])
            max_x = np.max(all_coords[:, 0])
            min_y = np.min(all_coords[:, 1])
            max_y = np.max(all_coords[:, 1])
            
            # Add padding (10% of range)
            x_padding = max(1, (max_x - min_x) * 0.1)
            y_padding = max(1, (max_y - min_y) * 0.1)
            
            plot_min_x = max(0, min_x - x_padding)
            plot_max_x = min(img_width * downsample, max_x + x_padding)
            plot_min_y = max(0, min_y - y_padding)
            plot_max_y = min(img_height * downsample, max_y + y_padding)
        else:
            # Fallback if no coordinates
            plot_min_x, plot_max_x = 0, img_width * downsample
            plot_min_y, plot_max_y = 0, img_height * downsample
            
        ax1.set_xlim(plot_min_x, plot_max_x)
        ax1.set_ylim(plot_min_y, plot_max_y)
        ax1.invert_yaxis()  # Image coordinate system y-axis points down
        ax1.set_title(f"Original Coordinates - Subblock {subblock_id}\n"
                    f"(rays: {ray_y},{ray_x}, samples: {sample_start})")
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Downsampled coordinate space
        downsampled_coords = all_coords / downsample
        ax2.scatter(downsampled_coords[:, 0], downsampled_coords[:, 1], alpha=0.5, s=10)
        ax2.set_xlim(plot_min_x/downsample, plot_max_x/downsample)
        ax2.set_ylim(plot_min_y/downsample, plot_max_y/downsample)
        ax2.invert_yaxis()  # Image coordinate system y-axis points down
        ax2.set_title(f"Downsampled Coordinates (÷{downsample}) - Subblock {subblock_id}\n"
                    f"Source view: {source_view_id}, Total: {len(all_coords)}")
        ax2.set_xlabel("X Coordinate (downsampled)")
        ax2.set_ylabel("Y Coordinate (downsampled)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/{prefix}_all_coords.png")
        plt.close()
        
        # Coverage visualization with blocks shown in downsampled space
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Original space
        ax1.scatter(all_coords[:, 0], all_coords[:, 1], alpha=0.3, s=5, c='gray', label='All Coordinates')
        ax1.set_xlim(plot_min_x, plot_max_x)
        ax1.set_ylim(plot_min_y, plot_max_y)
        ax1.invert_yaxis()
        ax1.set_title(f"Original Space - Block Coverage - Subblock {subblock_id}")
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate") 
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Downsampled space
        ax2.scatter(downsampled_coords[:, 0], downsampled_coords[:, 1], alpha=0.3, s=5, c='gray', label='All Coordinates')
        ax2.set_xlim(plot_min_x/downsample, plot_max_x/downsample)
        ax2.set_ylim(plot_min_y/downsample, plot_max_y/downsample)
        ax2.invert_yaxis()
        ax2.set_title(f"Downsampled Space - Block Coverage - Subblock {subblock_id}")
        ax2.set_xlabel("X Coordinate (downsampled)")
        ax2.set_ylabel("Y Coordinate (downsampled)") 
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Color table
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # Mark first 10 reads in both spaces
        for i, info in enumerate(read_infos[:10]):
            color = colors[i % len(colors)]
            
            # Draw block boundary in downsampled space
            x_start, y_start, x_end, y_end = info["block_bounds"]
            
            # Original space - just plot covered points
            if len(info["covered_coords"]) > 0:
                covered = np.array(info["covered_coords"])
                ax1.scatter(covered[:, 0], covered[:, 1], c=color, s=20, marker='o', alpha=0.7)
                
            # Draw target coordinate
            target_x, target_y = info["target_coord"]
            ax1.scatter([target_x], [target_y], c=color, s=100, marker='x')
            
            # Downsampled space - draw block and points
            rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
                                fill=False, edgecolor=color, linewidth=2, 
                                label=f"Read {i+1} ({info['read_type']})")
            ax2.add_patch(rect)
            
            # Draw target coordinate in downsampled space
            ax2.scatter([target_x/downsample], [target_y/downsample], c=color, s=100, marker='x')
            
            # Draw covered coordinates in downsampled space
            if len(info["covered_coords"]) > 0:
                covered = np.array(info["covered_coords"]) / downsample
                ax2.scatter(covered[:, 0], covered[:, 1], c=color, s=20, marker='o', alpha=0.7)
        
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/{prefix}_coverage.png")
        plt.close()
        
        # Output read statistics
        read_stats_file = f"{self.debug_dir}/{prefix}_read_stats.txt"
        with open(read_stats_file, 'w') as f:
            f.write(f"Subblock {subblock_id} Read Statistics (rays: {ray_y},{ray_x}, samples: {sample_start}, source_view: {source_view_id})\n")
            f.write(f"Total Coordinates: {len(all_coords)}\n")
            f.write(f"Coordinate Range (Original): X[{plot_min_x:.1f}-{plot_max_x:.1f}], Y[{plot_min_y:.1f}-{plot_max_y:.1f}]\n")
            f.write(f"Coordinate Range (Downsampled): X[{plot_min_x/downsample:.1f}-{plot_max_x/downsample:.1f}], Y[{plot_min_y/downsample:.1f}-{plot_max_y/downsample:.1f}]\n")
            f.write(f"Downsample Factor: {downsample}x\n")
            f.write(f"Block Size: {self.storage_config.block_size[0]}×{self.storage_config.block_size[1]}\n")
            f.write(f"Total Reads: {len(read_infos)}\n")
            f.write(f"Initial Reads: {sum(1 for info in read_infos if info['read_type'] == 'initial')}\n")
            f.write(f"Extra Reads: {sum(1 for info in read_infos if info['read_type'] == 'extra')}\n\n")
            
            f.write("Detailed Read Statistics:\n")
            for i, info in enumerate(read_infos):
                covered_count = len(info["covered_coords"])
                block_x, block_y = info["block"]
                f.write(f"Read {i+1} ({info['read_type']}):\n")
                f.write(f"  Block Coordinates: ({block_x}, {block_y})\n")
                f.write(f"  Block Bounds (downsampled): {info['block_bounds']}\n")
                f.write(f"  Covered Coordinates: {covered_count}\n")
                if i < 10:  # Only output detailed coordinates for first 10 reads
                    f.write(f"  Target Coordinate (original): {info['target_coord']}\n")
                    f.write(f"  Target Coordinate (downsampled): ({info['target_coord'][0]/downsample:.1f}, {info['target_coord'][1]/downsample:.1f})\n")
                    if covered_count > 0:
                        f.write(f"  First few covered coordinates (original): {info['covered_coords'][:5]}\n")
                        if covered_count > 5:
                            f.write(f"  ... and {covered_count-5} more\n")
                    f.write("\n")
                
        # Output debug info
        print(f"\n[DEBUG] Generated visualizations for subblock {subblock_id}, source view {source_view_id}")
        print(f"  Total reads: {len(read_infos)}, Initial: {sum(1 for info in read_infos if info['read_type'] == 'initial')}, "
            f"Extra: {sum(1 for info in read_infos if info['read_type'] == 'extra')}")
        print(f"  Coordinate Range (Original): X[{plot_min_x:.1f}-{plot_max_x:.1f}], Y[{plot_min_y:.1f}-{plot_max_y:.1f}]")
        print(f"  Coordinate Range (Downsampled): X[{plot_min_x/downsample:.1f}-{plot_max_x/downsample:.1f}], Y[{plot_min_y/downsample:.1f}-{plot_max_y/downsample:.1f}]")
    
    def _execute_batch_memory_access(self, batch: RayBatch) -> List[MemoryAccessInfo]:
        """
        执行批处理内存访问
        
        参数:
            batch: 光线批处理组
        
        返回:
            access_infos: 内存访问信息列表
        """
        access_infos = []
        
        # 为每个源视图按块读取内存
        for source_view_id in batch.source_views:
            # 获取该源视图的所有块
            source_view_blocks = [(sv_id, by, bx) for (sv_id, by, bx) in batch.block_access_map.keys() 
                                 if sv_id == source_view_id]
            
            # 读取每个块
            for block_key in source_view_blocks:
                _, block_y, block_x = block_key
                
                try:
                    # 读取内存块
                    block_data, latency = self.ddr_memory.read_block(source_view_id, block_y, block_x)
                    
                    if block_data is not None:
                        # 创建块对象
                        block = Block(
                            source_view_id=source_view_id,
                            block_x=block_x,
                            block_y=block_y,
                            data=block_data
                        )
                        
                        # 记录访问信息
                        access_info = MemoryAccessInfo(
                            block=block,
                            ray_sample_tuples=batch.block_access_map[block_key],
                            latency_cycles=latency
                        )
                        
                        access_infos.append(access_info)
                except ValueError as e:
                    # 块不在内存映射中，忽略这个块
                    print(f"警告: {e}")
                    continue
        
        return access_infos