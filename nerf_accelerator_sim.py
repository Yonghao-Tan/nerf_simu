import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
import time

from ddr3_memory_model import DDR3Config, SourceViewStorageConfig, DDR3Memory
from memory_scheduler import ReadScheduler, MemoryAccessStrategy, SimulationResult
from camera_utils import Camera

@dataclass
class SamplePoint:
    """光线上的采样点"""
    id: int                                        # 采样点ID
    position: np.ndarray                           # 3D空间位置
    coordinates: Dict[int, np.ndarray] = field(default_factory=dict)  # 每个源视图的投影坐标 {source_view_id: [x, y]}

@dataclass
class Ray:
    """包含多个采样点的光线"""
    id: int                                        # 光线ID
    origin: np.ndarray                             # 光线起点
    direction: np.ndarray                          # 光线方向
    grid_y: int = 0                                # 网格中的Y位置
    grid_x: int = 0                                # 网格中的X位置
    sample_points: List[SamplePoint] = field(default_factory=list)  # 光线上的所有采样点

@dataclass
class SourceView:
    """源视图（参考图像）"""
    id: int                                        # 源视图ID
    height: int                                    # 高度
    width: int                                     # 宽度
    feature_dim: int                               # 特征维度
    camera_params: Dict                            # 相机参数（位置、旋转等）

class NeRFAcceleratorSimulator:
    """NeRF加速器仿真器"""
    def __init__(
        self, 
        rays: List[Ray],
        source_views: List[SourceView],
        ddr_config: DDR3Config,
        storage_config: SourceViewStorageConfig,
        ray_grid_dims: Tuple[int, int] = None,  # (height, width)
        memory_access_strategy: str = "baseline"  # 内存访问策略
    ):
        self.rays = rays
        self.source_views = source_views
        self.ray_grid_dims = ray_grid_dims
        
        # 创建DDR3内存
        self.ddr_memory = DDR3Memory(ddr_config, storage_config)
        
        # 设置内存访问策略
        try:
            self.strategy = MemoryAccessStrategy(memory_access_strategy)
        except ValueError:
            print(f"警告: 未知的内存访问策略 '{memory_access_strategy}'，使用基线策略")
            self.strategy = MemoryAccessStrategy.BASELINE
        
        # 创建读取调度器
        self.read_scheduler = ReadScheduler(rays, source_views, self.ddr_memory, storage_config, self.strategy)
        
        # 添加网格尺寸信息
        if ray_grid_dims:
            self.read_scheduler.ray_grid_height = ray_grid_dims[0]
            self.read_scheduler.ray_grid_width = ray_grid_dims[1]
        
    def run_simulation(self, ray_batch_size=(16, 16), sub_batch_size=(8, 8), samples_per_batch=8) -> Dict:
        """
        运行完整仿真
        
        参数:
            ray_batch_size: (height, width) 主块大小
            sub_batch_size: (height, width) 子块大小（用于spatial策略）
            samples_per_batch: 每批处理的采样点数
        
        返回:
            结果统计
        """
        print(f"开始仿真 - 策略: {self.strategy.value}")
        print(f"参数: ray_batch_size={ray_batch_size}, sub_batch_size={sub_batch_size}, samples_per_batch={samples_per_batch}")
        
        # 运行仿真
        result = self.read_scheduler.process_all_rays(ray_batch_size, sub_batch_size, samples_per_batch)
        
        # 获取DDR统计
        ddr_stats = self.ddr_memory.get_statistics()
        
        # 构建结果字典
        return {
            "total_memory_accesses": result.total_memory_accesses,
            "total_memory_cycles": result.total_memory_cycles,
            "total_bytes_read": result.total_bytes_read,
            "total_mb_read": result.total_bytes_read / (1024*1024),
            "row_hit_rate": result.row_hit_rate * 100,  # 转为百分比
            "blocks_processed": result.blocks_processed,
            "ddr_stats": ddr_stats,
            "simulation_runtime_s": result.simulation_runtime_s,
        }
        
    def load_3d_data(self, sample_locations_file=None, source_view_poses_file=None, target_view_pose_file=None):
        """
        Load 3D sample locations and camera poses for epipolar strategy
        
        Args:
            sample_locations_file: Path to sample_locations_valid.npy
            source_view_poses_file: Path to source_view_poses.npy
            target_view_pose_file: Path to target_view_pose.npy
        """
        try:
            # Load 3D sample locations
            if sample_locations_file:
                print(f"Loading 3D sample locations from {sample_locations_file}")
                self.sample_locations_3d = np.load(sample_locations_file)
                print(f"Sample locations loaded, shape: {self.sample_locations_3d.shape}")
                
                # Update sample points with 3D positions
                num_rays = len(self.rays)
                for ray_idx, ray in enumerate(self.rays):
                    if ray_idx < self.sample_locations_3d.shape[0]:
                        for sample_idx, sample in enumerate(ray.sample_points):
                            if sample_idx < self.sample_locations_3d.shape[1]:
                                sample.position = self.sample_locations_3d[ray_idx, sample_idx]
            
            # Load source view poses
            if source_view_poses_file:
                print(f"Loading source view poses from {source_view_poses_file}")
                source_poses = np.load(source_view_poses_file)
                print(f"Source view poses loaded, shape: {source_poses.shape}")
                
                # Create camera objects for each source view
                self.source_cameras = []
                for i, source_view in enumerate(self.source_views):
                    if i < source_poses.shape[1]:
                        camera = Camera(source_poses[0, i])
                        self.source_cameras.append(camera)
                        
                        # Update source view dimensions if needed
                        source_view.width = camera.img_width
                        source_view.height = camera.img_height
            
            # Load target view pose
            if target_view_pose_file:
                print(f"Loading target view pose from {target_view_pose_file}")
                target_pose = np.load(target_view_pose_file)
                print(f"Target view pose loaded, shape: {target_pose.shape}")
                
                # Create camera object for target view
                self.target_camera = Camera(target_pose[0])
            
            # Pass camera information to the read scheduler
            if hasattr(self, 'source_cameras') and hasattr(self, 'target_camera'):
                self.read_scheduler.set_camera_info(self.source_cameras, self.target_camera)
            
            return True
        
        except Exception as e:
            print(f"Error loading 3D data: {e}")
            import traceback
            traceback.print_exc()
            return False