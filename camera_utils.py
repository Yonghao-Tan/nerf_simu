import numpy as np
from typing import Tuple, List, Dict, Optional, Any

class Camera:
    """Camera model with intrinsics and extrinsics"""
    def __init__(self, pose_data):
        """
        Initialize camera from pose data array
        Format: [img_width, img_height, intrinsics(16), extrinsics(16)]
        """
        self.img_width = int(pose_data[0])
        self.img_height = int(pose_data[1])
        
        # Reshape intrinsics to 4x4 matrix
        self.intrinsic = pose_data[2:18].reshape(4, 4)
        
        # Reshape extrinsics to 4x4 matrix
        self.extrinsic = pose_data[18:34].reshape(4, 4)
        
        # Calculate view matrix (world to camera)
        self.view_matrix = self.extrinsic
        
        # Calculate camera position in world space
        self.position = np.linalg.inv(self.view_matrix)[:3, 3]
        
        # Extract focal length and principal point
        self.fx = self.intrinsic[0, 0]
        self.fy = self.intrinsic[1, 1]
        self.cx = self.intrinsic[0, 2]
        self.cy = self.intrinsic[1, 2]
    
    def world_to_camera(self, points):
        """
        Transform points from world to camera coordinates
        
        Args:
            points: [N, 3] array of points in world space
            
        Returns:
            [N, 3] array of points in camera space
        """
        N = points.shape[0]
        # Convert to homogeneous coordinates
        points_h = np.hstack((points, np.ones((N, 1))))
        # Transform by view matrix
        cam_points = points_h @ self.view_matrix.T
        # Return 3D points
        return cam_points[:, :3]
    
    def project_points(self, world_points):
        """
        Project 3D world points onto image plane
        
        Args:
            world_points: [N, 3] array of points in world space
            
        Returns:
            pixels: [N, 2] array of pixel coordinates
            depths: [N] array of depths in camera space
        """
        # Transform to camera space
        cam_points = self.world_to_camera(world_points)
        
        # Project to image plane
        z = cam_points[:, 2]
        x = cam_points[:, 0] / z
        y = cam_points[:, 1] / z
        
        # Apply intrinsic parameters
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        
        return np.column_stack((u, v)), z
    
    def get_frustum_corners(self, block_x, block_y, block_w, block_h, near, far):
        """
        Get frustum corners for the specified block and depth range
        
        Args:
            block_x, block_y: Block top-left corner
            block_w, block_h: Block dimensions
            near, far: Near and far depth planes
            
        Returns:
            corners: 8 corners of the frustum in world space
        """
        # Define block corners in image space
        block_corners = np.array([
            [block_x, block_y],                     # Top-left
            [block_x + block_w, block_y],           # Top-right
            [block_x, block_y + block_h],           # Bottom-left
            [block_x + block_w, block_y + block_h]  # Bottom-right
        ])
        
        # Unproject to get ray directions
        corners_near = []
        corners_far = []
        
        for u, v in block_corners:
            # Convert to normalized device coordinates
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
            
            # Near plane point in camera space
            near_point = near * np.array([x, y, 1.0])
            
            # Far plane point in camera space
            far_point = far * np.array([x, y, 1.0])
            
            # Convert to world space
            inv_view = np.linalg.inv(self.view_matrix)
            near_point_world = (inv_view @ np.append(near_point, 1.0))[:3]
            far_point_world = (inv_view @ np.append(far_point, 1.0))[:3]
            
            corners_near.append(near_point_world)
            corners_far.append(far_point_world)
        
        return np.array(corners_near + corners_far)
    
    def ray_direction(self, pixel_x, pixel_y):
        """
        Get ray direction in world space for a given pixel
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            
        Returns:
            direction: Normalized ray direction in world space
        """
        # Convert to normalized device coordinates
        x = (pixel_x - self.cx) / self.fx
        y = (pixel_y - self.cy) / self.fy
        
        # Direction in camera space
        cam_dir = np.array([x, y, 1.0])
        
        # Convert to world space
        inv_rot = np.linalg.inv(self.view_matrix)[:3, :3]
        world_dir = inv_rot @ cam_dir
        
        # Normalize
        return world_dir / np.linalg.norm(world_dir)

def point_in_frustum(point, frustum_corners):
    """
    Test if a 3D point is inside a frustum
    
    Args:
        point: [3] array, 3D point to test
        frustum_corners: [8, 3] array, corners of the frustum
        
    Returns:
        bool: True if point is inside the frustum
    """
    # Simplified test - check if point is inside the convex hull
    # This is a basic implementation; a more efficient approach would use 
    # plane tests for the 6 faces of the frustum
    try:
        from scipy.spatial import ConvexHull, Delaunay
        hull = Delaunay(frustum_corners)
        return hull.find_simplex(point) >= 0
    except ImportError:
        # Fallback if scipy is not available
        # This is very approximate - proper frustum check would test against all 6 planes
        # Just check if point is within bounding box of frustum corners
        mins = np.min(frustum_corners, axis=0)
        maxs = np.max(frustum_corners, axis=0)
        return np.all(point >= mins) and np.all(point <= maxs)