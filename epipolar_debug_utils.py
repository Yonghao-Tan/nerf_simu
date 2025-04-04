"""
Epipolar Debug Utilities
Created for Yonghao-Tan - 2025-04-04
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def create_debug_directory():
    """Create debug output directory"""
    debug_dir = "epipolar_debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    return debug_dir

def draw_frustum(ax, corners, color='blue', alpha=0.3, label=None):
    """
    Draw a frustum on 3D axis with correct edge connections
    
    Args:
        ax: matplotlib 3D axis
        corners: Coordinates of 8 corners [8, 3]
        color: Edge color
        alpha: Face transparency
        label: Legend label
    """
    # This function assumes corners are in the following order:
    # 0-3: near plane in counter-clockwise order (bottom-left, bottom-right, top-right, top-left)
    # 4-7: far plane in counter-clockwise order (bottom-left, bottom-right, top-right, top-left)
    
    # For debugging purposes - uncomment to see corner ordering
    # for i, corner in enumerate(corners):
    #     ax.text(corner[0], corner[1], corner[2], str(i), color='black', fontsize=10)
    
    # Draw each face separately using the appropriate corner indices
    # Near plane (0,1,2,3)
    ax.plot([corners[0][0], corners[1][0]], [corners[0][1], corners[1][1]], [corners[0][2], corners[1][2]], color=color, linestyle='-', linewidth=1, label=label)
    ax.plot([corners[1][0], corners[2][0]], [corners[1][1], corners[2][1]], [corners[1][2], corners[2][2]], color=color, linestyle='-', linewidth=1)
    ax.plot([corners[2][0], corners[3][0]], [corners[2][1], corners[3][1]], [corners[2][2], corners[3][2]], color=color, linestyle='-', linewidth=1)
    ax.plot([corners[3][0], corners[0][0]], [corners[3][1], corners[0][1]], [corners[3][2], corners[0][2]], color=color, linestyle='-', linewidth=1)
    
    # Far plane (4,5,6,7)
    ax.plot([corners[4][0], corners[5][0]], [corners[4][1], corners[5][1]], [corners[4][2], corners[5][2]], color=color, linestyle='-', linewidth=1)
    ax.plot([corners[5][0], corners[6][0]], [corners[5][1], corners[6][1]], [corners[5][2], corners[6][2]], color=color, linestyle='-', linewidth=1)
    ax.plot([corners[6][0], corners[7][0]], [corners[6][1], corners[7][1]], [corners[6][2], corners[7][2]], color=color, linestyle='-', linewidth=1)
    ax.plot([corners[7][0], corners[4][0]], [corners[7][1], corners[4][1]], [corners[7][2], corners[4][2]], color=color, linestyle='-', linewidth=1)
    
    # Connecting edges
    ax.plot([corners[0][0], corners[4][0]], [corners[0][1], corners[4][1]], [corners[0][2], corners[4][2]], color=color, linestyle='-', linewidth=1)
    ax.plot([corners[1][0], corners[5][0]], [corners[1][1], corners[5][1]], [corners[1][2], corners[5][2]], color=color, linestyle='-', linewidth=1)
    ax.plot([corners[2][0], corners[6][0]], [corners[2][1], corners[6][1]], [corners[2][2], corners[6][2]], color=color, linestyle='-', linewidth=1)
    ax.plot([corners[3][0], corners[7][0]], [corners[3][1], corners[7][1]], [corners[3][2], corners[7][2]], color=color, linestyle='-', linewidth=1)
    
    # # Create faces - 6 faces (near, far, 4 sides)
    # faces = [
    #     [corners[0], corners[1], corners[2], corners[3]],  # near plane
    #     [corners[4], corners[5], corners[6], corners[7]],  # far plane
    #     [corners[0], corners[1], corners[5], corners[4]],  # side 1
    #     [corners[1], corners[2], corners[6], corners[5]],  # side 2
    #     [corners[2], corners[3], corners[7], corners[6]],  # side 3
    #     [corners[3], corners[0], corners[4], corners[7]]   # side 4
    # ]
    
    # # Draw semi-transparent faces
    # for face in faces:
    #     x = [p[0] for p in face]
    #     y = [p[1] for p in face]
    #     z = [p[2] for p in face]
        
    #     # Add first point to close polygon
    #     x.append(face[0][0])
    #     y.append(face[0][1])
    #     z.append(face[0][2])
        
    #     ax.plot_trisurf(x, y, z, color=color, alpha=alpha)

def visualize_frustum_and_points(frustum_corners, points_in_frustum, points_outside=None, 
                               camera_position=None, layer_info=None, output_file=None):
    """
    Visualize frustum and points with high resolution
    
    Args:
        frustum_corners: 8 corners of the frustum [8, 3]
        points_in_frustum: Points inside the frustum [N, 3]
        points_outside: Points outside the frustum [M, 3] (optional)
        camera_position: Camera position [3,] (optional)
        layer_info: String with depth layer info (optional)
        output_file: Output file path (optional)
    """
    # Create figure with higher resolution
    fig = plt.figure(figsize=(16, 14), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw frustum with corrected edges
    draw_frustum(ax, frustum_corners, color='blue', alpha=0.15, label="Frustum")
    
    # Draw points inside frustum - same size as outside points
    if len(points_in_frustum) > 0:
        ax.scatter(points_in_frustum[:, 0], points_in_frustum[:, 1], points_in_frustum[:, 2], 
                  c='green', marker='.', s=10, alpha=0.7, label="Points Inside Frustum")
    
    # Draw points outside frustum (if provided)
    if points_outside is not None and len(points_outside) > 0:
        if len(points_outside) > 500:  # If too many outside points, sample some
            indices = np.random.choice(len(points_outside), size=500, replace=False)
            sampled_points = points_outside[indices]
            ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                      c='red', marker='.', s=10, alpha=0.3, label="Points Outside (Sampled)")
        else:
            ax.scatter(points_outside[:, 0], points_outside[:, 1], points_outside[:, 2], 
                      c='red', marker='.', s=10, alpha=0.3, label="Points Outside Frustum")
    
    # Draw camera position (if provided)
    if camera_position is not None:
        ax.scatter([camera_position[0]], [camera_position[1]], [camera_position[2]], 
                  c='purple', marker='*', s=100, label="Camera Position")
    
    # Set title and legend
    title = "Frustum and Sample Points Visualization"
    if layer_info:
        title += f" - {layer_info}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    
    # Adjust view angle for better 3D effect
    ax.view_init(elev=25, azim=30)
    
    # Set axis labels with larger font
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # Add grid
    ax.grid(True)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save with high resolution or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_summary_visualization(debug_stats, frustum_coords, output_file):
    """
    Create summary visualization showing multiple frustums and their point counts
    
    Args:
        debug_stats: List of debug statistics
        frustum_coords: List of frustum coordinates [(fx, fy), ...]
        output_file: Output file path
    """
    # Create grid to display frustum positions and point counts
    # Determine grid size
    fx_values = [coord[0] for coord in frustum_coords]
    fy_values = [coord[1] for coord in frustum_coords]
    
    max_fx = max(fx_values) if fx_values else 0
    max_fy = max(fy_values) if fy_values else 0
    
    grid = np.zeros((max_fy + 1, max_fx + 1))
    
    # Fill grid
    for stats, (fx, fy) in zip(debug_stats, frustum_coords):
        grid[fy, fx] = stats['points_in_frustum']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap='viridis')
    
    # Add color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Points in Frustum')
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, max_fx + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, max_fy + 0.5, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    # Annotate each cell with point count
    for fy in range(max_fy + 1):
        for fx in range(max_fx + 1):
            value = int(grid[fy, fx])
            if value > 0:
                ax.text(fx, fy, str(value), ha="center", va="center", color="w" if value > grid.max()/2 else "black")
    
    # Set axis labels and title
    ax.set_xlabel('Frustum X Coordinate')
    ax.set_ylabel('Frustum Y Coordinate')
    ax.set_title('Distribution of Sample Points in Frustums')
    
    # Save image
    plt.savefig(output_file, dpi=150)
    plt.close()