import sys
import os
import shutil
from run_simulation import run_nerf_accelerator_simulation

def debug_spatial_locality(threshold=10, start_after=5):
    """
    Run debug version of spatial locality strategy
    
    Parameters:
        threshold: Read count threshold to trigger detailed debugging
        start_after: Number of subblocks to process before starting debug
    """
    # Delete existing debug_output directory and recreate it
    if os.path.exists("debug_output"):
        print("Removing existing debug_output directory...")
        shutil.rmtree("debug_output")
    
    os.makedirs("debug_output", exist_ok=True)
    print("Created fresh debug_output directory")
    
    print(f"Starting spatial locality strategy debugging - Threshold: {threshold}, Start after: {start_after}")
    
    # Run simulation
    run_nerf_accelerator_simulation(
        memory_access_strategy="spatial_locality",
        enable_debug=True,
        debug_threshold=threshold,
        debug_start_after=start_after
    )
    
    print("\nDebugging completed. Please check output files in debug_output/ directory")
    print("Generated visualizations include:")
    print("1. *_all_coords.png - Distribution of all coordinates in the subblock")
    print("2. *_coverage.png - Visualization of coordinates covered by each read")
    print("3. *_read_stats.txt - Detailed read statistics")

if __name__ == "__main__":
    # Default threshold
    threshold = 10
    start_after = 5
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            threshold = int(sys.argv[1])
        except:
            pass
    
    if len(sys.argv) > 2:
        try:
            start_after = int(sys.argv[2])
        except:
            pass
    
    debug_spatial_locality(threshold, start_after)