import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

@dataclass
class DDR3Config:
    """DDR3 memory configuration"""
    num_banks: int = 8                # Number of DDR3 banks
    num_rows: int = 65536             # Rows per bank
    num_columns: int = 1024           # Columns per bank
    device_width: int = 64            # 8×8-bit DDR3 chip width
    burst_length: int = 8             # Burst length
    tRCD: int = 13                    # Row activation to column access delay (ns)
    tRP: int = 13                     # Precharge delay (ns)
    tCL: int = 13                     # Column access delay (ns)
    clock_period: float = 1.25        # DDR3-1600 clock period (ns)
    
    # Calculate delay for reading one burst from DRAM (clock cycles)
    def get_burst_read_latency(self) -> int:
        """Calculate delay for reading one burst (clock cycles)"""
        # tRCD + tCL + BL/2
        return int(self.tRCD / self.clock_period) + int(self.tCL / self.clock_period) + self.burst_length//2

@dataclass
class SourceViewStorageConfig:
    """Source view storage configuration"""
    block_size: Tuple[int, int] = (4, 4)  # Block size (height, width) - changed from 16×16 to 4×4
    feature_dim: int = 32                 # Feature dimension
    data_width: int = 8                   # Data width per element (bit)
    downsample_factor: int = 4            # Downsample factor for coordinates
    
    def get_block_bytes(self) -> int:
        """Get bytes per block"""
        return self.block_size[0] * self.block_size[1] * self.feature_dim * (self.data_width // 8)

class DDR3Memory:
    """DDR3 memory model"""
    def __init__(self, ddr_config: DDR3Config, storage_config: SourceViewStorageConfig):
        self.ddr_config = ddr_config
        self.storage_config = storage_config
        
        # Track open row in each bank
        self.open_rows = [-1] * ddr_config.num_banks
        
        # Memory access statistics
        self.total_reads = 0
        self.total_row_activations = 0
        self.total_precharges = 0
        self.total_bursts = 0
        self.total_read_latency_cycles = 0
        self.total_bytes_read = 0
        
        # Mapping information
        self.source_view_to_bank = {}  # {source_view_id: bank_id}
        self.block_to_row = {}  # {(source_view_id, block_y, block_x): row}
        
        # Simulated memory data (for simulation only)
        self.memory_data = {}  # {(bank, row, block_in_row): data_block}
        
        # Calculate how many blocks can fit in one row
        block_bytes = storage_config.get_block_bytes() 
        self.blocks_per_row = max(1, (ddr_config.num_columns * (ddr_config.device_width // 8)) // block_bytes)
        print(f"Memory configuration: {self.blocks_per_row} blocks (4×4) per DRAM row")
    
    def initialize_mapping(self, source_views: List, block_dimensions: Tuple[int, int]):
        """Initialize source view to memory mapping"""
        num_blocks_x, num_blocks_y = block_dimensions
        
        for i, source_view in enumerate(source_views):
            # Determine which bank stores this source view
            bank_id = i % self.ddr_config.num_banks
            self.source_view_to_bank[source_view.id] = bank_id
            
            # Calculate row number for each block, with multiple blocks per row
            for block_y in range(num_blocks_y):
                for block_x in range(num_blocks_x):
                    # Calculate block index within this source view
                    block_idx = block_y * num_blocks_x + block_x
                    
                    # Calculate row and position within row
                    row = (i // self.ddr_config.num_banks) * ((num_blocks_y * num_blocks_x + self.blocks_per_row - 1) // self.blocks_per_row)
                    row += block_idx // self.blocks_per_row
                    pos_in_row = block_idx % self.blocks_per_row
                    
                    # Store mapping
                    self.block_to_row[(source_view.id, block_y, block_x)] = (row, pos_in_row)
                    
                    # Create simulated data (not needed in real system)
                    block_data = np.random.randint(0, 256, 
                                                 (self.storage_config.block_size[0], 
                                                  self.storage_config.block_size[1], 
                                                  self.storage_config.feature_dim), 
                                                 dtype=np.uint8)
                    self.memory_data[(bank_id, row, pos_in_row)] = block_data
    
    def read_block(self, source_view_id: int, block_y: int, block_x: int) -> Tuple[np.ndarray, int]:
        """
        Read specified block from a source view
        
        Returns:
            (block_data, latency_cycles)
        """
        # Get bank_id and row
        if source_view_id not in self.source_view_to_bank:
            raise ValueError(f"Source view ID {source_view_id} not found")
            
        bank_id = self.source_view_to_bank[source_view_id]
        row_info = self.block_to_row.get((source_view_id, block_y, block_x))
        
        if row_info is None:
            raise ValueError(f"Block ({source_view_id}, {block_y}, {block_x}) not found in memory mapping")
        
        row, pos_in_row = row_info
        
        # Calculate read latency
        latency_cycles = 0
        
        # If need to open a new row
        if self.open_rows[bank_id] != row:
            # If a row is already open, precharge first
            if self.open_rows[bank_id] != -1:
                latency_cycles += int(self.ddr_config.tRP / self.ddr_config.clock_period)
                self.total_precharges += 1
            
            # Activate new row
            latency_cycles += int(self.ddr_config.tRCD / self.ddr_config.clock_period)
            self.open_rows[bank_id] = row
            self.total_row_activations += 1
        
        # Calculate burst reads needed
        block_bytes = self.storage_config.get_block_bytes()
        bytes_per_burst = self.ddr_config.device_width * self.ddr_config.burst_length // 8
        num_bursts = (block_bytes + bytes_per_burst - 1) // bytes_per_burst  # Ceiling division
        
        # Calculate column access latency
        burst_latency = int(self.ddr_config.tCL / self.ddr_config.clock_period) + self.ddr_config.burst_length//2
        latency_cycles += num_bursts * burst_latency
        
        # Update statistics
        self.total_reads += 1
        self.total_bursts += num_bursts
        self.total_read_latency_cycles += latency_cycles
        self.total_bytes_read += block_bytes
        
        # Return data and latency
        return self.memory_data.get((bank_id, row, pos_in_row), None), latency_cycles

    def get_statistics(self) -> Dict:
        """Get memory access statistics"""
        return {
            "total_reads": self.total_reads,
            "total_row_activations": self.total_row_activations,
            "total_precharges": self.total_precharges,
            "total_bursts": self.total_bursts,
            "total_read_latency_cycles": self.total_read_latency_cycles,
            "average_latency_per_read": self.total_read_latency_cycles / max(1, self.total_reads),
            "row_hit_rate": 1.0 - (self.total_row_activations / max(1, self.total_reads)),
            "total_bytes_read": self.total_bytes_read,
            "total_mb_read": self.total_bytes_read / (1024*1024),
            "blocks_per_row": self.blocks_per_row
        }

def coordinate_to_block(coord: np.ndarray, block_size: Tuple[int, int], downsample_factor: int = 4) -> Tuple[int, int]:
    """
    Convert floating point coordinate to block coordinates, applying downsampling
    
    Parameters:
        coord: [x, y] coordinate
        block_size: (height, width) block size
        downsample_factor: coordinate downsampling factor
        
    Returns:
        (block_x, block_y)
    """
    x, y = coord
    # Apply downsampling and convert to integer coordinates
    x = int(x / downsample_factor)
    y = int(y / downsample_factor)
    block_x = x // block_size[1]
    block_y = y // block_size[0]
    return block_x, block_y