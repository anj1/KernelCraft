import numpy as np
from numba import njit, int64, float64
from numba.types import Tuple

@njit
def get_node_level(morton_code):
    """Get the level of a node from its morton code."""
    # Assuming the level is stored in the last 4 bits
    return morton_code & 0xF

@njit
def get_node_coords(morton_code):
    """Extract x,y,z coordinates from morton code."""
    x = y = z = 0
    for i in range(20):  # Assuming 20 bits per dimension
        mask = 1 << i
        x |= (morton_code & (mask << 2)) >> (2 * i)
        y |= (morton_code & (mask << 1)) >> (2 * i - 1)
        z |= (morton_code & mask) >> (2 * i - 2)
    return x, y, z

@njit
def compute_morton_code(x: int64, y: int64, z: int64, level: int64) -> int64:
    """Compute morton code for given coordinates and level."""
    code = 0
    for i in range(20):  # Support up to 20 bits per dimension
        mask = 1 << i
        code |= (x & mask) << (2 * i)
        code |= (y & mask) << (2 * i + 1)
        code |= (z & mask) << (2 * i + 2)
    return (code << 4) | level

@njit
def find_neighbors(node_coords: np.ndarray, node_indices: np.ndarray, 
                  morton_codes: np.ndarray, max_neighbors: int = 26) -> Tuple:
    """
    Find neighbors for each node in the octree.
    
    Parameters:
    -----------
    node_coords : np.ndarray (N, 3)
        Coordinates of query points
    node_indices : np.ndarray (N,)
        Indices of octree nodes containing query points
    morton_codes : np.ndarray (M,)
        Morton codes for all nodes in the octree
    max_neighbors : int
        Maximum number of neighbors per node (default 26)
        
    Returns:
    --------
    n_neighbors : np.ndarray (N,)
        Number of neighbors for each node
    neighbor_offsets : np.ndarray (N,)
        Offset into neighbor_list for each node
    neighbor_list : np.ndarray (K,)
        Flattened list of neighbor indices
    """
    n_nodes = len(node_indices)
    n_neighbors = np.zeros(n_nodes, dtype=np.int64)
    neighbor_offsets = np.zeros(n_nodes, dtype=np.int64)
    neighbor_list = np.zeros(n_nodes * max_neighbors, dtype=np.int64)
    
    # Calculate offsets
    offset = 0
    for i in range(n_nodes):
        node_idx = node_indices[i]
        node_code = morton_codes[node_idx]
        x, y, z = get_node_coords(node_code)
        level = get_node_level(node_code)
        
        # Store offset
        neighbor_offsets[i] = offset
        
        # Check potential neighbors (26-connectivity)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                        
                    # Compute potential neighbor coordinates
                    nx = x + dx
                    ny = y + dy
                    nz = z + dz
                    
                    # Compute morton code for neighbor
                    neighbor_code = compute_morton_code(nx, ny, nz, level)
                    
                    # Search for neighbor in morton_codes
                    for j in range(len(morton_codes)):
                        if morton_codes[j] == neighbor_code:
                            neighbor_list[offset] = j
                            n_neighbors[i] += 1
                            offset += 1
                            break
    
    # Trim neighbor_list to actual size
    neighbor_list = neighbor_list[:offset]
    
    return n_neighbors, neighbor_offsets, neighbor_list