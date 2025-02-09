import numpy as np
from numba import njit
import time

from tasks.octree.reference import get_node_level, get_node_coords, compute_morton_code, find_neighbors

def create_test_octree():
    """
    Creates a simple test octree.
    Returns coordinates, node indices, and morton codes.
    """
    # Create a 4x4x4 grid of points
    coords = []
    node_indices = []
    morton_codes = []
    
    test_points = [
        (1, 1, 1),  # Point 0
        (1, 1, 2),  # Point 1 - neighbor of 0
        (2, 1, 1),  # Point 2 - neighbor of 0
        (1, 2, 1),  # Point 3 - neighbor of 0
        (2, 2, 2),  # Point 4 - diagonal neighbor
    ]
    
    for i, (x, y, z) in enumerate(test_points):
        coords.append([x, y, z])
        node_indices.append(i)
        morton_codes.append(compute_morton_code(x, y, z, 1))  # All at level 1
        
    return (np.array(coords, dtype=np.float64),
            np.array(node_indices, dtype=np.int64),
            np.array(morton_codes, dtype=np.int64))

def validate_neighbors(n_neighbors, neighbor_offsets, neighbor_list):
    """
    Validates the neighbor finding results.
    """
    print("\nValidation Results:")
    for i in range(len(n_neighbors)):
        start = neighbor_offsets[i]
        end = start + n_neighbors[i]
        neighbors = neighbor_list[start:end]
        print(f"Node {i} has {n_neighbors[i]} neighbors: {neighbors}")
    
def main():
    # Create test data
    coords, node_indices, morton_codes = create_test_octree()
    
    print("Test Octree Structure:")
    print("Coordinates:\n", coords)
    print("Node Indices:\n", node_indices)
    print("Morton Codes:\n", morton_codes)
    
    # Warm up Numba
    print("\nWarming up Numba...")
    _ = find_neighbors(coords, node_indices, morton_codes)
    
    # Time the neighbor finding
    print("\nTiming neighbor search...")
    n_runs = 100
    start_time = time.time()
    
    for _ in range(n_runs):
        n_neighbors, neighbor_offsets, neighbor_list = find_neighbors(
            coords, node_indices, morton_codes
        )
    
    elapsed = time.time() - start_time
    print(f"Average time per run: {elapsed/n_runs*1000:.3f} ms")
    
    validate_neighbors(n_neighbors, neighbor_offsets, neighbor_list)
    
    print("\nDetailed neighbor analysis:")
    for i in range(len(coords)):
        x, y, z = coords[i]
        print(f"\nPoint ({x}, {y}, {z}):")
        start = neighbor_offsets[i]
        end = start + n_neighbors[i]
        neighbors = neighbor_list[start:end]
        for n in neighbors:
            nx, ny, nz = coords[n]
            dist = np.sqrt((x-nx)**2 + (y-ny)**2 + (z-nz)**2)
            print(f"  - Neighbor at ({nx}, {ny}, {nz}), distance: {dist:.2f}")

if __name__ == "__main__":
    main()