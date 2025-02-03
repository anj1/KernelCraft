import numpy as np 

class AABB:
    def __init__(self, min_bound=None, max_bound=None):
        self.min_bound = min_bound if min_bound is not None else np.array([float('inf')] * 3)
        self.max_bound = max_bound if max_bound is not None else np.array([-float('inf')] * 3)
    
    def expand(self, point):
        self.min_bound = np.minimum(self.min_bound, point)
        self.max_bound = np.maximum(self.max_bound, point)
    
    def union(self, other):
        return AABB(
            np.minimum(self.min_bound, other.min_bound),
            np.maximum(self.max_bound, other.max_bound)
        )
    
    def surface_area(self):
        extent = self.max_bound - self.min_bound
        return 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])

class BVHNode:
    def __init__(self, triangles, start, end):
        self.left = None
        self.right = None
        self.aabb = AABB()
        self.start = start
        self.end = end
        
        # Compute bounding box for all triangles in this node
        for i in range(start, end):
            for vertex in triangles[i]:
                self.aabb.expand(vertex)

def compute_triangle_centroid(triangle):
    return np.mean(triangle, axis=0)

class BVH:
    def __init__(self, triangles, max_triangles_in_node=2):
        self.triangles = triangles
        self.max_triangles_in_node = max_triangles_in_node
        self.root = self._build(0, len(triangles))
    
    def _build(self, start, end):
        node = BVHNode(self.triangles, start, end)
        
        if end - start <= self.max_triangles_in_node:
            return node
        
        # Find the axis with the largest extent
        centroids = np.array([compute_triangle_centroid(triangle) for triangle in self.triangles[start:end]])
        extent = np.max(centroids, axis=0) - np.min(centroids, axis=0)
        axis = np.argmax(extent)
        
        # Sort triangles along the chosen axis
        mid = (start + end) // 2
        indices = np.argsort([compute_triangle_centroid(triangle)[axis] 
                            for triangle in self.triangles[start:end]])
        
        # Reorder triangles based on split
        self.triangles[start:end] = self.triangles[start:end][indices]
        
        # Recursively build child nodes
        node.left = self._build(start, mid)
        node.right = self._build(mid, end)
        
        return node
    
    def traverse(self, node=None, level=0):
        """Helper method to visualize the BVH structure"""
        if node is None:
            node = self.root
            
        prefix = "  " * level
        print(f"{prefix}Node: {node.start}-{node.end}")
        print(f"{prefix}AABB: min={node.aabb.min_bound}, max={node.aabb.max_bound}")
        
        if node.left:
            print(f"{prefix}Left:")
            self.traverse(node.left, level + 1)
        if node.right:
            print(f"{prefix}Right:")
            self.traverse(node.right, level + 1)
            
    def flatten(self):
        """Convert BVH tree to flat arrays for kernel"""
        nodes = []
        def collect_nodes(node, idx):
            if idx >= len(nodes):
                nodes.append(node)
            
            if node.left is not None:
                left_idx = len(nodes)
                collect_nodes(node.left, left_idx)
            else:
                left_idx = -1
                
            if node.right is not None:
                right_idx = len(nodes)
                collect_nodes(node.right, right_idx)
            else:
                right_idx = -1
                
            nodes[idx].left_idx = left_idx
            nodes[idx].right_idx = right_idx
        
        collect_nodes(self.root, 0)
        
        node_mins = np.array([node.aabb.min_bound for node in nodes])
        node_maxs = np.array([node.aabb.max_bound for node in nodes])
        node_lefts = np.array([node.left_idx for node in nodes])
        node_rights = np.array([node.right_idx for node in nodes])
        node_starts = np.array([node.start for node in nodes])
        node_ends = np.array([node.end for node in nodes])
        
        return (node_mins, node_maxs, node_lefts, node_rights, node_starts, node_ends)
