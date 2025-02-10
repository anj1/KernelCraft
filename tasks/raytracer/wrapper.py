# gpu_kernel_eval/kernels/raytracer/wrapper.py

import ctypes
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
from core.compiler import CUDACompiler

@dataclass
class Vec3(ctypes.Structure):
    """Vector3 structure matching CUDA implementation"""
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float)
    ]


class BVHData(ctypes.Structure):
    _fields_ = [
        ("n_nodes", ctypes.c_size_t),
        ("node_mins", ctypes.POINTER(Vec3)),
        ("node_maxs", ctypes.POINTER(Vec3)),
        ("node_lefts", ctypes.POINTER(ctypes.c_int32)),
        ("node_rights", ctypes.POINTER(ctypes.c_int32)),
        ("node_starts", ctypes.POINTER(ctypes.c_int32)),
        ("node_ends", ctypes.POINTER(ctypes.c_int32))
    ]

# Fast conversion function
def fast_convert_to_vec3(arr):
    vec3_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # Ensure input is float32 to match Vec3's c_float
    arr = arr.astype(np.float32)
    
    # Create structured array
    structured = np.empty(arr.shape[0], dtype=vec3_dtype)
    structured['x'] = arr[:, 0]
    structured['y'] = arr[:, 1]
    structured['z'] = arr[:, 2]
    
    # Convert to Vec3 array using frombuffer
    return np.frombuffer(structured.tobytes(), dtype=np.dtype(Vec3)).copy()

def fast_convert_from_vec3(arr):
    vec3_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # Create structured array
    structured = np.frombuffer(arr.tobytes(), dtype=vec3_dtype)
    
    # Convert to regular array
    return np.stack([structured['x'], structured['y'], structured['z']], axis=-1)

def fast_convert_to_ptr(arr):
    if isinstance(arr, int):
        return arr, None
    elif arr.dtype == np.dtype(Vec3):
        return arr.ctypes.data_as(ctypes.POINTER(Vec3)), arr
    elif (arr.dtype == np.float32 or arr.dtype == np.float64) and arr.ndim == 2 and arr.shape[1] == 3:
        vec3_arr = fast_convert_to_vec3(arr)
        return vec3_arr.ctypes.data_as(ctypes.POINTER(Vec3)), vec3_arr
    elif arr.dtype == np.int32:
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), arr
    else:
        raise ValueError(f"Unsupported array type: {arr.dtype}")
    
    
@dataclass
class BVHData(ctypes.Structure):
    """BVH data structure matching CUDA implementation"""
    _fields_ = [
        ("n_nodes", ctypes.c_size_t),
        ("node_mins", ctypes.POINTER(Vec3)),
        ("node_maxs", ctypes.POINTER(Vec3)),
        ("node_lefts", ctypes.POINTER(ctypes.c_int32)),
        ("node_rights", ctypes.POINTER(ctypes.c_int32)),
        ("node_starts", ctypes.POINTER(ctypes.c_int32)),
        ("node_ends", ctypes.POINTER(ctypes.c_int32))
    ]

def fast_convert_to_vec3(arr: np.ndarray) -> np.ndarray:
    """Convert numpy array to array of Vec3 structures efficiently"""
    vec3_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # Ensure input is float32
    arr = arr.astype(np.float32)
    
    # Create structured array
    structured = np.empty(arr.shape[0], dtype=vec3_dtype)
    structured['x'] = arr[:, 0]
    structured['y'] = arr[:, 1]
    structured['z'] = arr[:, 2]
    
    # Convert to Vec3 array
    return np.frombuffer(structured.tobytes(), dtype=np.dtype(Vec3)).copy()

def fast_convert_from_vec3(arr: np.ndarray) -> np.ndarray:
    """Convert array of Vec3 structures back to numpy array efficiently"""
    vec3_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # Convert to structured array
    structured = np.frombuffer(arr.tobytes(), dtype=vec3_dtype)
    
    # Convert to regular array
    return np.stack([structured['x'], structured['y'], structured['z']], axis=-1)

class RaytracerKernel:
    def __init__(self, lib_path: Optional[Path] = None):
        """
        Initialize raytracer kernel wrapper
        
        Args:
            lib_path: Optional path to pre-compiled library
        """
        
        # Compile or load library
        if lib_path is None:
            source_path = Path("example-solutions") / "raytracer.cu"
            compiler = CUDACompiler()
            self.lib_path = compiler.compile_file(source_path)
        else:
            self.lib_path = lib_path
            
        self.lib = ctypes.CDLL(str(self.lib_path))
        self._setup_function_types()
        self.gpu_resources = None
        
    def _setup_function_types(self):
        """Setup ctypes function signatures"""
        # Initialize GPU resources
        self.lib.initialize_gpu.argtypes = [
            ctypes.c_size_t,  # n_rays
            ctypes.c_size_t,  # n_triangles
            ctypes.c_size_t   # n_bvh_nodes
        ]
        self.lib.initialize_gpu.restype = ctypes.c_void_p
        
        # Cast rays kernel
        self.lib.launchCastRays.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=Vec3),
            np.ctypeslib.ndpointer(dtype=Vec3),
            np.ctypeslib.ndpointer(dtype=Vec3),
            np.ctypeslib.ndpointer(dtype=Vec3),
            BVHData,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=ctypes.c_float),
            np.ctypeslib.ndpointer(dtype=Vec3),
            np.ctypeslib.ndpointer(dtype=np.bool_),
            np.ctypeslib.ndpointer(dtype=Vec3),
            np.ctypeslib.ndpointer(dtype=Vec3),
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        
        # Cleanup
        self.lib.cleanup_gpu.argtypes = [ctypes.c_void_p]
        
    def initialize(self, n_rays: int, n_triangles: int, n_bvh_nodes: int):
        """
        Initialize GPU resources
        
        Args:
            n_rays: Number of rays to trace
            n_triangles: Number of triangles in scene
            n_bvh_nodes: Number of BVH nodes
        """
        if self.gpu_resources is not None:
            self.cleanup()
            
        self.gpu_resources = self.lib.initialize_gpu(n_rays, n_triangles, n_bvh_nodes)
        if not self.gpu_resources:
            raise RuntimeError("Failed to initialize GPU resources")
            
    def cast_rays(self,
                 ray_origins: np.ndarray,
                 ray_directions: np.ndarray,
                 triangles: np.ndarray,
                 colors: np.ndarray,
                 bvh_data: dict,
                 depth: int = 0,
                 pixel_seeds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Cast rays through the scene
        
        Args:
            ray_origins: (N, 3) array of ray origin points
            ray_directions: (N, 3) array of ray direction vectors
            triangles: (M*3, 3) array of triangle vertices
            colors: (M, 3) array of triangle colors
            bvh_data: Dictionary containing BVH node data
            depth: Ray bounce depth
            pixel_seeds: Optional (N,) array of random seeds
            
        Returns:
            Tuple of (direct_colors, did_hits, hit_points, bounce_dirs)
        """
        if self.gpu_resources is None:
            raise RuntimeError("GPU resources not initialized")
            
        n_rays = len(ray_origins)
        n_triangles = len(triangles) // 3
        
        # Generate random seeds if not provided
        if pixel_seeds is None:
            pixel_seeds = np.random.rand(n_rays).astype(np.float32)
            
        # Convert input arrays to Vec3 format
        ray_origins_v3 = fast_convert_to_vec3(ray_origins)
        ray_directions_v3 = fast_convert_to_vec3(ray_directions)
        triangles_v3 = fast_convert_to_vec3(np.reshape(triangles, (-1, 3)))
        colors_v3 = fast_convert_to_vec3(colors)
        pixel_seeds = pixel_seeds.astype(np.float32)        
        
        self._bvh_arrays = {}
        bvh = BVHData()
        for field, value in bvh_data.items():
            #print(field)
            ptr, arr = fast_convert_to_ptr(value)
            setattr(bvh, field, ptr)
            self._bvh_arrays[field] = arr
        
        # Prepare output arrays
        direct_colors = np.empty(n_rays, dtype=Vec3)
        did_hits = np.empty(n_rays, dtype=np.bool_)
        hit_points = np.empty(n_rays, dtype=Vec3)
        bounce_dirs = np.empty(n_rays, dtype=Vec3)
        
        # Launch kernel
        self.lib.launchCastRays(
            self.gpu_resources,
            ray_origins_v3,
            ray_directions_v3,
            triangles_v3,
            colors_v3,
            bvh,
            depth,
            pixel_seeds,
            direct_colors,
            did_hits,
            hit_points,
            bounce_dirs,
            n_rays,
            n_triangles
        )
        
        # Convert outputs back to numpy arrays
        return (
            fast_convert_from_vec3(direct_colors),
            did_hits,
            fast_convert_from_vec3(hit_points),
            fast_convert_from_vec3(bounce_dirs)
        )
        
    def cleanup(self):
        """Release GPU resources"""
        if self.gpu_resources is not None:
            self.lib.cleanup_gpu(self.gpu_resources)
            self.gpu_resources = None
            
    def __del__(self):
        """Ensure GPU resources are cleaned up"""
        self.cleanup()