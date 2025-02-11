# gpu_kernel_eval/kernels/raytracer/evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union

import time

from .wrapper import RaytracerKernel
from .scene import Scene, create_rays_vectorized
from .ply import load_ply_triangles
from .reference import cast_rays as cast_rays_cpu

class RaytracerKernelCPU:
    def __init__(self):
        pass
            
    def cast_rays(self,
                 ray_origins: np.ndarray,
                 ray_directions: np.ndarray,
                 triangles: np.ndarray,
                 colors: np.ndarray,
                 bvh_data: NamedTuple,
                 depth: int = 0,
                 pixel_seeds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Cast rays through the scene
        """
        return cast_rays_cpu(ray_origins, ray_directions, triangles, colors, bvh_data, depth, pixel_seeds)
    
    
class RaytracerEvaluator:
    def __init__(self, 
                 width: int = 1600, 
                 height: int = 1200,
                 max_triangles_in_node: int = 8,
                 output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize raytracer evaluator
        
        Args:
            width: Image width
            height: Image height
            max_triangles_in_node: Max triangles per BVH node
            output_dir: Directory for saving results
        """
        self.width = width
        self.height = height
        self.max_triangles_in_node = max_triangles_in_node
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load scene data
        self.scene = self._load_scene()
        
        # Initialize GPU implementation
        self.gpu_raytracer = RaytracerKernel()
        self.gpu_raytracer.initialize(
            width * height,
            len(self.scene.triangles),
            len(self.scene.bvh_data.node_starts)
        )
        
        self.cpu_raytracer = RaytracerKernelCPU() # TODO: Replace with CPU implementation
        
    def _load_scene(self) -> Scene:
        """Load and prepare scene data"""
        # Load ground plane
        with open('./tasks/raytracer/assets/ground.ply', 'rb') as f:
            tris_ground = load_ply_triangles(f)
            
        # Load and transform bunny model    
        with open('./tasks/raytracer/assets/bunny.ply', 'rb') as f:
            tris_object = 15 * load_ply_triangles(f) * [1, -1, 1]
            # Place on ground plane
            tris_object += [0, -np.min(tris_object[:, 1]) - 2.3, 0]
            
        # Combine meshes
        triangles = np.concatenate([tris_ground, tris_object])
        
        # Setup colors
        colors_ground = np.array([
            [1.0, 0.4, 0.4],
            [0.4, 0.4, 1.0]
        ])
        colors_object = np.ones((len(tris_object), 3))
        colors = np.concatenate([colors_ground, colors_object])
        
        return Scene(triangles, colors, self.max_triangles_in_node)
        
    def render_single(self, 
                     px_seed,
                     implementation: str = 'gpu',
                     camera_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Render single image using specified implementation
        
        Args:
            implementation: 'gpu' or 'cpu'
            camera_pos: Optional camera position override
            
        Returns:
            Tuple of (rendered image, render time in seconds)
        """
        if camera_pos is None:
            camera_pos = np.array([0., 1., -3.])
            
        n_rays = self.width * self.height
        ray_org = np.tile(camera_pos, (n_rays, 1))
        ray_dir = create_rays_vectorized(self.width, self.height, self.width / self.height)
        
        start_time = time.perf_counter()
        
        if implementation == 'gpu':
            image = self.scene.render(
                self.width, self.height,
                camera_pos,
                self.gpu_raytracer,
                px_seed,
                ray_dir
            )
        else:
            image = self.scene.render(
                self.width, self.height,
                camera_pos,
                self.cpu_raytracer,
                px_seed,
                ray_dir
            )
            
        render_time = time.perf_counter() - start_time
        
        return np.clip(image, 0, 1), render_time
        
    def render_multisampled(self,
                          n_samples: int = 128,
                          implementation: str = 'gpu',
                          camera_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Render with multisampling for anti-aliasing
        
        Args:
            n_samples: Number of samples per pixel
            implementation: 'gpu' or 'cpu'
            camera_pos: Optional camera position override
            
        Returns:
            Tuple of (rendered image, total render time in seconds)
        """
        if camera_pos is None:
            camera_pos = np.array([0., 1., -3.])
            
        image_accum = np.zeros((self.height, self.width, 3), dtype=np.float32)
        total_time = 0
        
        for i in range(n_samples):
            n_rays = self.width * self.height
            ray_dir = create_rays_vectorized(self.width, self.height, self.width / self.height)
            # Add jitter for anti-aliasing
            ray_dir += np.random.uniform(-0.001, 0.001, (n_rays, 3))
            
            px_seed = np.random.rand(n_rays).astype(np.float32)
            image, render_time = self.render_single(px_seed, implementation, camera_pos)
            image_accum += image
            total_time += render_time
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{n_samples} samples")
                
        return np.clip(image_accum / n_samples, 0, 1), total_time
        
    def run_comparison(self,
                      n_samples: int = 128,
                      save_results: bool = True) -> None:
        """Run full comparison between GPU and CPU implementations"""
        print("Rendering single sample comparison...")
        
        # Single sample comparison
        px_seed = np.zeros(self.width*self.height, dtype=np.float32)
        gpu_image, gpu_time = self.render_single(px_seed, 'gpu')
        cpu_image, cpu_time = self.render_single(px_seed, 'cpu')
        
        print(f"Single sample render times:")
        print(f"  GPU: {gpu_time:.3f}s")
        print(f"  CPU: {cpu_time:.3f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
        
        # Compute difference
        diff = gpu_image - cpu_image
        max_diff = np.max(np.abs(diff))
        rms_diff = np.sqrt(np.mean(diff ** 2))
        print(f"\nOutput differences:")
        print(f"  Max pixel difference: {max_diff:.6f}")
        print(f"  Root mean squared pixel difference: {rms_diff:.6f}")
        
        if save_results:
            plt.imsave(self.output_dir / 'single_gpu.png', gpu_image)
            plt.imsave(self.output_dir / 'single_cpu.png', cpu_image)
            #plt.imsave(self.output_dir / 'single_diff.png', diff * 10)  # Multiply for visibility
            
        # Multisampled comparison
        print(f"\nRendering {n_samples} sample comparison...")
        
        gpu_ms_image, gpu_ms_time = self.render_multisampled(n_samples, 'gpu')
        if save_results:
            plt.imsave(self.output_dir / 'ms_gpu.png', gpu_ms_image)
        cpu_ms_image, cpu_ms_time = self.render_multisampled(n_samples, 'cpu')
        if save_results:
            plt.imsave(self.output_dir / 'ms_cpu.png', cpu_ms_image)
        
        print(f"\nMultisampled render times:")
        print(f"  GPU: {gpu_ms_time:.3f}s")
        print(f"  CPU: {cpu_ms_time:.3f}s")
        print(f"  Speedup: {cpu_ms_time/gpu_ms_time:.1f}x")
        
        # Compute difference
        ms_diff = np.abs(gpu_ms_image - cpu_ms_image)
        ms_max_diff = np.max(ms_diff)
        ms_mean_diff = np.mean(ms_diff)
        print(f"\nMultisampled output differences:")
        print(f"  Max pixel difference: {ms_max_diff:.6f}")
        print(f"  Mean pixel difference: {ms_mean_diff:.6f}")
            
    def cleanup(self):
        """Release GPU resources"""
        if hasattr(self, 'gpu_raytracer'):
            self.gpu_raytracer.cleanup()

def main():
    """Main evaluation entry point"""
    try:
        evaluator = RaytracerEvaluator()
        evaluator.run_comparison()
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    main()