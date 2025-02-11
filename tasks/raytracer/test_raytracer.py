# gpu_kernel_eval/tests/benchmarks/test_raytracer.py

import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import NamedTuple, Tuple, Optional

from tasks.raytracer.wrapper import RaytracerKernel as RaytracerKernelGPU
from tasks.raytracer.reference import cast_rays as cast_rays_cpu
from tasks.raytracer.scene import Scene, create_rays_vectorized
from tasks.raytracer.ply import load_ply_triangles

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
    
@pytest.fixture(scope="module")
def scene():
    """Create scene fixture used across tests"""
    # Load meshes
    with open('./tasks/raytracer/assets/ground.ply', 'rb') as f:
        tris_ground = load_ply_triangles(f)
    with open('./tasks/raytracer/assets/bunny.ply', 'rb') as f:
        tris_object = 15 * load_ply_triangles(f) * [1, -1, 1]
        tris_object += [0, -np.min(tris_object[:, 1]) - 2.5, 0]
    
    triangles = np.concatenate([tris_ground, tris_object])
    colors = np.concatenate([
        np.array([[1.0, 0.4, 0.4], [0.4, 0.4, 1.0]]),  # ground
        np.ones((len(tris_object), 3))                  # bunny
    ])
    
    return Scene(triangles, colors, max_triangles_in_node=8)

@pytest.fixture(scope="module")
def raytracer(scene):
    """Create GPU raytracer fixture"""
    width, height = 1600, 1200
    
    raytracer_gpu = RaytracerKernelGPU()
    raytracer_gpu.initialize(
        width * height,
        len(scene.triangles),
        len(scene.bvh_data.node_starts)
    )
    
    raytracer_cpu = RaytracerKernelCPU()
    
    yield {"gpu": raytracer_gpu, "cpu": raytracer_cpu}
    raytracer_gpu.cleanup()

@pytest.fixture
def ray_batch(batch_size=1600*1200):
    """Generate a batch of rays for testing"""
    camera_pos = np.array([0., 1., -3.])
    ray_origins = np.tile(camera_pos, (batch_size, 1))
    ray_dirs = create_rays_vectorized(1600, 1200, 1600/1200)
    pixel_seeds = np.random.rand(batch_size).astype(np.float32)
    return ray_origins, ray_dirs, pixel_seeds

class TestRaytracer:
    """Test suite for raytracer implementation"""
    
    def test_basic_render(self, scene, raytracer, ray_batch):
        """Test basic render produces valid output"""
        ray_origins, ray_dirs, pixel_seeds = ray_batch
        
        # Run GPU implementation
        colors, hits, points, bounce = raytracer["gpu"].cast_rays(
            ray_origins, ray_dirs, scene.triangles, scene.colors,
            scene.bvh_data, depth=0, pixel_seeds=pixel_seeds
        )
        
        # Basic validation
        assert colors.shape == (len(ray_origins), 3)
        assert hits.shape == (len(ray_origins),)
        assert points.shape == (len(ray_origins), 3)
        assert bounce.shape == (len(ray_origins), 3)
        assert np.all(colors >= 0) and np.all(colors <= 1)
    
    def test_against_reference(self, scene, raytracer, ray_batch, benchmark):
        """Compare GPU implementation against CPU reference"""
        ray_origins, ray_dirs, pixel_seeds = ray_batch
        
        # Run GPU implementation
        gpu_colors, gpu_hits, gpu_points, gpu_bounce = raytracer["gpu"].cast_rays(
            ray_origins, ray_dirs, scene.triangles, scene.colors,
            scene.bvh_data, depth=0, pixel_seeds=pixel_seeds
        )
        
        # Run CPU reference
        cpu_colors, cpu_hits, cpu_points, cpu_bounce = raytracer["cpu"].cast_rays(
            ray_origins, ray_dirs, scene.triangles, scene.colors,
            scene.bvh_data, depth=0, pixel_seeds=pixel_seeds
        )
        # show mismatched pixels
        mismatched = np.where(np.abs(gpu_colors - cpu_colors) > 1e-5)[0]
        if len(mismatched) > 0:
            print(f"Mismatched pixels: {len(mismatched)}")
            print(gpu_colors[mismatched[0:32]])
            print(cpu_colors[mismatched[0:32]])
        
        # write to file 
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        plt.imsave(output_dir / "gpu.png", np.clip(gpu_colors.reshape(1200, 1600, 3), 0, 1))   
        plt.imsave(output_dir / "cpu.png", np.clip(cpu_colors.reshape(1200, 1600, 3), 0, 1))   
        
        # Compare results
        np.testing.assert_allclose(gpu_colors, cpu_colors, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(gpu_hits, cpu_hits)
        np.testing.assert_allclose(gpu_points, cpu_points, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(gpu_bounce, cpu_bounce, rtol=1e-5, atol=1e-5)
        


@pytest.mark.benchmark
class TestRaytracerPerformance:
    """Performance benchmarks for raytracer"""
    
    @pytest.mark.parametrize("batch_size", [
        640*480,        # 480p
        1920*1080,      # Full HD
    ])
    def test_kernel_performance(self, scene, raytracer, batch_size, benchmark):
        """Benchmark kernel performance at different resolutions"""
        camera_pos = np.array([0., 1., -3.])
        ray_origins = np.tile(camera_pos, (batch_size, 1))
        ray_dirs = np.random.randn(batch_size, 3)  # Random directions for bench
        ray_dirs /= np.linalg.norm(ray_dirs, axis=1)[:, np.newaxis]
        pixel_seeds = np.random.rand(batch_size).astype(np.float32)
        
        def run_kernel():
            return raytracer["gpu"].cast_rays(
                ray_origins, ray_dirs, scene.triangles, scene.colors,
                scene.bvh_data, depth=0, pixel_seeds=pixel_seeds
            )
        
        # Run benchmark
        #result = benchmark(run_kernel)
        benchmark.pedantic(run_kernel, iterations=1, rounds=5)
        
        # Optional: Save benchmark results
        benchmark.extra_info.update({
            'batch_size': batch_size,
            'rays_per_second': batch_size / benchmark.stats['mean']
        })
    
    def test_multisampled_performance(self, scene, raytracer, benchmark):
        """Benchmark multisampled rendering performance"""
        width, height = 1600, 1200
        n_samples = 2 #128
        camera_pos = np.array([0., 1., -3.])
        
        def run_multisampled():
            image = np.zeros((height, width, 3), dtype=np.float32)
            for _ in range(n_samples):
                ray_dirs = create_rays_vectorized(width, height, width/height)
                ray_dirs += np.random.uniform(-0.001, 0.001, ray_dirs.shape)
                ray_origins = np.tile(camera_pos, (width*height, 1))
                pixel_seeds = np.random.rand(width*height).astype(np.float32)
                
                colors, _, _, _ = raytracer["gpu"].cast_rays(
                    ray_origins, ray_dirs, scene.triangles, scene.colors,
                    scene.bvh_data, depth=0, pixel_seeds=pixel_seeds
                )
                image += colors.reshape(height, width, 3)
            
            return image / n_samples
        
        # Run benchmark
        result = benchmark(run_multisampled)
        
        # Optional: Save result image
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        #plt.imsave(output_dir / "multisampled.png", np.clip(result.stats['last_run'], 0, 1))
        result = benchmark.pedantic(run_multisampled, iterations=1, rounds=5)
        plt.imsave(output_dir / "multisampled.png", np.clip(result, 0, 1))

def main():
    """Run benchmarks from command line"""
    pytest.main([__file__, "-v", "--benchmark-only"])

if __name__ == "__main__":
    main()