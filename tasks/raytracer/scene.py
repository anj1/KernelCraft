import ctypes 
import numpy as np
from .bvh import BVH

def create_rays_vectorized(width, height, aspect_ratio):
    # Create coordinate grids
    x = np.linspace(0.5/width, 1 - 0.5/width, width) * 2 - 1
    y = np.linspace(1 - 0.5/height, 0.5/height, height) * 2 - 1
    
    xx, yy = np.meshgrid(x * aspect_ratio, y)
    
    rays = np.stack([xx.ravel(), yy.ravel(), np.ones(width * height)], axis=1)
    rays /= np.linalg.norm(rays, axis=1)[:, np.newaxis]
    
    return rays

class Scene:
    def __init__(self, triangles, colors, max_triangles_in_node=8):
        self.triangles = triangles
        self.colors = colors
        self.bvh_data = BVH(triangles, max_triangles_in_node).flatten()
    
    def render(self, width, height, camera_pos, raytracer, px_seed=None, ray_dir=None):
        
        aspect_ratio = width / height
        
        # Generate all primary rays
        n_rays = width * height
        ray_org = np.tile(camera_pos, (n_rays, 1))
        if px_seed is None:
            px_seed = np.zeros(n_rays, dtype=np.float32)
                
        if ray_dir is None:
            ray_dir = create_rays_vectorized(width, height, aspect_ratio)

        
        # Cast all primary rays
        triangles = np.reshape(self.triangles, (-1, 3))
        primary_colors, did_hits, hit_points, bounce_dirs = raytracer.cast_rays(
            ray_org, ray_dir, triangles, self.colors, self.bvh_data, 0, px_seed
        )
                
        # Prepare secondary rays where we had hits
        hit_mask = did_hits
        if np.any(hit_mask):
            # Offset hit points slightly along bounce direction to avoid self-intersection
            secondary_origins = hit_points[hit_mask] + bounce_dirs[hit_mask] * 0.001
            secondary_directions = bounce_dirs[hit_mask]
            secondary_seeds = px_seed[hit_mask]
            
            #secondary_origins = fast_convert_to_vec3(secondary_origins)
            #secondary_directions = fast_convert_to_vec3(secondary_directions)
            
            # Cast secondary rays
            secondary_colors, _, _, _ = raytracer.cast_rays(
                secondary_origins, 
                secondary_directions,
                triangles, 
                self.colors,
                self.bvh_data,
                1,
                secondary_seeds
            )
            #secondary_colors = fast_convert_from_vec3(secondary_colors)
            
            # Combine primary and secondary colors where we had hits
            final_colors = primary_colors.copy()
            final_colors[hit_mask] = primary_colors[hit_mask] * 0.7 + secondary_colors * 0.3
        else:
            final_colors = primary_colors
        
        # Reshape into image
        return np.clip(final_colors.reshape(height, width, 3), 0, 1)
