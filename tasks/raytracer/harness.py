
import numpy as np
import numpy.typing as npt 
import matplotlib.pyplot as plt
from tasks.raytracer.bvh import BVH
from numba import jit
import ctypes 

from core.compile import compile_lib

from tasks.raytracer.obj import obj_to_triangles
from tasks.raytracer.reference import Vec3, cast_rays, normalize

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

class BVHData(ctypes.Structure):
    _fields_ = [
        ("node_mins", ctypes.POINTER(Vec3)),
        ("node_maxs", ctypes.POINTER(Vec3)),
        ("node_lefts", ctypes.POINTER(ctypes.c_int)),
        ("node_rights", ctypes.POINTER(ctypes.c_int)),
        ("node_starts", ctypes.POINTER(ctypes.c_int)),
        ("node_ends", ctypes.POINTER(ctypes.c_int))
    ]

def wrapper(raytracer_lib):
    # Define argument types
    raytracer_lib.launchCastRays.argtypes = [
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
        ctypes.c_int
    ]
        
    def cast_rays_gpu(
        ray_origins,
        ray_directions,
        triangles,
        colors,
        bvh,
        depth,
        pixel_seeds
    ):
        n_rays = len(ray_origins)
        
        direct_colors = np.zeros(n_rays, dtype=Vec3)
        did_hits = np.zeros(n_rays, dtype=np.bool_)
        hit_points = np.zeros(n_rays, dtype=Vec3)
        bounce_dirs = np.zeros(n_rays, dtype=Vec3)
        
        raytracer_lib.launchCastRays(
            ray_origins,
            ray_directions,
            triangles,
            colors,
            bvh,
            depth,
            pixel_seeds,
            direct_colors,
            did_hits,
            hit_points,
            bounce_dirs,
            n_rays
        )
        
        return direct_colors, did_hits, hit_points, bounce_dirs

    return cast_rays_gpu

@jit(nopython=True)
def create_rays(width, height, ray_dir, px_seed, aspect_ratio):
    idx = 0
    for i in range(height):
        for j in range(width):
            x = (2 * (j + 0.5) / width - 1) * aspect_ratio
            y = 1 - 2 * (i + 0.5) / height
            ray_dir[idx] = normalize(np.array([x, y, 1.]))
            px_seed[idx] = i * width + j
            idx += 1

class Scene:
    def __init__(self, triangles, colors, max_triangles_in_node=8):
        self.triangles = triangles
        self.colors = colors
        self.bvh_data = BVH(triangles, max_triangles_in_node).flatten()
    
    def render(self, width, height, camera_pos, cast_rays_func):
        
        aspect_ratio = width / height
        
        # Generate all primary rays
        n_rays = width * height
        ray_org = np.tile(camera_pos, (n_rays, 1))
        ray_dir = np.zeros((n_rays, 3))
        px_seed = np.random.rand(n_rays).astype(np.float32) # TODO: set random seeds
                
        create_rays(width, height, ray_dir, px_seed, aspect_ratio)

        # convert arrays to Vec3 to remove overhead
        ray_org = fast_convert_to_vec3(ray_org)
        ray_dir = fast_convert_to_vec3(ray_dir)
        triangles = fast_convert_to_vec3(np.reshape(self.triangles, (-1, 3)))
        colors = fast_convert_to_vec3(self.colors)
        
        # convert BVH data to ctypes struct
        # our tuple is:
        # (node_mins, node_maxs, node_lefts, node_rights, node_starts, node_ends)
        bvh_data = BVHData(
            fast_convert_to_vec3(self.bvh_data[0]).ctypes.data_as(ctypes.POINTER(Vec3)),
            fast_convert_to_vec3(self.bvh_data[1]).ctypes.data_as(ctypes.POINTER(Vec3)),
            self.bvh_data[2].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.bvh_data[3].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.bvh_data[4].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.bvh_data[5].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )
        
        # Cast all primary rays
        print("Casting rays")
        primary_colors, did_hits, hit_points, bounce_dirs = cast_rays_func(
            ray_org, ray_dir, triangles, colors, bvh_data, 0, px_seed
        )
        print(primary_colors[0:3])
        
        primary_colors = fast_convert_from_vec3(primary_colors)
        hit_points = fast_convert_from_vec3(hit_points)
        bounce_dirs = fast_convert_from_vec3(bounce_dirs)
        
        print(primary_colors[0:3])
        
        # Prepare secondary rays where we had hits
        hit_mask = did_hits
        if np.any(hit_mask):
            # Offset hit points slightly along bounce direction to avoid self-intersection
            secondary_origins = hit_points[hit_mask] + bounce_dirs[hit_mask] * 0.001
            secondary_directions = bounce_dirs[hit_mask]
            secondary_seeds = px_seed[hit_mask]
            
            secondary_origins = fast_convert_to_vec3(secondary_origins)
            secondary_directions = fast_convert_to_vec3(secondary_directions)
            
            # Cast secondary rays
            secondary_colors, _, _, _ = cast_rays_func(
                secondary_origins, 
                secondary_directions,
                triangles, 
                colors,
                bvh_data,
                1,
                secondary_seeds
            )
            secondary_colors = fast_convert_from_vec3(secondary_colors)
            
            # Combine primary and secondary colors where we had hits
            final_colors = primary_colors.copy()
            final_colors[hit_mask] = primary_colors[hit_mask] * 0.7 + secondary_colors * 0.3
        else:
            final_colors = primary_colors
        
        # Reshape into image
        return np.clip(final_colors.reshape(height, width, 3), 0, 1)

def eval():
    tris_ground = obj_to_triangles(open('./tasks/raytracer/assets/ground.obj').read())
    tris_object = obj_to_triangles(open('./tasks/raytracer/assets/pyramid.obj').read())
    #tris_object = 15*obj_to_triangles(open('./tasks/raytracer/assets/bunny.obj').read())
    tris = np.concatenate([tris_ground, tris_object])
    
    colors_ground = np.array([
        [0.8, 0.8, 0.8],  # Ground - light gray
        [0.8, 0.8, 0.8],  # Ground - light gray
    ])
    
    colors_object = np.array([
        [0.2, 0.2, 1.0],  # Pyramid - blue
        [1.0, 0.2, 0.2],  # Pyramid - red
        [0.2, 1.0, 0.2],  # Pyramid - green
        [1.0, 1.0, 0.2]   # Pyramid - yellow
    ])
    # # array of blue color for the object
    # colors_object = np.zeros((len(tris_object), 3))
    # colors_object[:, 0] = 0.1
    # colors_object[:, 1] = 0.2
    # colors_object[:, 2] = 1.0
    
    colors = np.concatenate([colors_ground, colors_object])

    scene = Scene(tris, colors)

    # gpu library 
    lib = compile_lib('./example-solutions/raytracer.cu')
    if lib is None:
        print("GPU implementation not available")
        return
    
    cast_rays_func = wrapper(lib)
    
    # Render the scene
    width, height = 1600, 1200
    print("Rendering scene...")
    image0 = scene.render(width, height, np.array([0., 1., -3.]), cast_rays_func)
    #print("Rescene.ndering scene...")
    #image1 = scene.render(width, height, np.array([0., 1.2, -3.]))

    # Display the result
    plt.figure(figsize=(10, 7.5))
    plt.imshow(image0)
    plt.axis('off')
    plt.show()

    # Save the result
    plt.imsave('raytracing0.png', image0)
    #plt.imsave('raytracing1.png', image1)
    
if __name__ == "__main__":
    eval()