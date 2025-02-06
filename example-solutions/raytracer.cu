#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <math.h>
#include <iostream> 

struct Vec3 {
    float x, y, z;
    
    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __device__ Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    __device__ Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    __device__ Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
};

struct BVHData {
    size_t n_nodes;
    Vec3* node_mins;
    Vec3* node_maxs;
    int32_t* node_lefts;
    int32_t* node_rights;
    int32_t* node_starts;
    int32_t* node_ends;
};

__device__ float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vec3 normalize(const Vec3& v) {
    float norm = sqrtf(dot(v, v));
    if (norm > 0) {
        return Vec3(v.x / norm, v.y / norm, v.z / norm);
    }
    return v;
}

__device__ Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ Vec3 random_in_unit_sphere(float seed) {
    float x = sinf(seed * 78.233f + 0.5f) * 0.5f + 0.5f;
    float y = sinf(seed * 127.432f + 1.3f) * 0.5f + 0.5f;
    float z = sinf(seed * 213.835f + 2.7f) * 0.5f + 0.5f;
    
    Vec3 vec(x * 2.0f - 1.0f, y * 2.0f - 1.0f, z * 2.0f - 1.0f);
    return normalize(vec);
}

__device__ bool aabb_intersect(const Vec3& ray_origin, const Vec3& ray_direction, 
                             const Vec3& min_bound, const Vec3& max_bound) {
    Vec3 inv_dir(1.0f / ray_direction.x, 1.0f / ray_direction.y, 1.0f / ray_direction.z);
    
    Vec3 t1 = Vec3(
        (min_bound.x - ray_origin.x) * inv_dir.x,
        (min_bound.y - ray_origin.y) * inv_dir.y,
        (min_bound.z - ray_origin.z) * inv_dir.z
    );
    
    Vec3 t2 = Vec3(
        (max_bound.x - ray_origin.x) * inv_dir.x,
        (max_bound.y - ray_origin.y) * inv_dir.y,
        (max_bound.z - ray_origin.z) * inv_dir.z
    );
    
    float tmin = fmaxf(fmaxf(fminf(t1.x, t2.x), fminf(t1.y, t2.y)), fminf(t1.z, t2.z));
    float tmax = fminf(fminf(fmaxf(t1.x, t2.x), fmaxf(t1.y, t2.y)), fmaxf(t1.z, t2.z));
    
    return tmin <= tmax && tmax > 0;
}

__device__ bool triangle_intersect(const Vec3& v0, const Vec3& v1, const Vec3& v2,
                                 const Vec3& ray_origin, const Vec3& ray_direction,
                                 float& t_out, Vec3& normal_out) {
    const float epsilon = 1e-6f;
    
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = cross(ray_direction, edge2);
    float a = dot(edge1, h);
    
    if (fabsf(a) < epsilon) {
        return false;
    }
    
    float f = 1.0f / a;
    Vec3 s = Vec3(
        ray_origin.x - v0.x,
        ray_origin.y - v0.y,
        ray_origin.z - v0.z
    );
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    
    Vec3 q = cross(s, edge1);
    float v = f * dot(ray_direction, q);
    
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    
    t_out = f * dot(edge2, q);
    
    if (t_out < epsilon) {
        return false;
    }
    
    normal_out = normalize(cross(edge1, edge2));
    return true;
}

// Helper function to get minimum intersection distance with AABB
__device__ float aabb_intersection_distance(const Vec3& ray_origin, const Vec3& ray_direction,
                                          const Vec3& aabb_min, const Vec3& aabb_max) {
    float t_min = 0.0f;
    float t_max = CUDART_INF_F;
    
    // X axis intersection
    float t1 = (aabb_min.x - ray_origin.x) / ray_direction.x;
    float t2 = (aabb_max.x - ray_origin.x) / ray_direction.x;
    if (ray_direction.x < 0.0f) {
        float temp = t1;
        t1 = t2;
        t2 = temp;
    }
    t_min = max(t_min, t1);
    t_max = min(t_max, t2);
    
    // Y axis intersection
    t1 = (aabb_min.y - ray_origin.y) / ray_direction.y;
    t2 = (aabb_max.y - ray_origin.y) / ray_direction.y;
    if (ray_direction.y < 0.0f) {
        float temp = t1;
        t1 = t2;
        t2 = temp;
    }
    t_min = max(t_min, t1);
    t_max = min(t_max, t2);
    
    // Z axis intersection
    t1 = (aabb_min.z - ray_origin.z) / ray_direction.z;
    t2 = (aabb_max.z - ray_origin.z) / ray_direction.z;
    if (ray_direction.z < 0.0f) {
        float temp = t1;
        t1 = t2;
        t2 = temp;
    }
    t_min = max(t_min, t1);
    t_max = min(t_max, t2);
    
    return (t_max >= t_min) ? t_min : CUDART_INF_F;
}

// Stack entry for BVH traversal
struct StackEntry {
    int32_t node_idx;
    float t_min;  // Used for sorting traversal order
};

__device__ bool intersect_bvh_node(const Vec3& ray_origin, const Vec3& ray_direction,
                                 const Vec3* triangles, const Vec3* colors,
                                 const BVHData& bvh, int32_t root_idx,
                                 float& closest_t, Vec3& closest_normal, Vec3& closest_color) {
    // Stack for iterative traversal - adjust size based on your BVH depth
    constexpr int MAX_STACK = 64;
    StackEntry stack[MAX_STACK];
    int stack_ptr = 0;
    
    bool hit_anything = false;
    closest_t = CUDART_INF_F;
    
    // Push root node
    stack[stack_ptr++] = {root_idx, 0.0f};
    
    while (stack_ptr > 0) {
        // Pop node from stack
        StackEntry current = stack[--stack_ptr];
        int32_t node_idx = current.node_idx;
        
        // Skip if we already found something closer
        if (current.t_min >= closest_t) {
            continue;
        }
        
        // Test AABB intersection
        if (!aabb_intersect(ray_origin, ray_direction, 
                          bvh.node_mins[node_idx], bvh.node_maxs[node_idx])) {
            continue;
        }
        
        // Leaf node - test triangles
        if (bvh.node_lefts[node_idx] == -1) {
            for (int32_t i = bvh.node_starts[node_idx]; i < bvh.node_ends[node_idx]; i++) {
                float t;
                Vec3 normal;
                if (triangle_intersect(triangles[i * 3], triangles[i * 3 + 1], triangles[i * 3 + 2],
                                     ray_origin, ray_direction, t, normal)) {
                    if (t < closest_t) {
                        closest_t = t;
                        closest_normal = normal;
                        closest_color = colors[i];
                        hit_anything = true;
                    }
                }
            }
            continue;
        }
        
        // Internal node - push children
        int32_t left_idx = bvh.node_lefts[node_idx];
        int32_t right_idx = bvh.node_rights[node_idx];
        
        // Calculate distances to child AABBs for traversal order
        float t_left = aabb_intersection_distance(ray_origin, ray_direction, 
                                                bvh.node_mins[left_idx], 
                                                bvh.node_maxs[left_idx]);
        float t_right = aabb_intersection_distance(ray_origin, ray_direction, 
                                                 bvh.node_mins[right_idx], 
                                                 bvh.node_maxs[right_idx]);
        
        // Push nodes in far-to-near order (stack is LIFO)
        if (t_left < t_right) {
            if (t_right < CUDART_INF_F && stack_ptr < MAX_STACK) {
                stack[stack_ptr++] = {right_idx, t_right};
            }
            if (t_left < CUDART_INF_F && stack_ptr < MAX_STACK) {
                stack[stack_ptr++] = {left_idx, t_left};
            }
        } else {
            if (t_left < CUDART_INF_F && stack_ptr < MAX_STACK) {
                stack[stack_ptr++] = {left_idx, t_left};
            }
            if (t_right < CUDART_INF_F && stack_ptr < MAX_STACK) {
                stack[stack_ptr++] = {right_idx, t_right};
            }
        }
    }
    
    return hit_anything;
}


__global__ void cast_rays_kernel(
    Vec3* ray_origins,
    Vec3* ray_directions,
    Vec3* triangles,
    Vec3* colors,
    BVHData bvh,
    int depth,
    float* pixel_seeds,
    Vec3* direct_colors,
    bool* did_hits,
    Vec3* hit_points,
    Vec3* bounce_dirs,
    size_t n_rays
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rays) return;

    Vec3 ray_origin = ray_origins[idx];
    Vec3 ray_direction = ray_directions[idx];
    float pixel_seed = pixel_seeds[idx];
    
    float t;
    Vec3 normal, color;
    bool hit = intersect_bvh_node(ray_origin, ray_direction, triangles, colors,
                                 bvh, 0, t, normal, color);

    if (!hit) {
        // Sky color
        //direct_colors[idx] = Vec3(0.2f, 0.3f, 0.5f);
        float ray_length = sqrtf(dot(ray_direction, ray_direction));
        float az = ray_direction.y / ray_length;
        direct_colors[idx] = Vec3(az, az, 1.0);

        did_hits[idx] = false;
        hit_points[idx] = Vec3();
        bounce_dirs[idx] = Vec3();
        return;
    }
    
    // Compute hit point and random bounce direction
    Vec3 hit_point = ray_origin + ray_direction * t;
    
    // Create random bounce direction in hemisphere around normal
    Vec3 random_dir = random_in_unit_sphere(pixel_seed + depth * 1.6180339887f);
    if (dot(random_dir, normal) < 0) {
        random_dir = Vec3(-random_dir.x, -random_dir.y, -random_dir.z);
    }
    Vec3 bounce_dir = normalize(Vec3(
        normal.x + random_dir.x,
        normal.y + random_dir.y,
        normal.z + random_dir.z
    ));
    
    // Direct lighting
    Vec3 light_dir = normalize(Vec3(1.0f, 1.0f, 1.0f));
    float diffuse = fmaxf(0.0f, dot(normal, light_dir));
    Vec3 direct_color = Vec3(
        color.x * (0.2f + 0.8f * diffuse),
        color.y * (0.2f + 0.8f * diffuse),
        color.z * (0.2f + 0.8f * diffuse)
    );
    
    direct_colors[idx] = direct_color;
    did_hits[idx] = true;
    hit_points[idx] = hit_point;
    bounce_dirs[idx] = bounce_dir;
}

// Host function to launch the kernel
extern "C" void launchCastRays(
    Vec3* ray_origins,
    Vec3* ray_directions,
    Vec3* triangles,
    Vec3* colors,
    BVHData bvh,
    int depth,
    float* pixel_seeds,
    Vec3* direct_colors,
    bool* did_hits,
    Vec3* hit_points,
    Vec3* bounce_dirs,
    size_t n_rays,
    size_t n_triangles
) {
    int threads_per_block = 256;
    int blocks = (n_rays + threads_per_block - 1) / threads_per_block;
    
    Vec3* d_ray_origins;
    Vec3* d_ray_directions;
    Vec3* d_triangles;
    Vec3* d_colors;
    Vec3* d_direct_colors;
    bool* d_did_hits;
    Vec3* d_hit_points;
    Vec3* d_bounce_dirs;
    float* d_pixel_seeds;

    Vec3* node_mins;
    Vec3* node_maxs;
    int32_t* node_lefts;
    int32_t* node_rights;
    int32_t* node_starts;
    int32_t* node_ends;

    // Allocate memory 
    cudaMalloc(&d_ray_origins, n_rays * sizeof(Vec3));
    cudaMalloc(&d_ray_directions, n_rays * sizeof(Vec3));
    cudaMalloc(&d_triangles, 3 * n_triangles * sizeof(Vec3));
    cudaMalloc(&d_colors, n_triangles * sizeof(Vec3));
    cudaMalloc(&d_direct_colors, n_rays * sizeof(Vec3));
    cudaMalloc(&d_did_hits, n_rays * sizeof(bool));
    cudaMalloc(&d_hit_points, n_rays * sizeof(Vec3));
    cudaMalloc(&d_bounce_dirs, n_rays * sizeof(Vec3));
    cudaMalloc(&d_pixel_seeds, n_rays * sizeof(float));

    cudaMalloc(&node_mins, bvh.n_nodes * sizeof(Vec3));
    cudaMalloc(&node_maxs, bvh.n_nodes * sizeof(Vec3));
    cudaMalloc(&node_lefts, bvh.n_nodes * sizeof(int32_t));
    cudaMalloc(&node_rights, bvh.n_nodes * sizeof(int32_t));
    cudaMalloc(&node_starts, bvh.n_nodes * sizeof(int32_t));
    cudaMalloc(&node_ends, bvh.n_nodes * sizeof(int32_t));

    // Copy data to device
    cudaMemcpy(d_ray_origins, ray_origins, n_rays * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_directions, ray_directions, n_rays * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, triangles, 3 * n_triangles * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, colors, n_triangles * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixel_seeds, pixel_seeds, n_rays * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(node_mins, bvh.node_mins, bvh.n_nodes * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(node_maxs, bvh.node_maxs, bvh.n_nodes * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(node_lefts, bvh.node_lefts, bvh.n_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(node_rights, bvh.node_rights, bvh.n_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(node_starts, bvh.node_starts, bvh.n_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(node_ends, bvh.node_ends, bvh.n_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);

    BVHData d_bvh = {
        bvh.n_nodes,
        node_mins,
        node_maxs,
        node_lefts,
        node_rights,
        node_starts,
        node_ends
    };

    cast_rays_kernel<<<blocks, threads_per_block>>>(
        d_ray_origins, d_ray_directions, d_triangles, d_colors,
        d_bvh, depth, d_pixel_seeds, d_direct_colors,
        d_did_hits, d_hit_points, d_bounce_dirs, n_rays
    );

    cudaDeviceSynchronize();

    // If there are any errors, print them to stdout
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    } else {
        // Copy result back to host
        cudaMemcpy(direct_colors, d_direct_colors, n_rays * sizeof(Vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(did_hits, d_did_hits, n_rays * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(hit_points, d_hit_points, n_rays * sizeof(Vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(bounce_dirs, d_bounce_dirs, n_rays * sizeof(Vec3), cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(d_ray_origins);
    cudaFree(d_ray_directions);
    cudaFree(d_triangles);
    cudaFree(d_colors);
    cudaFree(d_direct_colors);
    cudaFree(d_did_hits);
    cudaFree(d_hit_points);
    cudaFree(d_bounce_dirs);
    cudaFree(d_pixel_seeds);

    cudaFree(node_mins);
    cudaFree(node_maxs);
    cudaFree(node_lefts);
    cudaFree(node_rights);
    cudaFree(node_starts);
}