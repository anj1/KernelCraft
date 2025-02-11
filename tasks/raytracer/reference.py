import numpy as np
import ctypes
from numba import jit
    
@jit(nopython=True)
def normalize(vector):
    norm = np.sqrt(np.sum(vector**2))
    if norm > 0:
        return vector / norm
    return vector


@jit(nopython=True)
def cross_product(a, b):
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


@jit(nopython=True)
def random_in_unit_sphere(seed):
    x = np.sin(seed * 78.233 + 0.5) * 0.5 + 0.5
    y = np.sin(seed * 127.432 + 1.3) * 0.5 + 0.5
    z = np.sin(seed * 213.835 + 2.7) * 0.5 + 0.5

    vec = np.array([x, y, z]) * 2.0 - 1.0
    return normalize(vec)


@jit(nopython=True)
def aabb_intersect(ray_origin, ray_direction, min_bound, max_bound):
    inv_dir = 1.0 / ray_direction
    t1 = (min_bound - ray_origin) * inv_dir
    t2 = (max_bound - ray_origin) * inv_dir

    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)

    enter = np.max(tmin)
    exit = np.min(tmax)

    return enter <= exit and exit > 0


@jit(nopython=True)
def triangle_intersect(v0, v1, v2, ray_origin, ray_direction):
    epsilon = 1e-6

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = cross_product(ray_direction, edge2)
    a = np.dot(edge1, h)

    if abs(a) < epsilon:
        return np.inf, np.zeros(3)

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return np.inf, np.zeros(3)

    q = cross_product(s, edge1)
    v = f * np.dot(ray_direction, q)

    if v < 0.0 or u + v > 1.0:
        return np.inf, np.zeros(3)

    t = f * np.dot(edge2, q)

    if t < epsilon:
        return np.inf, np.zeros(3)

    normal = normalize(cross_product(edge1, edge2))
    return t, normal


@jit(nopython=True)
def intersect_bvh_node(
    ray_origin,
    ray_direction,
    triangles,
    colors,
    node_mins,
    node_maxs,
    node_lefts,
    node_rights,
    node_starts,
    node_ends,
    node_idx,
):
    if not aabb_intersect(
        ray_origin, ray_direction, node_mins[node_idx], node_maxs[node_idx]
    ):
        return np.inf, np.zeros(3), np.zeros(3)

    if node_lefts[node_idx] == -1:
        closest_t = np.inf
        closest_normal = np.zeros(3)
        closest_color = np.zeros(3)

        for i in range(node_starts[node_idx], node_ends[node_idx]):
            v0, v1, v2 = triangles[3*i, :], triangles[3*i+1, :], triangles[3*i+2, :]
            ro = ray_origin
            rd = ray_direction
            t, normal = triangle_intersect(v0, v1, v2, ro, rd)

            if t < closest_t:
                closest_t = t
                closest_normal = normal
                closest_color = colors[i]

        return closest_t, closest_normal, closest_color

    t1, n1, c1 = intersect_bvh_node(
        ray_origin,
        ray_direction,
        triangles,
        colors,
        node_mins,
        node_maxs,
        node_lefts,
        node_rights,
        node_starts,
        node_ends,
        node_lefts[node_idx],
    )

    t2, n2, c2 = intersect_bvh_node(
        ray_origin,
        ray_direction,
        triangles,
        colors,
        node_mins,
        node_maxs,
        node_lefts,
        node_rights,
        node_starts,
        node_ends,
        node_rights[node_idx],
    )

    if t1 < t2:
        return t1, n1, c1
    return t2, n2, c2


@jit(nopython=True)
def cast_ray(ray_origin, ray_direction, triangles, colors, bvh_data, depth, pixel_seed):
    ndmins, ndmaxs, ndlefts, ndrights, ndstarts, ndends = bvh_data

    # Primary intersection
    t, normal, color = intersect_bvh_node(
        ray_origin,
        ray_direction,
        triangles,
        colors,
        ndmins,
        ndmaxs,
        ndlefts,
        ndrights,
        ndstarts,
        ndends,
        0,
    )

    hit = t < np.inf
    if not hit:
        # Sky color based on direction
        azimuth = ray_direction[1] / np.linalg.norm(ray_direction)
        return np.array([azimuth, azimuth, 1.0]), False, np.zeros(3), np.zeros(3)

    # Compute hit point and random bounce direction
    hit_point = ray_origin + t * ray_direction

    # Create random bounce direction in hemisphere around normal
    random_dir = random_in_unit_sphere(pixel_seed + depth * 1.6180339887)
    if np.dot(random_dir, normal) < 0:
        random_dir = -random_dir
    bounce_dir = normalize(normal + random_dir)

    # Direct lighting
    light_dir = normalize(np.array([1.0, 1.0, 1.0]))
    diffuse = max(0, np.dot(normal, light_dir))
    direct_color = color * (0.2 + 0.8 * diffuse)

    return direct_color, True, hit_point, bounce_dir


@jit(nopython=True)
def cast_rays(
    ray_origins, ray_directions, triangles, colors, bvh_data, depth, pixel_seeds
):
    n_rays = ray_origins.shape[0]
    dcolors = np.zeros((n_rays, 3))
    did_hits = np.zeros(n_rays, dtype=np.bool_)
    hit_points = np.zeros((n_rays, 3))
    bounce_dirs = np.zeros((n_rays, 3))

    for i in range(n_rays):
        dcolors[i], did_hits[i], hit_points[i], bounce_dirs[i] = cast_ray(
            ray_origins[i],
            ray_directions[i],
            triangles,
            colors,
            bvh_data,
            depth,
            pixel_seeds[i],
        )

    return dcolors, did_hits, hit_points, bounce_dirs