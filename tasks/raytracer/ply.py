import numpy as np
from plyfile import PlyData, PlyElement

def load_ply_triangles(plyfile):
    plydata = PlyData.read(plyfile)
    vertices = np.array([[vertex[0], vertex[1], vertex[2]] for vertex in plydata['vertex']])
    faces = np.array([[face[0], face[1], face[2]] for face in plydata['face']['vertex_indices']])
    
    tris = []
    for face in faces:
        triangle = [
            vertices[face[0]],
            vertices[face[1]],
            vertices[face[2]]
        ]
        tris.append(triangle)
    
    return np.array(tris)