import numpy as np

def parse_obj_file(content):
    vertices = []
    faces = []
    
    for line in content.split('\n'):
        if line.startswith('v '):
            # Parse vertex coordinates
            _, x, y, z = line.split()
            vertices.append([float(x), float(y), float(z)])
        elif line.startswith('f '):
            # Parse face indices (subtract 1 because OBJ indices start at 1)
            _, v1, v2, v3 = line.split()
            faces.append([int(v1)-1, int(v2)-1, int(v3)-1])
            
    return np.array(vertices), np.array(faces)

def obj_to_triangles(objfile):
    # Parse both OBJ files
    verts, faces = parse_obj_file(objfile)
    
    # Initialize empty list for all triangles
    tris = []
    
    # Convert ground faces to triangles
    for face in faces:
        triangle = [
            verts[face[0]],
            verts[face[1]],
            verts[face[2]]
        ]
        tris.append(triangle)
    
    return np.array(tris)