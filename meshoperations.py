import numpy as np


def apply_mesh(matrix, mesh,size):
    shape = matrix.shape
    print(shape)
    if len(shape) > 2:
        out = np.zeros(shape,dtype=np.uint8)
        for i in range(shape[2]):
            out[:, :, i] = apply_mesh_one_dimension(matrix[:, :, i], mesh, size)
        return out
    else:
        return apply_mesh_one_dimension(matrix,mesh,size)


def apply_mesh_one_dimension(matrix, mesh, size):
    out = np.zeros(matrix.shape, dtype=np.uint8)
    radius = int(size / 2)
    shape = matrix.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i >= shape[0] - radius or i < radius or j < radius or j >= shape[1] - radius:
                out[i, j] = matrix[i, j]
            else:
                out[i, j] = int(np.sum(mesh * matrix[i - radius:i + radius + 1, j - radius:j + radius + 1]))

    return out

def mean_filter(matrix, size):
    mesh = np.full((size, size), 1/(size*size), dtype=np.float32)
    return apply_mesh(matrix, mesh, size)
