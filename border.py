import meshoperations as mesh
import numpy as np
import math
import actions

def prewit(matrix):
    m = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    return mesh.apply_double_mesh(matrix, m)

def sobel(matrix):
    m = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return mesh.apply_double_mesh(matrix, m)


def multi_prewit(matrix):
    m1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    out1 = mesh.apply_mesh_one_dimension(matrix, m1, 3)
    m2 = m1.transpose()
    out2 = mesh.apply_mesh_one_dimension(matrix, m2, 3)
    m3 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    out3 = mesh.apply_mesh_one_dimension(matrix, m3, 3)
    m4 = m3.transpose()
    out4 = mesh.apply_mesh_one_dimension(matrix, m4, 3)
    out = np.zeros(matrix.shape, dtype=np.int16)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            out[i, j] = max(abs(out1[i, j]), abs(out2[i, j]), abs(out3[i, j]), abs(out4[i, j]))

    return actions.linear_transform(out).astype(np.uint8)


def multi_prewit(matrix):
    m1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    out1 = mesh.apply_mesh_one_dimension(matrix, m1, 3)
    m2 = m1.transpose()
    out2 = mesh.apply_mesh_one_dimension(matrix, m2, 3)
    m3 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    out3 = mesh.apply_mesh_one_dimension(matrix, m3, 3)
    m4 = m3.transpose()
    out4 = mesh.apply_mesh_one_dimension(matrix, m4, 3)
    out = np.zeros(matrix.shape, dtype=np.int16)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            out[i, j] = max(abs(out1[i, j]), abs(out2[i, j]), abs(out3[i, j]), abs(out4[i, j]))

    return actions.linear_transform(out).astype(np.uint8)


def multi_sobel(matrix):
    m1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    out1 = mesh.apply_mesh_one_dimension(matrix, m1, 3)
    m2 = m1.transpose()
    out2 = mesh.apply_mesh_one_dimension(matrix, m2, 3)
    m3 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    out3 = mesh.apply_mesh_one_dimension(matrix, m3, 3)
    m4 = m3.transpose()
    out4 = mesh.apply_mesh_one_dimension(matrix, m4, 3)
    out = np.zeros(matrix.shape, dtype=np.int16)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            out[i, j] = max(abs(out1[i, j]), abs(out2[i, j]), abs(out3[i, j]), abs(out4[i, j]))

    return actions.linear_transform(out).astype(np.uint8)


def laplace(matrix,threshold):
    m = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    aux = mesh.apply_mesh_one_dimension(matrix, m, 3)
    out = np.zeros(matrix.shape,dtype=np.uint8)
    for i in range(1,matrix.shape[0]):
        for j in range(1,matrix.shape[1]):
            if np.sign(aux[i-1, j])*np.sign(aux[i, j]) < 0:
                out[i, j] = 255 if abs(aux[i-1, j]) + abs(aux[i, j]) >= threshold else 0
            if np.sign(aux[i, j-1])*np.sign(aux[i, j]) < 0 and out[i, j] != 255:
                out[i, j] = 255 if abs(aux[i, j - 1]) + abs(aux[i, j]) >= threshold else 0

    return out


def intelligent_laplace(matrix):
    # threshold = 40
    threshold = np.mean(matrix)
    return laplace(matrix, threshold)

def laplace_gauss(matrix):
    std = 1
    size = 5
    m = np.zeros((size, size))
    radius = int(size / 2)
    cst = -1 / (math.sqrt(2 * math.pi) * std * std * std)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            aux = 2 - (i*i + j*j) / (std*std)
            m[i + radius, j + radius] = cst * aux * math.exp(-(j * j + i * i) / (2 * std * std))
    ma = mesh.apply_mesh_one_dimension(matrix, m, size)
    return actions.linear_transform(ma).astype(np.uint8)