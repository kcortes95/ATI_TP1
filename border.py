import meshoperations as mesh
import numpy as np
import scipy as sp
import math
import actions
import thresholds
from scipy import signal

sobel_matrix = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
def prewit(matrix):
    m = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    return mesh.apply_double_mesh(matrix, m)


def sobel(matrix):
    return mesh.apply_double_mesh(matrix, sobel_matrix)


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


def laplace(matrix, threshold=0):
    m = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    aux = mesh.apply_mesh_one_dimension(matrix, m, 3)
    return zero_cross(aux, threshold)


def zero_cross(aux, threshold=0):
    out = np.zeros(aux.shape, dtype=np.uint8)
    for i in range(1,aux.shape[0]-1):
        for j in range(1,aux.shape[1]-1):
            if np.sign(aux[i-1, j])*np.sign(aux[i, j]) < 0:
                out[i, j] = 255 if abs(aux[i-1, j]) + abs(aux[i, j]) >= threshold else 0
            if np.sign(aux[i, j-1])*np.sign(aux[i, j]) < 0 and out[i, j] != 255:
                out[i, j] = 255 if abs(aux[i, j - 1]) + abs(aux[i, j]) >= threshold else 0
    return out


def intelligent_laplace(matrix):
    # threshold = 40
    threshold = np.mean(matrix)
    return laplace(matrix, threshold)


def laplace_gauss(matrix, std):
    size = 6*std + 1
    m = np.zeros((size, size))
    radius = int(size / 2)
    cst = -1 / (math.sqrt(2 * math.pi) * std * std * std)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            aux = 2 - (i*i + j*j) / (std*std)
            m[i + radius, j + radius] = cst * aux * math.exp(-(j * j + i * i) / (2 * std * std))

    print(m)
    print(sum(sum(m)))
    ma = signal.convolve2d(matrix, m, "same", "symm")
    return zero_cross(ma)


def hough(matrix, a, b, self):
    aux = sobel(matrix)
    aux = thresholds.threshold(aux, 128)
    D = max(aux.shape)
    p_range = 2 * math.sqrt(2) * D
    a_max = a
    b_max = b
    acumulator = np.zeros((a_max+1, b_max+1), dtype=np.uint32)
    e = 0.9
    cos_a = np.zeros(a_max + 1, dtype=np.float32)
    sin_a = np.zeros(a_max + 1, dtype=np.float32)
    for a_i in range(a_max+1):
        cos_a[a_i] = math.cos(-math.pi/2 + (a_i/a_max)*math.pi)
        sin_a[a_i] = math.sin(-math.pi/2 + (a_i/a_max)*math.pi)

    for y in range(aux.shape[0]):
        for x in range(aux.shape[1]):
            if aux[y, x] == 255:
                for a in range(a_max + 1):
                    for b in range(b_max + 1):
                        bi = -p_range/2 + (b/b_max)*p_range
                        if abs(bi - x*cos_a[a] - y*sin_a[a]) < e:
                            acumulator[a, b] += 1

    m = np.max(acumulator)
    points = get_tops(acumulator, m)
    np.set_printoptions(threshold=np.inf)
    for (a, b) in points:
        bi = -p_range / 2 + (b / b_max) * p_range
        if sin_a[a] == 0:
            self.canvas[0].create_line(bi, 0, bi, aux.shape[0], fill="red")
        else:
            self.canvas[0].create_line(0, bi/sin_a[a], aux.shape[1], bi/sin_a[a] - aux.shape[1]*cos_a[a]/sin_a[a], fill="red")
    return matrix


def get_tops(acumulator, m):
    thres = m*0.7
    res = []
    for a in range(acumulator.shape[0]):
        for b in range(acumulator.shape[1]):
            if acumulator[a, b] > thres:
                res.append((a, b))
    return res


def harris(matrix, threshold):
    if len(matrix.shape) == 3:
        matrix_to_op = actions.to_grayscale(matrix)
    else:
        matrix_to_op = matrix

    dx = sp.signal.convolve2d(matrix_to_op, sobel_matrix, boundary='symm', mode='same')
    dy = sp.signal.convolve2d(matrix_to_op, np.transpose(sobel_matrix), boundary='symm', mode='same')
    gm = mesh.gauss_mesh(7, 2)
    dx2 = sp.signal.convolve2d(np.power(dx, 2), gm, boundary='symm', mode='same')
    dy2 = sp.signal.convolve2d(np.power(dy, 2), gm, boundary='symm', mode='same')
    lxy = sp.signal.convolve2d(np.multiply(dx, dy), gm, boundary='symm', mode='same')
    k = 0.04
    res = (np.multiply(dx2, dy2) - np.power(lxy, 2)) - k*np.power(np.add(dx2, dy2), 2)
    res = actions.linear_transform(res)
    return combine(matrix, thresholds.threshold(res, threshold))


def combine(original, modified):
    shape = original.shape
    res = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if len(original.shape) == 2:
        res[:, :, 0] = original
        res[:, :, 1] = original
        res[:, :, 2] = original
    else:
        res = original
    for i in range(modified.shape[0]):
        for j in range(modified.shape[1]):
            if modified[i, j] == 255:
                res[i, j, 0] = 255
                res[i, j, 1] = 0
                res[i, j, 2] = 0
    return res
