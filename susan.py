from tkinter import Tk, Toplevel, Scale, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage, LEFT
from PIL import Image, ImageTk
import numpy as np
import border as bd
import thresholds as th
import meshoperations as mesh
import math as math
import actions as actions

def susan_function(matrix):
    img_arr = matrix
    width, height = matrix.shape
    img_aux = np.zeros((width, height, 1), dtype=np.uint8)

    circular_mask = [[0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0]]

    img_aux = apply_mesh(img_arr, circular_mask, 7, 27)


def apply_mesh(matrix, mesh, size, threshold):
    shape = matrix.shape
    print(shape)
    if len(shape) > 2:
        out = np.zeros(shape,dtype=np.int16)
        for i in range(shape[2]):
            out[:, :, i] = apply_mesh_one_dimension(matrix[:, :, i], mesh, size)
        return actions.linear_transform(out).astype(np.uint8)
    else:
        return actions.linear_transform(apply_mesh_one_dimension(matrix,mesh,size, threshold)).astype(np.uint8)

def apply_mesh_one_dimension(matrix, mesh, size, threshold):
    print("SUSAAAN")

    width, height = matrix.shape
    out = np.zeros(matrix.shape, dtype=np.float32)
    radius = int(size / 2)
    shape = matrix.shape
    print("radius: " + str(radius))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i >= shape[0] - radius or i < radius or j < radius or j >= shape[1] - radius:
                out[i, j] = matrix[i, j]
            else:
                out[i][j] = calculate(matrix, i, j, width, height, threshold)

    return out


def calculate(matrix, i, j, width, height, threshold):
    sum = 0
    central_pixel = matrix[i][j]

    for k in range(-3,4):
        for l in range(-3,4):
            posx = i + k
            posy = j + l

            if (posx >= 0 or posx < width or posy >= 0 or posy < height):
                if(matrix[posx][posy] * mesh[k + 3][l + 3] != 0):
                    if( math.fabs(matrix[posx][posy] - central_pixel) < threshold ):
                        sum += 1

    tot = 1 - sum/49 #49 total en la máscara (CREO)

    if( math.fabs(tot - 0.5) < 0.1 ):
        return 255

    if( math.fabs(tot - 0.75) < 0.1 ):
        return 255

    return 0
