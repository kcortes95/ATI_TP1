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
                sum = 0
                central_pixel = matrix[i+radius][j+radius]
                for k in range(7):
                    for l in range(7):
                        if(matrix[i + k][j + l] * mesh[i + k][j + l] != 0):
                            if( math.fabs(matrix[i + k][j + l] - central_pixel) < threshold ):
                                sum += 1

                tot = 1 - sum/49 #49 total en la máscara

                if( math.fabs(tot - 0.5) < 0.1 ):
                    #borde
                    out[i][j] = 255

                if( math.fabs(tot - 0.75) < 0.1 )
                    #esquina
                    out[i][j] = 255

    return out
