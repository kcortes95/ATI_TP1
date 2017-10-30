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

def calculate_s(img, threshold, i, j, width, height):
    sum = 0;

    for k in range(7):
        for l in range(7):
            posx = k + i
            posy = l + j
            if(posx >= 0 and posx < width and posy >= 0 and posy < height):
                #img[posx + 3][posy + 3] -> es el valor del centro
                if( math.fabs(img[posx][posy] - img[posx + 3][posy + 3]) < threshold):
                    sum += 1

    return 1 - sum/49 #49 es la cantidad de elementos en la máscara

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
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i >= shape[0] - radius or i < radius or j < radius or j >= shape[1] - radius:
                out[i, j] = matrix[i, j]
            else:
                if( math.fabs(calculate_s(matrix, threshold, i, j, width, height) - 0.5) < 0.10 ):
                    #es border
                    out[i, j] = 255;
                if( math.fabs(calculate_s(matrix, threshold, i, j, width, height) - 0.75) < 0.10 ):
                    #es esquina
                    out[i, j] = 255;

    return out
