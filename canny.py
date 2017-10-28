from tkinter import Tk, Toplevel, Scale, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage, LEFT
from PIL import Image, ImageTk
import numpy as np
import canny as canny
import random
import myrandom as myrand
import meshoperations as mesh

#http://www.vision.uji.es/courses/Doctorado/FVC/FVC-T5-DetBordes-Parte1-4p.pdf

"""
Funciones auxiliares
"""
def get_img_type(self):
    return get_img_type_from_canvas(self, self.canvas[0])

def get_img_type_from_canvas(self, canv):
    true_img = canv.true_image
    pixels = true_img.load()

    type = 'RGB'
    try:
        len(pixels[0, 0])
    except TypeError:
        type = 'L'

    return type

def linear_transform(matrix):
    return np.interp(matrix, [np.min(matrix),np.max(matrix)], [0,255]).astype(np.uint8)

def matrix_to_window(self, out, title, type):
    height = out.shape[0]
    width = out.shape[1]

    self.result_window = Toplevel()
    self.result_window.minsize(width=width, height=height)
    self.result_window.title(title)
    canvas_result = Canvas(self.result_window, height=height, width=width)

    # img = Image.fromarray(np.array(out))
    # img = Image.fromarray(out, mode=type)
    out = linear_transform(out)
    img = Image.fromarray(np.array(out, dtype=np.int16))

    photo = ImageTk.PhotoImage(img)
    canvas_result.image = photo
    canvas_result.true_image = img
    canvas_result.configure(width=width, height=height)
    canvas_result.create_image((0, 0), image=photo, anchor='nw')
    canvas_result.grid(row=0,column=0)

    menu = Menu(self.result_window)
    self.result_window.config(menu=menu)
    filemenu = Menu(menu, tearoff=0)
    menu.add_cascade(label="File", menu=filemenu)
    filemenu.add_command(label="Save", command=lambda: save(self.result_window,canvas_result))
    filemenu.add_command(label="Load on canvas", command=lambda: to_main_canvas(self, canvas_result))
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=lambda: self.result_window.quit)

"""
Fin de funciones auxiliares
"""

#Porcentaje y otros datos hardcodeados
def canny_function(self):
    pixels = self.true_image.load()
    height, width = self.true_image.size
    img_arr = np.array(self.true_image, dtype=np.int16)

    type = get_img_type(self)
    len = 1
    if (type == 'RGB'):
        len = 3

    #Deshardcodear esto!
    percentage = 25
    mu = 10
    sigma = 0

    #Matrices auxiliares
    img_gauss = np.zeros((width, height, len), dtype=np.int16)
    phis = np.zeros((width, height, len), dtype=np.cfloat)
    img_2 = np.zeros((width, height, len), dtype=np.int16)
    final = np.zeros((width, height, len), dtype=np.int16)

    #1 - aplicar GAUSS

    img_gauss = mesh.gauss_filter(img_arr, 5, 1);

    m = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]); #Mascara de Sobel
    mp = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]); #Mascara de Prewit

    #PHIS es la matriz de arctan de Dy/Dx
    phis = apply_double_mesh_one_dimension_atan(img_gauss, m)

    img_2 = supr_no_max(phis, img_gauss)

    final = umbral_histeresis(img_2, width, height)

    matrix_to_window(self, final, "Resultado final", 'L')

def supr_no_max(matrix_directions, matrix_original):
    w, h = matrix_original.shape #buscar
    to_ret = np.zeros((w, h, 1), dtype=np.int16)

    for i in range(w):
        for j in range(h):
            if ( i != 0 or i != w-1 or j != 0 or j != h-1 ): #casos de los bordes
                to_ret[i,j] = get_value(matrix_original, matrix_directions[i,j], i, j, w, h)

    return to_ret

def get_value(matrix_original, phi, i, j, w, h):
    angle = get_area(phi)
    return get_val_from_neigh(matrix_original, get_neigh(angle), i, j, w, h)

def get_area(phi):
    if (phi >= 0 and phi < 22.5) or (phi >= 157.5 and phi < 180):
        return 0
    if phi >= 22.5 and phi < 67.5:
        return 45
    if phi >= 67.5 and phi < 112.5:
        return 90
    if phi >= 112.5 and phi < 157.5:
        return 135

def get_neigh(angle):
    neighbours = np.zeros((2,2))
    if angle == 0:
        neighbours[0] = [-1,0]
        neighbours[1] = [1,0]
    if angle == 45:
        neighbours[0] = [1,-1]
        neighbours[1] = [-1,1]
    if angle == 90:
        neighbours[0] = [0,-1]
        neighbours[1] = [0,1]
    if angle == 135:
        neighbours[0] = [-1,-1]
        neighbours[1] = [1,1]

    return neighbours

def get_val_from_neigh(matrix_original, neighbours, i, j, w, h):
    val = np.zeros(2)
    me = matrix_original[i,j]

    if( (i + int(neighbours[0][0])) >= 0 and (i + int(neighbours[0][0])) < w and (j + int(neighbours[0][1])) >= 0 and (j + int(neighbours[0][1])) < h):
        val[0] = matrix_original[ i + int(neighbours[0][0]) , j + int(neighbours[0][1]) ]
    else:
        val[0] = 0

    if( (i + int(neighbours[1][0])) >= 0 and (i + int(neighbours[1][0])) < w and (j + int(neighbours[1][1])) >= 0 and (j + int(neighbours[1][1])) < h):
        val[1] = matrix_original[ i + int(neighbours[1][0]) , j + int(neighbours[1][1]) ]
    else:
        val[1] = 0

    if( val[0] >= me or val[1] >= me ):
        return 0 #no soy borde
    else:
        return me

def umbral_histeresis(img, w, h):

    #La idea es hacerlo con Otsu, pero tambien tengo anotado que lo podemos hacer con un slider!!
    #lo dejo hardcodeado porque no se como es tu implementación, Lucas...
    #el umbral segun otzu es 101
    t1 = 99
    t2 = 101

    to_ret = np.zeros((w, h), dtype=np.int16)

    for i in range(w):
        for j in range(h):
            if ( img[i,j] > t2 ):
                to_ret[i,j] = 255
            if ( img[i,j] < t1 ):
                to_ret[i,j] = 0
            if ( img[i,j] <= t2 and img[i,j] >= t1 ):
                to_ret[i,j] = analize_4_neigh(img, t1, t2, w, h, i, j)

    return to_ret

def analize_4_neigh(img, t1, t2, w, h, i, j):
    vec = np.zeros((4,2))
    vec[0] = [ -1, 0 ]
    vec[1] = [ 0, -1 ]
    vec[2] = [ 1, 0 ]
    vec[3] = [ 0, 1 ]

    for k in range(4):
        if ( i + vec[k][0] >= 0 and i + vec[k][0] < w):
            if ( j + vec[k][1] >= 0 and j + vec[k][1] < h ):
                n = img[i + int(vec[k][0])][j + int(vec[k][1])]
                #Los pixels cuya magnitud de borde está entre t1 y t2 y están conectados con un borde, se marcan también como borde
                if ( n == 255 ):
                    return 255
    return 0

def apply_double_mesh_one_dimension_atan(matrix, mesh):
    mesh_trans = mesh.transpose()
    out = np.zeros(matrix.shape, dtype=np.cfloat)
    radius = 1
    shape = matrix.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i >= shape[0] - radius or i < radius or j < radius or j >= shape[1] - radius:
                out[i, j] = matrix[i, j]
            else:
                dx = int(np.sum(mesh * matrix[i - radius:i + radius + 1, j - radius:j + radius + 1]))
                dy = int(np.sum(mesh_trans * matrix[i - radius:i + radius + 1, j - radius:j + radius + 1]))
                if(dx == 0):
                    out[i, j] = 0
                else:
                    out[i, j] = np.arctan(dy/dx)
    return out
