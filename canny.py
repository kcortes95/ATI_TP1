from tkinter import Tk, Toplevel, Scale, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage, LEFT
from PIL import Image, ImageTk
import numpy as np
import border as bd
import thresholds as th
import meshoperations as mesh

#http://www.vision.uji.es/courses/Doctorado/FVC/FVC-T5-DetBordes-Parte1-4p.pdf

vec = np.zeros((4, 2), dtype=np.int8)  # Esto es mas ineficiente.....
vec[0] = (-1, 0)
vec[1] = (0, -1)
vec[2] = (1, 0)
vec[3] = (0, 1)

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

# Porcentaje y otros datos hardcodeados
def canny_function(matrix):
    img_arr = matrix

    depth = 1
    if len(matrix.shape) == 3:
        depth = 3

    # Deshardcodear esto!
    percentage = 25
    mu = 10
    sigma = 0

    # Matrices auxiliares
    img_gauss = np.zeros(matrix.shape, dtype=np.int16)
    phis = np.zeros(matrix.shape, dtype=np.cfloat)
    img_2 = np.zeros(matrix.shape, dtype=np.int16)
    final = np.zeros(matrix.shape, dtype=np.int16)

    # 1 - aplicar GAUSS

    img_gauss = mesh.gauss_filter(img_arr, 7, 1)

    m = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #Mascara de Sobel
    # mp = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]); #Mascara de Prewit

    # directions es la matrix de direcciones
    sobel_matrix = bd.sobel(img_arr)
    sobel_gauss_matrix = bd.sobel(img_gauss)

    directions = apply_double_mesh_one_dimension_atan(img_arr, m) # Ahora mesh ya devuelve las direcciones
    directions_gauss = apply_double_mesh_one_dimension_atan(img_gauss, m)  # Ahora mesh ya devuelve las direcciones

    img_2 = supr_no_max(directions, sobel_matrix)
    img_2_gauss = supr_no_max(directions_gauss, sobel_gauss_matrix)

    final = umbral_histeresis(img_2, np.std(sobel_matrix))
    final_gauss = umbral_histeresis(img_2_gauss, np.std(sobel_gauss_matrix))

    return np.minimum(final, final_gauss, dtype=np.uint8)


def supr_no_max(matrix_directions, matrix_original):
    h, w = matrix_original.shape #buscar

    for i in range(1, h-1):  # Asi saco los bordes sin el if
        for j in range(1, w-1):
            ad = tuple(np.add((i, j), matrix_directions[i, j]))
            sb = tuple(np.subtract((i, j), matrix_directions[i, j]))
            if (matrix_original[i, j] <= matrix_original[ad]
                    or matrix_original[i, j] <= matrix_original[sb]):
                matrix_original[i, j] = 0

    return matrix_original


def get_value(matrix_original, phi, i, j, w, h):
    angle = get_area(phi)
    return get_val_from_neigh(matrix_original, get_neigh(angle), i, j, w, h)


def get_area(phi):
    if phi < 0 or phi > np.pi:
        print("ERROR " + str(phi))

    if phi < 0.3926991 or phi > 2.7488936:
        return 0, 1
    elif phi < 1.178097:
        return -1, 1
    elif phi < 1.9634954:
        return -1, 0
    else:
        return -1, -1


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


def umbral_histeresis(img, std):
    h, w = img.shape
    #La idea es hacerlo con Otsu, pero tambien tengo anotado que lo podemos hacer con un slider!!
    #lo dejo hardcodeado porque no se como es tu implementación, Lucas...
    #el umbral segun otzu es 101
    me = th.otsu_threshold(img)
    print(std)
    t1 = me - std/2
    t2 = me + std/2

    to_ret = np.zeros((w, h), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > t2:
                to_ret[i, j] = 255
            elif img[i, j] < t1:
                to_ret[i, j] = 0
            else:
                to_ret[i, j] = analize_4_neigh(img, t1, t2, w, h, i, j)

    return to_ret


def analize_4_neigh(img, t1, t2, w, h, i, j):

    for k in vec:
        if 0 <= i + k[0] < w and 0 <= j + k[1] < h:
            n = img[i + k[0], j + k[1]]
            #Los pixels cuya magnitud de borde está entre t1 y t2 y están conectados con un borde, se marcan también como borde
            if n >= t2:
                return 255
    return 0


def apply_double_mesh_one_dimension_atan(matrix, mesh):
    mesh_trans = mesh.transpose()
    out = np.zeros(np.append(matrix.shape, [2]), dtype=np.int8)  # La nueva dimension es por la direccion
    radius = 1
    shape = matrix.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i >= shape[0] - radius or i < radius or j < radius or j >= shape[1] - radius:
                out[i, j] = matrix[i, j]
            else:
                dx = int(np.sum(mesh * matrix[i - radius:i + radius + 1, j - radius:j + radius + 1]))
                dy = int(np.sum(mesh_trans * matrix[i - radius:i + radius + 1, j - radius:j + radius + 1]))
                if dx == 0:
                    out[i, j] = 0, 1
                else:
                    out[i, j] = get_area(np.arctan(dy/dx) + (np.pi / 2))
    return out
