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

    # 1 - aplicar GAUSS

    img_gauss = mesh.gauss_filter(img_arr, 7, 1) #podria ser 2 tambien

    m = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #Mascara de Sobel
    # mp = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]); #Mascara de Prewit

    # directions es la matrix de direcciones
    sobel_matrix = bd.sobel(img_gauss)
    sobel_gauss_matrix = bd.sobel(img_gauss)

    directions = apply_double_mesh_one_dimension_atan(sobel_matrix, m) # Ahora mesh ya devuelve las direcciones
    directions_gauss = apply_double_mesh_one_dimension_atan(sobel_gauss_matrix, m)  # Ahora mesh ya devuelve las direcciones

    img_2 = supr_no_max(directions, sobel_matrix)
    img_2_gauss = supr_no_max(directions_gauss, sobel_gauss_matrix)

    final = umbral_histeresis(img_2, np.std(img_2))
    final_gauss = umbral_histeresis(img_2_gauss, np.std(img_2_gauss))

    return np.minimum(final, final_gauss, dtype=np.uint8)


def supr_no_max(matrix_directions, matrix_original):
    h, w = matrix_original.shape #buscar

    for i in range(1, h-1):
        for j in range(1, w-1):
            ad = tuple(np.add((i, j), matrix_directions[i, j]))
            sb = tuple(np.subtract((i, j), matrix_directions[i, j]))
            if matrix_original[i, j] <= matrix_original[ad] or matrix_original[i, j] <= matrix_original[sb]:
                matrix_original[i, j] = 0

    return matrix_original

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

def umbral_histeresis(img, std):
    h, w = img.shape
    me = th.otsu_threshold(img)
    print(std)
    t1 = me - std/2
    t2 = me + std/2

    to_ret = np.zeros(img.shape, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > t2:
                to_ret[i, j] = 255
            elif img[i, j] < t1:
                to_ret[i, j] = 0
            else:
                to_ret[i, j] = analize_4_neigh(img, t1, t2, img.shape[0], img.shape[1], i, j)

    return to_ret


def analize_4_neigh(img, t1, t2, w, h, i, j):

    for k in vec:
        if 0 <= i + k[0] < w and 0 <= j + k[1] < h:
            n = img[i + k[0], j + k[1]]
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
                if dx == 0 and dy == 0:
                    out[i, j] = 0, 0
                elif dx == 0:
                    out[i, j] = 0, 1
                else:
                    out[i, j] = get_area(np.arctan(dy/dx) + (np.pi / 2))
    return out
