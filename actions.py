from tkinter import Tk, Toplevel, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage, LEFT
from PIL import Image, ImageTk
import numpy as np
import math
import json
import histoperations as hist
MAX_HEIGHT = 1024
MAX_WIDTH = 1024


def set_color(self):
    color = int(self.color_text.get())
    set_pixel(self,color, int(self.x_text.get()), int(self.y_text.get()))


def set_pixel(self, color, x, y):
    self.canvas.true_image.putpixel((int(x), int(y)), color)
    self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)


def crop(self, master):
    self.new_window = Toplevel()
    self.new_window.minsize(width=640, height=480)

    button = Button(self.new_window, text="Guardar", command=lambda: save_cropped(self, master))
    button.pack()
    img = self.canvas.true_image.load()

    xstart = self.x_start if self.x_start < self.x_finish else self.x_finish
    ystart = self.y_start if self.y_start < self.y_finish else self.y_finish
    xfinish = self.x_start if self.x_start > self.x_finish else self.x_finish
    yfinish = self.y_start if self.y_start > self.y_finish else self.y_finish
    if isinstance(img[0, 0], tuple):
        new_image = np.zeros((yfinish - ystart, xfinish - xstart, len(img[0, 0])), dtype=np.uint8)
    else:
        if self.canvas.true_image.mode == 'F' :
            new_image = np.zeros((yfinish - ystart, xfinish - xstart), dtype=np.float32)
        else:
            new_image = np.zeros((yfinish - ystart, xfinish - xstart), dtype=np.uint8)

    x = 0
    y = 0
    for x_pos in range(xstart, xfinish):
        for y_pos in range(ystart, yfinish):
            aux = img[x_pos, y_pos]
            new_image[x, y] = aux
            x += 1
        y += 1
        x = 0
    true_cropped = Image.fromarray(new_image, self.canvas.true_image.mode)
    cropped = ImageTk.PhotoImage(true_cropped)
    self.new_window.canvas = Canvas(self.new_window, width=200, height=200)
    self.new_window.canvas.true_cropped = true_cropped
    self.new_window.canvas.cropped = cropped
    self.new_window.canvas.configure(width=true_cropped.width, height=true_cropped.height)
    self.new_window.canvas.create_image((0, 0), anchor="nw", image=cropped)
    self.new_window.canvas.pack()


def save_cropped(self, master):
    filename = filedialog.asksaveasfilename(parent=master)
    self.new_window.canvas.true_cropped.save(filename)
    print(filename)


def get_area_info(self, master):
    pixel_count = abs(self.x_start - self.x_finish) * abs(self.y_start - self.y_finish)
    print("Cantidad de pixeles:" + str(pixel_count))
    xstart = self.x_start if self.x_start < self.x_finish else self.x_finish
    ystart = self.y_start if self.y_start < self.y_finish else self.y_finish
    xfinish = self.x_start if self.x_start > self.x_finish else self.x_finish
    yfinish = self.y_start if self.y_start > self.y_finish else self.y_finish
    img = self.canvas.true_image.load()

    if isinstance(img[0, 0], tuple):
        if len(img[0,0]) == 4:
            total = (0, 0, 0, 0)
        else:
            total = (0, 0, 0)

        for i in range(xstart, xfinish):
            for j in range(ystart, yfinish):
                total = tuple(map(lambda x, y: x + y, total, img[i, j]))
        print("Promedio:" + str(tuple([x/pixel_count for x in total])))

    else:
        total = 0
        for i in range(xstart, xfinish):
            for j in range(ystart, yfinish):
                total += img[i, j]
        print("Promedio:" + str(total/pixel_count))


def do_clear(self):
    self.canvas.delete("all")
    print("Clear canvas done")


def generate_square(self):
    white = 255
    img_size = 200
    img_matrix = generate_empty(img_size)
    for x in range(40, img_size-40):
        for y in range(40, img_size-40):
            img_matrix[x, y] = white

    print(img_matrix.size)

    load_image_on_canvas(self,img_matrix)

    print("generate_square: DONE")


def generate_circle(self):
    white = 255
    img_size = 200
    img_matrix = generate_empty(img_size)
    center = img_size/2
    for x in range(0,img_size):
        for y in range(0,img_size):
            if (x-center)*(x-center) + (y-center)*(y-center) < 50*50:
                img_matrix[x, y] = white

    load_image_on_canvas(self, img_matrix)


def generate_degrade(self):
    matrix = np.zeros((255, 100), dtype=np.uint8)
    for y in range(0, 255):
        for x in range(0, 100):
            matrix[y][x] = y

    print(matrix)
    load_image_on_canvas(self, matrix)


def generate_color_degrade(self):
    matrix = np.zeros((255*255*255))


def generate_empty(img_size):
    default_color = 0
    return np.full((img_size, img_size), default_color, dtype=np.uint8)


def load_image_on_canvas(self, matrix):
    height, width = matrix.shape
    img = Image.fromarray(matrix, 'L')
    photo = ImageTk.PhotoImage(img)
    self.canvas.image = photo
    self.canvas.true_image = img
    self.canvas.configure(width=width, height=height)
    self.canvas.create_image((0, 0), image=photo, anchor='nw')
    self.canvas.pack()


def rgb_to_hsv(self):
    pixels = self.canvas.true_image.load()
    width, height = self.canvas.true_image.size
    data = np.zeros((width, height, len(pixels[0, 0])), dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, height):
            data[x, y] = pixels[x, y]
    load_image_on_canvas(self, data[:, :, 0])
    # img = Image.fromarray(data[:,:,1],'L')

def add(self):
    self.new_window = Toplevel()
    self.new_window.minsize(width=800, height=600)
    self.new_window.title("Add")
    canvas = Canvas(self.new_window,height=300,width=300)

    size1=[0,0]
    size2=[0,0]

    matrix_img1 = [[0 for i in range(MAX_WIDTH)] for j in range (MAX_HEIGHT)]
    matrix_img2 = [[0 for i in range(MAX_WIDTH)] for j in range (MAX_HEIGHT)]

    b1 = Button(self.new_window, text="Seleccionar IMAGEN 1", command=lambda: kevin_open(self, canvas, size1, matrix_img1))
    b2 = Button(self.new_window, text="Seleccionar IMAGEN 2", command=lambda: kevin_open(self, canvas, size2, matrix_img2))
    check = Button(self.new_window, text="Informacion de seleccionados", command=lambda: info_selected(self, size1, size2, matrix_img1, matrix_img2))

    b1.pack()
    b2.pack()
    check.pack()


def info_selected(self, size1, size2, matrix_img1, matrix_img2):
    print("Info de las imagenes seleccionadas")
    print("Size IMG1: " + str(size1))
    print("Size IMG2: " + str(size2))
    print("Primeros 4 de IMG1")
    print(matrix_img1[0][0])
    print(matrix_img1[0][1])
    print(matrix_img1[1][0])
    print(matrix_img1[1][1])
    print("Primeros 4 de IMG2")
    print(matrix_img2[0][0])
    print(matrix_img2[0][1])
    print(matrix_img2[1][0])
    print(matrix_img2[1][1])
    print("***********************")

    return

#TODO: Cambiar el nombre por algo mas generico
def kevin_open(self, canvas, to_ret, matrix):
    filename = filedialog.askopenfilename()
    print(filename)
    if filename.find("RAW") != -1:
        with open('raw.json') as json_data:
            d = json.load(json_data)
        dim = d['data'][filename.rsplit(".", 1)[0].rsplit("/", 1)[1]]
        print(dim['x'])
        print(dim['y'])
        image = Image.frombytes('F', (dim['x'], dim['y']), open(filename, "rb").read(), 'raw', 'F;8')
        filename = ImageTk.PhotoImage(image)
    else:
        image = Image.open(filename)
        filename = ImageTk.PhotoImage(image)

    canvas.image = filename
    width, height = image.size
    canvas.create_image(0,0,anchor='nw',image=filename)
    canvas.pack()

    canvas.true_image = image
    img_matrix = canvas.true_image.load()
    print("COLOR: " + str(img_matrix[0,0]))

    for i in range(width):
        for j in range(height):
            matrix[i][j] = img_matrix[i,j]

    to_ret[0] = width
    to_ret[1] = height
    print(to_ret)


def supr(self):
    print("TODO")

def mult(self):
    print("TODO")

# La funcion de escalar no funciona cuando la imagen NO es cuadrada. Dice que se va fuera de los bounds, pero no encuentro
# porque es que eso pasa! -> Linea 201: aux_matrix[i,j] = matrix[i,j] * scale
def scalar_mult(self):
    img2 = self.canvas.true_image.load()
    width, height = self.canvas.true_image.size
    scale = 2

    print("width: " + str(width))
    print("height: " + str(height))
    try:
        len(img2[0,0])
    except TypeError:
        print("BLANCO Y NEGRO")
        aux_matrix = np.zeros((height, width), dtype=np.dtype('i8'))
        aux_matrix = iterate(self, height, width, scale, img2)
        return aux_matrix

    print("COLOR")
    aux_matrix = iterate_color(self, height, width, scale, img2)
    return aux_matrix

def iterate(self, h, w, scale, matrix):
    aux_matrix = np.zeros((h, w), dtype=np.dtype('i8'))
    print("El valor de w: " + str(w))
    print("El valor de h: " + str(h))
    for i in range(w):
        for j in range(h):
            print("i: " + str(i) + " j: " + str(j))
            aux_matrix[i,j] = matrix[i,j] * scale

    return aux_matrix

#Esta funcion no esta terminada!
def iterate_color(self, h, w, scale, matrix):
    print("El valor de w: " + str(w))
    print("El valor de h: " + str(h))
    for i in range(w):
        for j in range(h):
            print(matrix[i,j])

    return


def show_hist(self):
    hist.get_histogram(np.array(self.canvas.true_image))


def equalize(self):

    matrix = hist.equalize(np.array(self.canvas.true_image, dtype=np.uint8))
    print("Shape" + str(matrix.shape))
    print("element" + str(matrix[0, 0]))
    e = Image.fromarray(np.array(matrix, dtype=np.uint8))
    i = ImageTk.PhotoImage(e)
    self.canvas.true_image = e
    self.canvas.image = i
    self.canvas.create_image((0, 0), anchor="nw", image=i)
