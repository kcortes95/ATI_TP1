from tkinter import Tk, Toplevel, Scale, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage, LEFT
from PIL import Image, ImageTk
import numpy as np
import math
import json
import histoperations as hist
import meshoperations as mesh
import tkinter.messagebox as msgbox
import random

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

def load_image_on_canvas(self, matrix):
    height, width = matrix.shape
    img = Image.fromarray(matrix, 'L')
    photo = ImageTk.PhotoImage(img)
    self.canvas.image = photo
    self.canvas.true_image = img
    self.canvas.configure(width=width, height=height)
    self.canvas.create_image((0, 0), image=photo, anchor='nw')
    self.canvas.grid(row=0,column=0)

def rgb_to_hsv(self):
    pixels = self.canvas.true_image.load()
    width, height = self.canvas.true_image.size
    data = np.zeros((width, height, len(pixels[0, 0])), dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, height):
            data[x, y] = pixels[x, y]
    load_image_on_canvas(self, data[:, :, 0])
    # img = Image.fromarray(data[:,:,1],'L')

#--------------------KEVIN--------------------

def opr(self):
    self.new_window = Toplevel()
    self.new_window.minsize(width=480, height=360)
    self.new_window.title("Add")

    canvas = Canvas(self.new_window,height=300,width=300)
    canvas2 = Canvas(self.new_window, height=300, width=300)
    canvas_result = Canvas(self.new_window, height=300, width=300)

    matrix_img1 = [[0 for i in range(MAX_WIDTH)] for j in range (MAX_HEIGHT)]
    matrix_img2 = [[0 for i in range(MAX_WIDTH)] for j in range (MAX_HEIGHT)]

    b1 = Button(self.new_window, text="Select IMG 1", command=lambda: kevin_open(self, canvas, matrix_img1, 1, 0))
    b2 = Button(self.new_window, text="Select IMG 2", command=lambda: kevin_open(self, canvas2, matrix_img2, 1, 1))
    check = Button(self.new_window, text="DONE", command=lambda: operations(self, np.array(canvas.true_image,dtype=np.uint8), np.array(canvas2.true_image,dtype=np.uint8)))

    b1.grid(row=0,column=0)
    b2.grid(row=0,column=1)
    check.place(relx=0.5, rely=1, relwidth=1 ,anchor="s", bordermode="outside")

def operations(self, matrix_img1, matrix_img2):
    sum(self, matrix_img1, matrix_img2, "SUM")
    sum(self, matrix_img1, -1*np.array(matrix_img2), "SUBSTRACT")
    multiply(self, matrix_img1, matrix_img2, "MULTIPLY")

def sum(self, matrix_img1, matrix_img2, title):

    if matrix_img1.shape == matrix_img2.shape:
        out = matrix_img1 + matrix_img2
    else:
        shape = (max(matrix_img1.shape[0], matrix_img2.shape[0]), max(matrix_img1.shape[1], matrix_img2.shape[1]))
        print(shape)
        # out = np.zeros(shape, dtype=np.uint8)
        out = np.zeros(shape, dtype=np.int8)
        out[:matrix_img1.shape[0], :matrix_img1.shape[1]] = matrix_img1

        matrix_to_window
        # print(matrix_img1.shape[0]) #alto img1
        # print(matrix_img1.shape[1]) #ancho img1
        # print(matrix_img2.shape[0])
        # print(matrix_img2.shape[1])

        out[:matrix_img2.shape[0], :matrix_img2.shape[1]] += matrix_img2

    #FALTA NORMALIZAR OUT
    matrix_to_window(self, out, title)
    return out

def multiply(self, matrix_img1, matrix_img2, title):

    if matrix_img1.shape == matrix_img2.shape:
        out = matrix_img1 + matrix_img2
    else:
        shape = (max(matrix_img1.shape[0], matrix_img2.shape[0]), max(matrix_img1.shape[1], matrix_img2.shape[1]))
        print(shape)
        out = np.zeros(shape, dtype=np.int8)

        out1 = np.zeros(shape, dtype=np.int8)
        out2 = np.zeros(shape, dtype=np.int8)
        out1[:matrix_img1.shape[0], :matrix_img1.shape[1]] += matrix_img1
        out2[:matrix_img2.shape[0], :matrix_img2.shape[1]] += matrix_img2

        out = np.array(out1) * np.array(out2)

    #FALTA NORMALIZAR OUT
    matrix_to_window(self, out, title)
    return out

def matrix_to_window(self, out, title):
    height, width = out.shape

    self.result_window = Toplevel()
    self.result_window.minsize(width=width, height=height)
    self.result_window.title(title)
    canvas_result = Canvas(self.result_window, height=height, width=width)

    img = Image.fromarray(out, 'L')
    photo = ImageTk.PhotoImage(img)
    canvas_result.image = photo
    canvas_result.true_image = img
    canvas_result.configure(width=width, height=height)
    canvas_result.create_image((0, 0), image=photo, anchor='nw')
    canvas_result.grid(row=0,column=0)

def kevin_open(self, canvas, matrix, r, c):
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
    canvas.true_image = image
    img_matrix = canvas.true_image.load()

    canvas.grid(row=r,column=c)
    matrix = np.array(canvas.true_image)

def scalar_mult_textbox(self):
    self.new_window = Toplevel()
    self.new_window.minsize(width=200, height=70)
    self.new_window.title("Ingrese valor")
    self.l=Label(self.new_window,text="Ingrese el valor entero")
    self.l.pack()
    self.scale = Entry(self.new_window)
    self.scale.pack()
    self.ok = Button(self.new_window, text="OK", width=10, height=1, command=lambda: scalar_mult(self, self.scale.get()))
    self.ok.pack()

def scalar_mult(self, scale):
    img2 = self.canvas.true_image.load()
    width, height = self.canvas.true_image.size
    ar = np.array(self.canvas.true_image, dtype=np.uint8)
    ar = int(scale) * ar
    img_ar = Image.fromarray(ar)
    img = ImageTk.PhotoImage(img_ar)
    self.canvas.true_image = img_ar
    self.canvas.image = img
    self.canvas.create_image((0, 0), anchor="nw", image=img)

def percentage_textbox(self, action):
    self.new_window = Toplevel()
    self.new_window.minsize(width=200, height=70)
    self.new_window.title("Enter percentage")
    self.l=Label(self.new_window,text="Integer value from 0 to 100")
    self.l.pack()
    self.per = Entry(self.new_window)
    self.per.pack()
    self.ok = Button(self.new_window, text="OK", width=10, height=1, command=lambda: get_pixels(self, int(self.per.get()), action ) )
    self.ok.pack()


def get_pixels(self, percentage, action):
    width, height = self.canvas.true_image.size
    img_arr = np.array(self.canvas.true_image, dtype=np.uint8)

    tot = width * height

    if percentage < 0 or percentage > 100:
        err_msg("Invalid percentage")

    print("IMPRIMO SIZE: " + str(self.canvas.true_image.size))
    print("TOT PIXELS: " + str(tot))
    print("PERCENTAGE: " + str(percentage))

    mod_tot_pixels = int(tot * percentage/100)
    print("TOTAL TO CHANGE: " + str(mod_tot_pixels))

    for i in range(mod_tot_pixels):
        ranx = random.randint(0,width-1)
        rany = random.randint(0, height-1)
        print("W: " + str(width) + " H: " + str(height) + " ||| " + "RANDOM X: " + str(ranx) + " RANDOM Y: " + str(rany))
        img_arr[rany][ranx] = probabilistic_function(self, action, img_arr[ranx][rany])

    matrix_to_window(self, img_arr, action)


# Deberia retornar un numero segun la funcion para despues sumarlo o multiplicarlo
def probabilistic_function(self, action, pixel_value):
    if action == 'gaussian':
        mu = 0
        # DEBERIA PARAMETRIZAR EL VALOR DE SIGMA!!!
        sigma = 2
        s = np.random.normal(mu, sigma, 1)
        return pixel_value + s
    elif action == 'rayleigh':
        # s = np.random.normal(mu, sigma, 1)
        print("rayleigh")
        # return pixel_value * s
    elif action == 'exponential':
        print("exponential")
    elif action == 'salt_and_pepper':
        print("salt and pepper")
    else:
        print("all")


def err_msg(message):
    window = Tk()
    window.wm_withdraw()
    msgbox.showinfo(title="Error", message=message)

#--------------------KEVIN--------------------

def show_hist(self):
    hist.get_histogram(np.array(self.canvas.true_image, dtype=np.uint8))

def equalize(self):
    matrix = hist.equalize(np.array(self.canvas.true_image, dtype=np.uint8))
    print("Shape" + str(matrix.shape))
    print("element" + str(matrix[0, 0]))
    e = Image.fromarray(np.array(matrix, dtype=np.uint8))
    i = ImageTk.PhotoImage(e)
    self.canvas.true_image = e
    self.canvas.image = i
    self.canvas.create_image((0, 0), anchor="nw", image=i)

def umbral(matrix, value):
    new_matrix = np.zeros(matrix.shape)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_matrix[i, j] = 255 if matrix[i, j] >= value else 0

    return new_matrix

def contrast(self, image,s1,s2):
    matrix = np.array(image)
    out = np.zeros(matrix.shape, dtype=np.uint8)
    if isinstance(matrix[0, 0], np.ndarray):
        print("3")
        for i in range(len(matrix[0, 0])):
            aux = matrix[:, :, i]
            out[:, :, i] = apply_contrast(aux,np.mean(aux),np.std(aux), s1, s2)
    else:
        print("UNA SOLA")
        out = apply_contrast(matrix,np.mean(matrix),np.std(matrix), s1, s2)
    return out

def apply_contrast(matrix, mean, std, s1, s2):
    r1 = mean - 1/2 * std
    r2 = mean + 1/2 * std
    out = np.zeros(matrix.shape,dtype=np.uint8)
    print(matrix.shape)
    m1 = s1/r1
    m2 = (s2 - s1) / (r2 - r1)
    b2 = s1 - r1 * m2
    m3 = (s2 - 255)/(r2 - 255)
    b3 = s2 - m3*r2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > r2:
                out[i, j] = m3*matrix[i, j] + b3
                #out[i, j] = getf3(matrix[i, j],r2,s2)
            elif matrix[i, j] > r1:
                out[i, j] = m2*matrix[i, j] + b2
                #out[i, j] = getf2(matrix[i, j], r1, r2, s1, s2)
            else:
                out[i, j] = m1*matrix[i, j]
                #out[i, j] = getf1(matrix[i, j], r1, s1)

    return out

def getf3(value, r2, s2):
    m = (s2 - 255)/(r2 - 255)
    b = s2 - m*r2
    return m*value + b

def getf2(value, r1, r2, s1, s2):
    m = (s2 - s1) / (r2 - r1)
    b = s1 - r1*m
    return m*value + b

def getf1(value, r1, s1):
    return s1/r1*value

def mean_filter(self, size):
    m = mesh.mean_filter(np.array(self.canvas.true_image), size)
    self.canvas.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

def weighted_mean_filter(self, size):
    m = mesh.weighted_mean_filter(np.array(self.canvas.true_image), size)
    self.canvas.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

def gauss_filter(self, size):
    m = mesh.gauss_filter(np.array(self.canvas.true_image), size, 10)
    self.canvas.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

def highpass_filter(self,size):
    m = mesh.highpass_filter(np.array(self.canvas.true_image), size)
    self.canvas.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

def median_filter(self,size):
    m = mesh.median_filter(np.array(self.canvas.true_image), size)
    self.canvas.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)
