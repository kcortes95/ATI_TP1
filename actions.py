from tkinter import Tk, Toplevel, Scale, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage, LEFT
from PIL import Image, ImageTk
import numpy as np
import math
import json
import histoperations as hist
import meshoperations as mesh
import tkinter.messagebox as msgbox
import random
import myrandom as myrand
import anistropic as ani

MAX_HEIGHT = 1024
MAX_WIDTH = 1024

def set_color(self):
    color = int(self.color_text.get())
    set_pixel(self,color, int(self.x_text.get()), int(self.y_text.get()))

def set_pixel(self, color, x, y):
    self.true_image.putpixel((int(x), int(y)), color)
    self.canvas.image = ImageTk.PhotoImage(self.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

def crop(self, master):
    self.new_window = Toplevel()
    self.new_window.minsize(width=640, height=480)

    button = Button(self.new_window, text="Guardar", command=lambda: save_cropped(self, master))
    button.pack()
    img = self.true_image.load()

    xstart = self.x_start if self.x_start < self.x_finish else self.x_finish
    ystart = self.y_start if self.y_start < self.y_finish else self.y_finish
    xfinish = self.x_start if self.x_start > self.x_finish else self.x_finish
    yfinish = self.y_start if self.y_start > self.y_finish else self.y_finish
    if isinstance(img[0, 0], tuple):
        new_image = np.zeros((yfinish - ystart, xfinish - xstart, len(img[0, 0])), dtype=np.uint8)
    else:
        if self.true_image.mode == 'F' :
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
    true_cropped = Image.fromarray(new_image, self.true_image.mode)
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

def get_area_info(self, matrix,left):
    if left :
        pixel_count = abs(self.x_start - self.x_finish) * abs(self.y_start - self.y_finish)
        print("Cantidad de pixeles:" + str(pixel_count))
        xstart = self.x_start if self.x_start < self.x_finish else self.x_finish
        ystart = self.y_start if self.y_start < self.y_finish else self.y_finish
        xfinish = self.x_start if self.x_start > self.x_finish else self.x_finish
        yfinish = self.y_start if self.y_start > self.y_finish else self.y_finish
    else:
        pixel_count = abs(self.x2_start - self.x2_finish) * abs(self.y2_start - self.y2_finish)
        print("Cantidad de pixeles:" + str(pixel_count))
        xstart = self.x2_start if self.x2_start < self.x2_finish else self.x2_finish
        ystart = self.y2_start if self.y2_start < self.y2_finish else self.y2_finish
        xfinish = self.x2_start if self.x2_start > self.x2_finish else self.x2_finish
        yfinish = self.y2_start if self.y2_start > self.y2_finish else self.y2_finish
    img = matrix

    if isinstance(img[0, 0], np.ndarray):
        if len(img[0,0]) == 4:
            total = (0, 0, 0, 0)
        else:
            total = (0, 0, 0)

        for j in range(xstart, xfinish):
            for i in range(ystart, yfinish):
                total = tuple(map(lambda x, y: x + y, total, img[i, j]))
        print("Promedio:" + str(tuple([x/pixel_count for x in total])))
        return tuple([x/pixel_count for x in total])
    else:
        total = 0
        for j in range(xstart, xfinish):
            for i in range(ystart, yfinish):
                total += img[i, j]
        print("Promedio:" + str(total/pixel_count))
        return total/pixel_count

def do_clear(self):
    self.canvas.delete("all")
    print("Clear canvas done")

def load_image_on_canvas(self, matrix):
    if len(matrix.shape) == 2:
        height, width = matrix.shape
    else:
        height, width, depth = matrix.shape
    img = Image.fromarray(matrix)
    photo = ImageTk.PhotoImage(img)
    self.canvas.image = photo
    self.true_image = img
    self.canvas.configure(width=width, height=height)
    self.canvas.create_image((0, 0), image=photo, anchor='nw')
    self.canvas.grid(row=0,column=0)

def rgb_to_hsv(self):
    pixels = self.true_image.load()
    width, height = self.true_image.size
    data = np.zeros((width, height, len(pixels[0, 0])), dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, height):
            data[x, y] = pixels[x, y]
    load_image_on_canvas(self, data[:, :, 0])
    # img = Image.fromarray(data[:,:,1],'L')

#--------------------KEVIN--------------------

def to_negative(self):
    true_img = self.true_image
    pixels = true_img.load()
    ar = np.array(true_img, dtype=np.uint8)
    width, height = true_img.size

    type = 'RGB'

    try:
        len(pixels[0, 0])
    except TypeError:
        type = 'L'

    img_neg = 255 - ar
    load_image_on_canvas(self,img_neg)
    # matrix_to_window(self, img_neg, "Negative", type)

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

def opr(self):
    self.new_window = Toplevel()
    self.new_window.minsize(width=480, height=360)
    self.new_window.title("Operations")

    canvas = Canvas(self.new_window,height=300,width=300)
    canvas2 = Canvas(self.new_window, height=300, width=300)
    canvas_result = Canvas(self.new_window, height=300, width=300)

    matrix_img1 = [[0 for i in range(MAX_WIDTH)] for j in range (MAX_HEIGHT)]
    matrix_img2 = [[0 for i in range(MAX_WIDTH)] for j in range (MAX_HEIGHT)]

    b1 = Button(self.new_window, text="Select IMG 1", command=lambda: kevin_open(self, canvas, matrix_img1, 1, 0))
    b2 = Button(self.new_window, text="Select IMG 2", command=lambda: kevin_open(self, canvas2, matrix_img2, 1, 1))
    check = Button(self.new_window, text="DONE", command=lambda: operations(self, canvas, canvas2))

    b1.grid(row=0,column=0)
    b2.grid(row=0,column=1)
    check.place(relx=0.5, rely=1, relwidth=1 ,anchor="s", bordermode="outside")

def operations(self, canvas1, canvas2):

    type1 = get_img_type_from_canvas(self, canvas1)
    type2 = get_img_type_from_canvas(self, canvas2)

    if type1 != type2:
        err_msg('You cant operate with color and blanck and white simultaneously.')
        raise ValueError('A very specific bad thing happened')

    sum(self, np.array(canvas1.true_image,dtype=np.uint8), np.array(canvas2.true_image,dtype=np.uint8), "SUM", type1)
    sum(self, np.array(canvas1.true_image,dtype=np.uint8), -1*np.array(canvas2.true_image,dtype=np.uint8), "SUBSTRACT", type1)
    multiply(self, np.array(canvas1.true_image,dtype=np.uint8), np.array(canvas2.true_image,dtype=np.uint8), "MULTIPLY", type1)

#Es la funcion para la suma, aunque la uso tambien para la resta
def sum(self, matrix_img1, matrix_img2, title, type):

    if matrix_img1.shape == matrix_img2.shape:
        out = matrix_img1 + matrix_img2
    else:
        shape = (max(matrix_img1.shape[0], matrix_img2.shape[0]), max(matrix_img1.shape[1], matrix_img2.shape[1]))
        print(shape)
        # out = np.zeros(shape, dtype=np.uint8)
        out = np.zeros(shape, dtype=np.int8)
        out[:matrix_img1.shape[0], :matrix_img1.shape[1]] = matrix_img1
        out[:matrix_img2.shape[0], :matrix_img2.shape[1]] += matrix_img2

    matrix_to_window(self, linear_transform(out), title + " CON TL", type)
    #matrix_to_window(self, out, title + " SIN TL", type)

    return out

def multiply(self, matrix_img1, matrix_img2, title, type):

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
    matrix_to_window(self, linear_transform(out), title, type)
    return out

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
    img = Image.fromarray(np.array(out, dtype=np.uint8))

    print("type: " + type)

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

def pscreen(self):
    print("Hola mundo")

def to_main_canvas(self, can):
    self.true_image = can.true_image
    self.canvas.image = ImageTk.PhotoImage(self.true_image)
    self.canvas.create_image((0, 0), image=self.canvas.image, anchor="nw")
    self.canvas.configure(width=can.true_image.size[0], height=can.true_image.size[1])

def default_save(can, filename):
    can.true_image.save(filename)
    print(filename)

def save(window, can):
    filename = filedialog.asksaveasfilename(parent=window)
    can.true_image.save(filename)
    print(filename)

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
    img2 = self.true_image.load()
    width, height = self.true_image.size
    ar = np.array(self.true_image, dtype=np.int32)
    ar = float(scale) * ar
    ar = din_range(self, ar)
    img_ar = Image.fromarray(ar)
    img = ImageTk.PhotoImage(img_ar)
    self.true_image = img_ar
    self.canvas.image = img
    self.canvas.create_image((0, 0), anchor="nw", image=img)

#DE ACA HASTA KEVIN SON LOS METODOS PARA LOS RUIDOS
def percentage_textbox(self, action):
    self.new_window = Toplevel()
    self.new_window.minsize(width=200, height=70)
    self.new_window.title("Enter percentage")
    self.l=Label(self.new_window,text="Integer value from 0 to 100")
    self.l.pack()
    self.per = Entry(self.new_window)
    self.per.pack()
    self.ok = Button(self.new_window, text="OK", width=10, height=1, command=lambda: generic_window(self, int(self.per.get()), action))
    self.ok.pack()

def generic_window(self, percentage, action):

    height, width = self.true_image.size
    img_arr = np.array(self.true_image, dtype=np.int16)

    type = get_img_type(self)
    #type = 'L'

    if percentage < 0 or percentage > 100:
        err_msg("Invalid percentage")

    if action == 'gaussian': #aditivo
        gaussian_window_values(self, width, height, img_arr, percentage, type)
    elif action == 'rayleigh': #multiplicativo
        rayleigh_window_values(self, width, height, img_arr, percentage, type)
    elif action == 'exponential': #multiplicativo
        exponential_window_values(self, width, height, img_arr, percentage, type)
    elif action == 'salt_and_pepper':
        sap_window_values(self, width, height, img_arr, percentage, type)
        print("")
    else:
        print("All")

def gaussian_window_values(self, width, height, img_arr, percentage, type):
    self.g_win = Toplevel()
    self.g_win.minsize(width=200, height=140)
    self.g_win.title("Gaussian Values")

    self.mu=Label(self.g_win,text="Mu:")
    self.mu.pack()
    self.mu_val = Entry(self.g_win)
    self.mu_val.pack()

    self.sigma=Label(self.g_win,text="Sigma:")
    self.sigma.pack()
    self.sigma_val = Entry(self.g_win)
    self.sigma_val.pack()

    self.ok = Button(self.g_win, text="OK", width=10, height=1, command=lambda: ret_gaussian_window(self, width, height, img_arr, percentage, float(self.mu_val.get()), float(self.sigma_val.get()), type))
    self.ok.pack()

def ret_gaussian_window(self, width, height, img_arr, percentage, mu, sigma, type):
    tot_pixels = int((width * height) * (percentage/100))

    for i in range(tot_pixels):
        ranx = random.randint(0, width-1)
        rany = random.randint(0, height-1)
        img_arr[ranx][rany] = random.gauss(mu,sigma) + np.array(img_arr[ranx][rany])

    matrix_to_window(self, linear_transform(img_arr), "Gaussian " + str(percentage) + "%", type )
    return linear_transform(img_arr)

def rayleigh_window_values(self, width, height, img_arr, percentage, type):
    self.g_win = Toplevel()
    self.g_win.minsize(width=200, height=140)
    self.g_win.title("Rayleigh Values")

    self.xi=Label(self.g_win,text="Xi:")
    self.xi.pack()
    self.xi_val = Entry(self.g_win)
    self.xi_val.pack()

    self.ok = Button(self.g_win, text="OK", width=10, height=1, command=lambda: ret_rayleigh_window(self, width, height, img_arr, percentage, float(self.xi_val.get()), type))
    self.ok.pack()

def ret_rayleigh_window(self, width, height, img_arr, percentage, xi, type):
    tot_pixels = int((width * height) * (percentage/100))
    print("w: " + str(width))
    print("h: " + str(height))
    print("tot_pixels: " + str(tot_pixels))
    print("%: " + str(percentage))
    print("xi: " + str(xi))

    for i in range(tot_pixels):
        ranx = random.randint(0, width-1)
        rany = random.randint(0, height-1)
        # print("W: " + str(width) + " H: " + str(height) + " ||| " + "RANDOM X: " + str(ranx) + " RANDOM Y: " + str(rany))
        # img_arr[ranx][rany] *= myrand.rayleight_random(xi)
        img_arr[ranx][rany] = myrand.rayleight_random(xi) * np.array(img_arr[ranx][rany])

    matrix_to_window(self, linear_transform(img_arr), "Rayleigh " + str(percentage) + "%", type)

def exponential_window_values(self, width, height, img_arr, percentage, type):
    self.g_win = Toplevel()
    self.g_win.minsize(width=200, height=140)
    self.g_win.title("Exponential Values")

    self.lam=Label(self.g_win,text="Lambda:")
    self.lam.pack()
    self.lam_val = Entry(self.g_win)
    self.lam_val.pack()

    self.ok = Button(self.g_win, text="OK", width=10, height=1, command=lambda: ret_exponential_window(self, width, height, img_arr, percentage, float(self.lam_val.get()), type))
    self.ok.pack()

def ret_exponential_window(self, width, height, img_arr, percentage, lam, type):
    tot_pixels = int((width * height) * (percentage/100))
    print("w: " + str(width))
    print("h: " + str(height))
    print("tot_pixels: " + str(tot_pixels))
    print("%: " + str(percentage))
    print("lambda: " + str(lam))

    print("[0][0]: " + str(img_arr[0][0]));
    print("[0][1]: " + str(img_arr[0][1]));
    print("[1][0]: " + str(img_arr[1][0]));

    for i in range(tot_pixels):
        ranx = random.randint(0, width-1)
        rany = random.randint(0, height-1)
        img_arr[ranx][rany] = myrand.exponential_random(lam) * np.array(img_arr[ranx][rany])

    matrix_to_window(self, linear_transform(img_arr), "Exponential " + str(percentage) + "%", type)

def sap_window_values(self, width, height, img_arr, percentage, type):
    p0 = np.random.rand()
    p1 = 1 - p0

    tot_pixels = int((width * height) * (percentage/100))

    for i in range(tot_pixels):
        x = np.random.rand()
        ranx = random.randint(0,width-1)
        rany = random.randint(0, height-1)

        if x <= p0:
            img_arr[ranx][rany] = 0
        elif x >= p1:
            img_arr[ranx][rany] = 255

    matrix_to_window(self, img_arr, "Salt and Pepper " + str(percentage) + "%", type)
    # matrix_to_window(self, linear_transform(img_arr), "Salt and Pepper" + str(percentage) + "%", type)

def err_msg(message):
    window = Tk()
    window.wm_withdraw()
    msgbox.showinfo(title="Error", message=message)

def gamma_textbox(self):
    self.new_window = Toplevel()
    self.new_window.minsize(width=200, height=70)
    self.new_window.title("Enter gamma")
    self.l=Label(self.new_window,text="Enter gamma number")
    self.l.pack()
    self.per = Entry(self.new_window)
    self.per.pack()
    self.ok = Button(self.new_window, text="OK", width=10, height=1, command=lambda: gamma_function(self, float(self.per.get())))
    self.ok.pack()

def gamma_function(self, gam):
    type = get_img_type(self)
    print(type)
    print("lambda " + str(gam))
    c = 255**(1-gam)
    img_arr = c * (np.array(self.true_image, dtype=np.uint8)**gam)
    # load_image_on_canvas(self, img_arr) #FUCK, con esto me rompe
    matrix_to_window(self, linear_transform(img_arr), "Gamma Function", type)


def din_range(self, matrix):
    prtn = False
    if matrix is None:
        matrix = np.array(self.true_image, dtype=np.uint8)
        prtn = True

    max = np.amax(matrix) #ESTO HAY QUE CALCULARLE EL MAXIMO DE LA IMAGEN
    print("RANGO DINAMICO. MAX: " + str(max))

    c = 255 / math.log(1+max)
    if len(matrix.shape) == 2:
        height, width = matrix.shape
    else:
        height, width,depth = matrix.shape

    type = get_img_type(self)
    img_arr = matrix

    if type == 'L':
        for i in range(width):
            for j in range(height):
                img_arr[j][i] = c*math.log(img_arr[j][i] + 1)
    else:
        for i in range(width):
            for j in range(height):
                img_arr[j][i][0] = c*math.log(img_arr[j][i][0] + 1)
                img_arr[j][i][1] = c*math.log(img_arr[j][i][1] + 1)
                img_arr[j][i][2] = c*math.log(img_arr[j][i][2] + 1)

    print(img_arr)
    img_arr = img_arr.astype(np.uint8)
    print(np.amax(img_arr))
    if prtn:
        matrix_to_window(self, img_arr, "Dinamic Range", type)

    return img_arr

#ACA EMPIEZA LO DE DIFERENCIA ANSIOTROPICA

def data_difansi(self, type):
    self.new_window = Toplevel()
    self.new_window.minsize(width=200, height=70)
    self.new_window.title(type + " - Enter Gamma:")
    self.l=Label(self.new_window,text="Enter a valid gamma number")
    self.l.pack()
    self.entry_gamma = Entry(self.new_window)
    self.entry_gamma.pack()
    self.ok = Button(self.new_window, text="OK", width=10, height=1, command=lambda: g_function(self, type, float(self.entry_gamma.get()), 1, 1))
    self.ok.pack()


def g_function(self, type, gamma, step, step_max):

    height, width = self.true_image.size
    img_arr = np.array(self.true_image, dtype=np.int16)

    print("w: " + str(width));
    print("h:" + str(height));
    print("[0][0]: " + str(img_arr[5][5]))

    #for iteration in range(20):
    #    print("Iteracion nº: " + str(iteration))
    #    for i in range(width):
    #        for j in range(height):
    #            derivadas = derivada(self, img_arr, i, j, width, height)
    #            constantes = constante(img_arr, i, j, gamma, derivadas, type)
    #            img_arr[i][j] = img_arr[i][j] + 0.25*( derivadas[0]*constantes[0] +  derivadas[1]*constantes[1] + derivadas[2]*constantes[2] + derivadas[2]*constantes[2])

    img_arr = ani.anisotropic_diffusion(img_arr, 20, gamma)
    print("[0][0]: " + str(img_arr[5][5]))
    matrix_to_window(self, img_arr, "PASO 1", get_img_type(self))
    # step += 1

# 0 NORTE
# 1 SUR
# 2 ESTE
# 3 OESTE
def derivada(self, img_arr, i, j, w, h):
    derivadas = []
    coordenadas = [[0,-1],[0,1],[1,0],[-1,0]]
    for each in range(4): #siempre van a ser 4 coordenadas.
        width = i + coordenadas[each][0]
        height = j + coordenadas[each][1]

        #el if de la muerte
        if width < w and height < h and width > 0 and height > 0:
            pixel_calculated = img_arr[width][height]
            pixel_center = img_arr[i][j]
            derivadas.append(pixel_calculated - pixel_center)
        else:
            # derivadas.append(img_arr[i][j])
            derivadas.append(0)

    return derivadas

def constante(img_arr, i, j, gamma, derivadas, type):
    constantes = np.zeros(4,np.int32)

    for each in range(4):
        if type == 'leclerc':
            dxi = derivadas[each]*img_arr[i][j]
            glerc = g_leclerc(gamma, dxi)
            # print(glerc)
            constantes[each] = glerc
        else:
            constantes.append(g_lorentziano(gamma, derivadas[each]*img_arr[i][j]))

    return constantes

#ACA TERMINA LO DE DIFERENCIA ANSIOTROPICA

#--------------------KEVIN--------------------

def show_hist(self):
    hist.get_histogram(np.array(self.true_image, dtype=np.uint8))


def equalize(self):
    matrix = hist.equalize(np.array(self.true_image, dtype=np.uint8))
    print("Shape" + str(matrix.shape))
    print("element" + str(matrix[0, 0]))
    e = Image.fromarray(np.array(matrix, dtype=np.uint8))
    i = ImageTk.PhotoImage(e)
    self.true_image = e
    self.canvas[0].image = i
    self.canvas[0].create_image((0, 0), anchor="nw", image=i)


def contrast(image,s1,s2):
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


def linear_transform(matrix):
    return np.interp(matrix, [np.min(matrix),np.max(matrix)], [0,255]).astype(np.uint8)


def rotate(matrix):
    return np.rot90(matrix)


def to_grayscale(matrix):
    if len(matrix.shape) == 1:
        return matrix
    #res = np.empty((matrix.shape[0], matrix.shape[1]))
    res = np.apply_over_axes(np.average, matrix, [2])
    print(res.shape)
    return res.reshape((matrix.shape[0], matrix.shape[1]))
    #for i in range(matrix.shape[0]):
    #    for j in range(matrix.shape[1]):
    #        res[i, j] =

#    return res