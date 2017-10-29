import actions as act
import numpy as np
import myrandom as rand
import random as rd

from tkinter import Tk, Entry, Toplevel,Scale, Label, LabelFrame, Button, messagebox, Menu, filedialog, Canvas, PhotoImage


def generate_square(m):
    white = 255
    img_size = 200
    img_matrix = generate_empty(img_size)
    for x in range(40, img_size-40):
        for y in range(40, img_size-40):
            img_matrix[x, y] = white

    return img_matrix

    print("generate_square: DONE")


def generate_circle(m):
    white = 255
    img_size = 200
    img_matrix = generate_empty(img_size)
    center = img_size/2
    for x in range(0,img_size):
        for y in range(0,img_size):
            if (x-center)*(x-center) + (y-center)*(y-center) < 50*50:
                img_matrix[x, y] = white

    return img_matrix


def generate_degrade(m):
    matrix = np.zeros((255, 100), dtype=np.uint8)
    for y in range(0, 255):
        for x in range(0, 100):
            matrix[y][x] = y

    return matrix


def generate_color_degrade(self):
    matrix = np.zeros((255*255*255))


def generate_empty(img_size):
    default_color = 0
    return np.full((img_size, img_size), default_color, dtype=np.uint8)


def generate_rayleigh(self):
    self.g_win = Toplevel()
    self.g_win.minsize(width=200, height=140)
    self.g_win.title("Rayleigh Values")

    self.xi=Label(self.g_win,text="Xi:")
    self.xi.pack()
    self.xi_val = Entry(self.g_win)
    self.xi_val.pack()

    self.ok = Button(self.g_win, text="OK", width=10, height=1, command=lambda: rayleigh(self,float(self.xi_val.get())))
    self.ok.pack()

def generate_gaussian(self):
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

    self.ok = Button(self.g_win, text="OK", width=10, height=1, command=lambda: gaussian(self,float(self.mu_val.get()), float(self.sigma_val.get())))
    self.ok.pack()

def generate_exponential(self):
    self.g_win = Toplevel()
    self.g_win.minsize(width=200, height=140)
    self.g_win.title("Exponential Values")

    self.lam = Label(self.g_win, text="Lambda:")
    self.lam.pack()
    self.lam_val = Entry(self.g_win)
    self.lam_val.pack()

    self.ok = Button(self.g_win, text="OK", width=10, height=1, command=lambda: exponential(self, float(self.lam_val.get())))
    self.ok.pack()

def generate_SAP(self):
    SAP(self, 0.2, 0.8)

def rayleigh(self,xi):
    matrix = np.full((300,300),128,dtype=np.int16)
    for i in range(300):
        for j in range(300):
            matrix[i, j] = 128 * rand.rayleight_random(xi)

    act.matrix_to_window(self, act.linear_transform(matrix),"Ruido Rayleigh","L")

def gaussian(self,mu,sigma):
    matrix = np.full((300,300),128,dtype=np.int16)
    for i in range(300):
        for j in range(300):
            matrix[i, j] = 128 + rand.gauss_random(sigma,mu)

    act.matrix_to_window(self, act.linear_transform(matrix),"Ruido Gaussiano","L")
    """
    Retorno la matriz de Gauss. La voy a necesitar en el primer paso de Canny.
    """
    return act.linear_transform(matrix)

def exponential(self,xi):
    matrix = np.full((300,300),128,dtype=np.int16)
    for i in range(300):
        for j in range(300):
            matrix[i, j] = 128 * rand.exponential_random(xi)

    act.matrix_to_window(self, act.linear_transform(matrix),"Ruido Exponencial","L")

def SAP(self,min,max):
    matrix = np.full((300,300),128,dtype=np.int16)
    for i in range(300):
        for j in range(300):
            random = rd.random()
            if random > max:
                matrix[i, j] = 255
            elif random < min:
                matrix[i, j] = 0
    act.matrix_to_window(self, act.linear_transform(matrix),"Ruido SAP","L")
