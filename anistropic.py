import numpy as np
import math
from tkinter import Toplevel, Button, Entry, Label
import actions
import meshoperations as mesh

direction = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
derivates = np.zeros(4)


def anisotropic_diffusion(matrix, iterations, sigma,function):
    for i in range(iterations):
        for y in range(1, matrix.shape[0]-1):
            for x in range(1, matrix.shape[1]-1):
                matrix[y, x] = matrix[y, x] + addition(matrix, x, y, sigma,function)

    return matrix


def addition(matrix, x, y, sigma,function):
    res = 0
    for i in range(4):
        derivates[i] = matrix[y + direction[i, 0], x + direction[i, 1]] - matrix[y, x]
        res += 0.25*derivates[i]*function(derivates[i], sigma)

    return res


def leclerc(matrix,iterations, sigma):
    return anisotropic_diffusion(matrix,iterations,sigma,leclerc_function)


def lorentziano(matrix, iterations, sigma):
    return anisotropic_diffusion(matrix, iterations, sigma, lorentziano_function)


def identity(module,sigma):
    return 1


def isotropic(matrix, iterations):
    return anisotropic_diffusion(matrix, iterations, 1, identity)


def leclerc_function(module, sigma):
    return math.exp(-(module*module) / (sigma*sigma))


def lorentziano_function(module, sigma):
    return 1 / (pow((module/sigma), 2) + 1)


def data_difiso(self):
    self.new_window = Toplevel()
    self.new_window.minsize(width=200, height=70)
    self.new_window.title("T value for Gauss")
    self.l=Label(self.new_window,text="Enter a valid t number")
    self.l.pack()
    self.entry_t = Entry(self.new_window)
    self.entry_t.pack()
    self.ok = Button(self.new_window, text="OK", width=10, height=1, command=lambda: isotropic(self, float(self.entry_t.get())))
    self.ok.pack()

