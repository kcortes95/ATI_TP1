import numpy as np
from tkinter import Toplevel, Button, Entry, Label
import actions

def anisotropic_diffusion(matrix, iterations, sigma):
    for i in range(iterations):
        for y in range(1, matrix.shape[0]-1):
            for x in range(1, matrix.shape[1]-1):
                matrix[y, x] = matrix[y, x] + addition(matrix, x, y, sigma)

    return matrix


def addition(matrix, x, y, sigma):
    direction = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
    derivates = np.zeros(4)
    res = 0
    for i in range(4):
        derivates[i] = matrix[y + direction[i, 0], x + direction[i, 1]] - matrix[y, x]
        res += 0.25*derivates[i]*leclerc(derivates[i], sigma)

    return res


def leclerc(der, sigma):
    return np.exp(-(der**2)/(sigma**2))


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

def isotropic(self, tvalue):
    height, width = self.canvas.true_image.size

    img_original = np.array(self.canvas.true_image, dtype=np.int16)
    actions.matrix_to_window(self, img_original, "Original", 'L')
    actions.gauss_filter(self, 3, tvalue)
    img_gauss = np.array(self.canvas.true_image, dtype=np.int16)
    actions.matrix_to_window(self, img_gauss, "Img w/ Gauss", 'L')
    actions.multiply(self, img_original, img_gauss, "Result", 'L')

