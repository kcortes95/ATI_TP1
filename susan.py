from tkinter import Tk, Toplevel, Scale, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage, LEFT
from PIL import Image, ImageTk
import numpy as np
import border as bd
import thresholds as th
import meshoperations as mesh

def main_susan():
    circular_mask = [[0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0]]

    img_aux =

def apply_mask(self, img_aux):
    width, height =
    for i in range(0, width - 7):
        for j in range(0, height - 7):
            img_aux =

def calculate_s(img, umbral):
    sum = 0;
    for i in range(6):
        for j in range(6):
            if(img[i][j] - img[3][3] < umbral)
                sum += 1

    return 1 - sum/49 #49 es la cantidad de elementos en la máscara
