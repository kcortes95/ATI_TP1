from tkinter import Tk, Toplevel, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk
import numpy as np
import math

def set_color(self):
    color = int(self.color_text.get())
    set_pixel(color, self.x_text.get(), self.y_text.get())

def set_pixel(self, color, x, y):
    self.canvas.true_image.putpixel((int(x), int(y)), color)
    self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

def crop(self, master):
    self.new_window = Toplevel()
    self.new_window.minsize(width=640, height=480)
    self.new_window.config(menu=self.menubar)
    img = self.canvas.true_image.load()

    xstart = self.x_start if self.x_start < self.x_finish else self.x_finish
    ystart = self.y_start if self.y_start < self.y_finish else self.y_finish
    xfinish = self.x_start if self.x_start > self.x_finish else self.x_finish
    yfinish = self.y_start if self.y_start > self.y_finish else self.y_finish
    new_image = np.zeros((yfinish - ystart, xfinish - xstart), dtype=np.uint8)
    # print("x from " + str(self.x_start) + " to  " + str(self.x_finish))
    # print("y from " + str(self.y_start) + " to  " + str(self.y_finish))
    x = 0
    y = 0
    for x_pos in range(xstart, xfinish):
        for y_pos in range(ystart, yfinish):
            print(str(x_pos) + " " + str(y_pos))
            aux = img[x_pos, y_pos]
            new_image[x, y] = aux
            x += 1
        y += 1
        x = 0
    true_cropped = Image.fromarray(new_image, 'L')
    cropped = ImageTk.PhotoImage(true_cropped)
    self.new_window.canvas = Canvas(self.new_window, width=200, height=200)
    self.new_window.canvas.true_cropped = true_cropped
    self.new_window.canvas.cropped = cropped
    self.new_window.canvas.configure(width=true_cropped.width, height=true_cropped.height)
    self.new_window.canvas.create_image((0, 0), anchor="nw", image=cropped)
    self.new_window.canvas.pack()


def get_area_info(self, master):
    pixel_count = abs(self.x_start - self.x_finish) * abs(self.y_start - self.y_finish)
    print("Cantidad de pixeles:" + str(pixel_count))
    img = self.canvas.true_image.load()
    total = 0
    for i in range(self.x_start, self.x_finish):
        for j in range(self.y_start, self.y_finish):
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
