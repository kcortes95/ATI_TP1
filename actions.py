from tkinter import Tk, Toplevel, Entry, Label, Button, messagebox, Menu, filedialog, Canvas
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
    ydif = abs(self.y_finish - self.y_start)
    xdif = abs(self.x_finish - self.x_start)
    new_image = np.zeros((ydif, xdif), dtype=np.uint8)
    x_starter = 0 if self.x_start < self.x_finish else xdif - 1
    x = x_starter
    y = 0 if self.y_start < self.y_finish else ydif - 1
    x_advance = 1 if self.x_start < self.x_finish else -1
    y_advance = 1 if self.y_start < self.y_finish else -1
    print("x from " + str(self.x_start) + " to  " + str(self.x_finish))
    print("y from " + str(self.y_start) + " to  " + str(self.y_finish))
    for x_pos in range(self.x_start, self.x_finish, x_advance):
        for y_pos in range(self.y_start, self.y_finish, y_advance):
            print(str(x_pos) + " " + str(y_pos))
            aux = img[y_pos, x_pos]
            new_image[y, x] = aux
            x += x_advance
        y += y_advance
        x = x_starter
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
    img = self.canvas.true_image.load();
    total = 0
    for i in range(self.x_start, self.x_finish):
        for j in range(self.y_start, self.y_finish):
            total += img[i, j]
    print("Promedio:" + str(total/pixel_count))

def do_clear(self):
    self.canvas.delete("all")
    print("Clear canvas done")

def generate_square(self):
    img_size = 200
    default_color = 0
    black = 0
    white = 255

    img_matrix = [[default_color] * img_size for i in range(img_size)]

    for x in range(img_size):
        for y in range(img_size):
            if (y==0 or y == (img_size-1)) or (x==0 or x == (img_size-1)):
                img_matrix[x][y] = white
                set_pixel(self, white, x, y)

    print("generate_square: DONE")

def generate_circle(self):
    print("generate_circle: TO DO")

def generate_degrade(self):
    print("generate_degrade: TO DO")
