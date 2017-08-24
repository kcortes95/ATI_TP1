import actions as act
import numpy as np


def generate_square(self):
    white = 255
    img_size = 200
    img_matrix = generate_empty(img_size)
    for x in range(40, img_size-40):
        for y in range(40, img_size-40):
            img_matrix[x, y] = white

    print(img_matrix.size)

    act.load_image_on_canvas(self,img_matrix)

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

    act.load_image_on_canvas(self, img_matrix)


def generate_degrade(self):
    matrix = np.zeros((255, 100), dtype=np.uint8)
    for y in range(0, 255):
        for x in range(0, 100):
            matrix[y][x] = y

    print(matrix)
    act.load_image_on_canvas(self, matrix)


def generate_color_degrade(self):
    matrix = np.zeros((255*255*255))


def generate_empty(img_size):
    default_color = 0
    return np.full((img_size, img_size), default_color, dtype=np.uint8)

