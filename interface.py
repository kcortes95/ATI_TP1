from tkinter import Tk, Entry, Label, Button, messagebox, Menu, filedialog, Canvas
from PIL import Image, ImageTk
import numpy as np
import actions as actions
import mouse as mouse
import math
import json


class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.minsize(width=640, height=480)
        master.title("Best Image Editor EVAR")

        # self.label = Label(master, text="This is our first GUI!")
        # self.label.pack()
        self.color_text = Entry()
        self.color_text.pack()

        self.x_text = Entry()
        self.x_text.pack()

        self.y_text = Entry()
        self.y_text.pack()

        self.set_button = Button(master, text="Set Pixel", command=lambda: actions.set_color(self))
        self.set_button.pack()

        self.set_button = Button(master, text="Clear", command=lambda: actions.do_clear(self))
        self.set_button.pack()

        # self.close_button = Button(master, text="Close", command=master.quit)
        # self.close_button.pack()

        self.canvas = Canvas(master, width=200, height=200, cursor="crosshair")

        menubar = Menu(master)

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open)
        filemenu.add_command(label="Save", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Crop", command=lambda: actions.crop(self, master))
        filemenu.add_separator()
        editmenu.add_command(label="Area Info", command=lambda: actions.get_area_info(self, master))
        editmenu.add_command(label="To HSV", command=lambda: actions.rgb_to_hsv(self))
        menubar.add_cascade(label="Edit", menu=editmenu)

        gimagemenu = Menu(menubar, tearoff=0)
        gimagemenu.add_command(label="Circle", command=lambda: actions.generate_circle(self))
        gimagemenu.add_command(label="Square", command=lambda: actions.generate_square(self))
        gimagemenu.add_command(label="Degrade", command=lambda: actions.generate_degrade(self))
        menubar.add_cascade(label="Images", menu=gimagemenu)

        master.config(menu=menubar)
        self.menubar = menubar;

    def greet(self):
        messagebox.showinfo("hola", "kevin");

    def hello(self):
        print("hello")

    def open(self):
        # def select_pixel(event):
            # print(self.canvas.true_image.getpixel((event.x, event.y)))

        def set_pixel(event):
            self.x_text.delete(0, len(self.x_text.get()))
            self.y_text.delete(0, len(self.y_text.get()))
            self.x_text.insert(0, event.x)
            self.y_text.insert(0, event.y)

        def set_area(event):
            if self.release:
                self.x_start = event.x
                self.x_finish = event.x
            else:
                self.x_finish = event.x

            if self.release:
                self.y_start = event.y
                self.y_finish = event.y
                self.release = False
            else:
                self.y_finish = event.y

            self.canvas.coords(self.canvas.rect, self.x_start, self.y_start, self.x_finish, self.y_finish)

        def release_left(event):
            self.release = True

        filename = filedialog.askopenfilename(parent=root)
        print(filename)
        if filename.find("RAW") != -1:
            with open('raw.json') as json_data:
                d = json.load(json_data)
            dim = d['data'][filename.rsplit(".", 1)[0].rsplit("/", 1)[1]]
            print(dim['x'])
            print(dim['y'])
            image = Image.frombytes('F', (dim['x'], dim['y']), open(filename, "rb").read(), 'raw', 'F;8')
            photo = ImageTk.PhotoImage(image)
        else:
            image = Image.open(filename)
            photo = ImageTk.PhotoImage(image)

        self.canvas.image = photo
        self.canvas.true_image = image
        width, height = image.size
        self.canvas.configure(width=width, height=height)
        self.canvas.create_image((0, 0), anchor="nw", image=photo)
        # self.canvas.bind("<Button-1>", select_pixel)
        self.canvas.bind("<Button-3>", set_pixel)
        self.canvas.bind("<B1-Motion>", set_area)
        self.canvas.bind("<ButtonRelease-1>", release_left)
        self.canvas.pack()
        self.canvas.rect = self.canvas.create_rectangle(-1, -1, -1, -1, fill='', outline='#ff0000')

        print("DONE")

    def save(self):
        filename = filedialog.asksaveasfilename(parent=root)
        self.canvas.true_image.save(filename)
        print(filename)

root = Tk()
my_gui = MyFirstGUI(root)
my_gui.release = True
root.mainloop()
