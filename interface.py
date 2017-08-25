from tkinter import Tk, Entry, Scale, Label, LabelFrame, Button, messagebox, Menu, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk

import actions as actions
import generation as gen
import json
import numpy as np

class MyFirstGUI:

    def __init__(self, master):
        self.master = master
        master.minsize(width=640, height=480)
        master.title("Adove Fotoyop")

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
        editmenu.add_separator()
        editmenu.add_command(label="Area Info", command=lambda: actions.get_area_info(self, master))
        editmenu.add_command(label="To HSV", command=lambda: actions.rgb_to_hsv(self))
        editmenu.add_separator()
        editmenu.add_command(label="Scalar Mult", command=lambda: actions.scalar_mult_textbox(self))
        menubar.add_cascade(label="Edit", menu=editmenu)

        datamenu = Menu(menubar,tearoff=0)
        datamenu.add_command(label="Histogram", command=lambda: actions.show_hist(self))
        datamenu.add_command(label="Threshold", command=lambda: self.umbral(master))
        datamenu.add_command(label="Equalize", command=lambda: actions.equalize(self))
        datamenu.add_command(label="Contrast",command=lambda: self.contrast(master))
        menubar.add_cascade(menu=datamenu, label="Data")

        image_menu = Menu(menubar,tearoff=0)
        image_menu.add_command(label="Add", command=lambda: actions.add(self))
        image_menu.add_command(label="Supr", command=lambda: actions.supr(self))
        image_menu.add_command(label="Mult", command=lambda: actions.mult(self))
        editmenu.add_cascade(menu=image_menu,label="Image")

        gimagemenu = Menu(menubar, tearoff=0)
        gimagemenu.add_command(label="Circle", command=lambda: gen.generate_circle(self))
        gimagemenu.add_command(label="Square", command=lambda: gen.generate_square(self))
        gimagemenu.add_command(label="Degrade", command=lambda: gen.generate_degrade(self))
        menubar.add_cascade(label="Generate", menu=gimagemenu)

        filter_menu = Menu(menubar, tearoff=0)
        filter_menu.add_command(label="Mean", command=lambda: actions.mean_filter(self, 5))
        filter_menu.add_command(label="Weighted Mean", command=lambda: actions.weighted_mean_filter(self, 3))
        filter_menu.add_command(label="Median", command=lambda: actions.median_filter(self, 3))
        filter_menu.add_command(label="Gauss", command=lambda: actions.gauss_filter(self, 41))
        filter_menu.add_command(label="High-Pass", command=lambda: actions.highpass_filter(self, 3))

        menubar.add_cascade(menu=filter_menu, label="Filters")

        master.config(menu=menubar)
        self.menubar = menubar
        self.label_frame = LabelFrame(master, text="Operation")
        self.label_frame.grid(column=0, row=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
    def open(self):

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
            self.canvas.tag_raise(self.canvas.rect)

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
        self.canvas.bind("<Button-3>", set_pixel)
        self.canvas.bind("<B1-Motion>", set_area)
        self.canvas.bind("<ButtonRelease-1>", release_left)
        self.canvas.grid(column=0,row=0)
        self.canvas.rect = self.canvas.create_rectangle(-1, -1, -1, -1, fill='', outline='#ff0000')

    def save(self):
        filename = filedialog.asksaveasfilename(parent=root)
        self.canvas.true_image.save(filename)
        print(filename)

    def umbral(self, master):
        def set_umbral(event):
            self.canvas.true_image = Image.fromarray(actions.umbral(np.array(self.canvas.saved_image), self.w.get()))
            self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

        def save():
            self.w.grid_forget()
            self.cancel.grid_forget()
            self.accept.grid_forget()

        def cancel():
            self.canvas.true_image = self.canvas.saved_image
            self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)
            self.w.grid_forget()
            self.cancel.grid_forget()
            self.accept.grid_forget()

        self.canvas.saved_image = self.canvas.true_image
        w = Scale(self.label_frame, from_=0, to=255, orient="h")
        w.bind("<ButtonRelease-1>", set_umbral)
        w.set(128)
        w.grid(row=0, column=0,columnspan=2)
        self.accept = Button(self.label_frame, text="OK", width=10, height=1, command=save)
        self.accept.grid(row=1, column=0)
        self.cancel = Button(self.label_frame, text="Cancel", width=10, height=1, command=cancel)
        self.cancel.grid(row=1, column=1)
        self.w = w

    def contrast(self, master):
        def set_contrast(event):
            mat = actions.contrast(self, self.canvas.saved_image, self.w1.get(), self.w2.get())
            self.canvas.true_image = Image.fromarray(mat)
            self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

        def save():
            self.w1.grid_forget()
            self.w2.grid_forget()
            self.cancel.grid_forget()
            self.accept.grid_forget()

        def cancel():
            self.canvas.true_image = self.canvas.saved_image
            self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)
            self.w1.grid_forget()
            self.w2.grid_forget()
            self.cancel.grid_forget()
            self.accept.grid_forget()

        self.canvas.saved_image = self.canvas.true_image
        w1 = Scale(self.label_frame, from_=0, to=255, orient="h", label="S1")
        w1.bind("<ButtonRelease-1>", set_contrast)
        w1.set(128)
        w1.grid(row=0, column=0)
        w2 = Scale(self.label_frame, from_=0, to=255, orient="h", label="S2")
        w2.bind("<ButtonRelease-1>", set_contrast)
        w2.set(128)
        w2.grid(row=0, column=1)
        self.w1 = w1
        self.w2 = w2
        self.accept = Button(self.label_frame, text="Ok", width=10, height=1, command=save)
        self.accept.grid(row=1, column=0)
        self.cancel = Button(self.label_frame, text="Cancel", width=10, height=1, command=cancel)
        self.cancel.grid(row=1, column=1)
        self.label_frame.text = "Contrast"


root = Tk()
my_gui = MyFirstGUI(root)
icon = PhotoImage(file='ati.gif')
root.tk.call('wm', 'iconphoto', root._w, icon)
my_gui.release = True
root.mainloop()
