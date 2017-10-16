from tkinter import Tk, Entry, Scale, Label, LabelFrame, Button, Menu, filedialog, Canvas
from PIL import Image, ImageTk
import actions as actions
import generation as gen
import border as border
import json
import numpy as np
import meshoperations as mesh
import anistropic
import thresholds


class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.minsize(width=640, height=480)
        master.title("Adove Fotoyop")

        self.canvas = Canvas(master, width=200, height=200, cursor="crosshair")
        self.saved_image = None
        self.true_image = None
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
        editmenu.add_command(label="Operations", command=lambda: actions.opr(self))
        editmenu.add_command(label="Negative", command=lambda: actions.to_negative(self))
        editmenu.add_command(label="Scalar Mult", command=lambda: actions.scalar_mult_textbox(self))
        editmenu.add_command(label="Gamma", command=lambda: actions.gamma_textbox(self))
        editmenu.add_command(label="Din. Range", command=lambda: actions.din_range(self, None))
        menubar.add_cascade(label="Edit", menu=editmenu)

        datamenu = Menu(menubar, tearoff=0)
        datamenu.add_command(label="Histogram", command=lambda: actions.show_hist(self))
        datamenu.add_command(label="Equalize", command=lambda: actions.equalize(self))
        datamenu.add_command(label="Contrast",
                             command=lambda: self.double_slider("S1", "S2", actions.contrast, "Contrast"))
        menubar.add_cascade(menu=datamenu, label="Data")

        gimagemenu = Menu(menubar, tearoff=0)
        gimagemenu.add_command(label="Circle", command=lambda: gen.generate_circle(self))
        gimagemenu.add_command(label="Square", command=lambda: gen.generate_square(self))
        gimagemenu.add_command(label="Degrade", command=lambda: gen.generate_degrade(self))
        gimagemenu.add_command(label="Rayleigh", command=lambda: gen.generate_rayleigh(self))
        gimagemenu.add_command(label="Gauss", command=lambda: gen.generate_gaussian(self))
        gimagemenu.add_command(label="Exponential", command=lambda: gen.generate_exponential(self))
        gimagemenu.add_command(label="Salt and Peper", command=lambda: gen.generate_SAP(self))
        menubar.add_cascade(label="Generate", menu=gimagemenu)

        filter_menu = Menu(menubar, tearoff=0)
        filter_menu.add_command(label="Mean", command=lambda: self.text_box(mesh.mean_filter, "Size", "Mean Filter"))
        filter_menu.add_command(label="Median",
                                command=lambda: self.text_box(mesh.median_filter, "Size", "Median Filter"))
        filter_menu.add_command(label="Gauss", command=lambda: self.double_text_box(mesh.gauss_filter, "Size", "Sigma",
                                                                                    "Gauss Filter"))
        filter_menu.add_command(label="Weighted Mean", command=lambda: self.apply_method(mesh.weighted_mean_filter, 3))
        filter_menu.add_command(label="Weighted Median",
                                command=lambda: self.apply_method(mesh.weighted_median_filter, 3))
        filter_menu.add_command(label="High-Pass", command=lambda: self.apply_method(mesh.highpass_filter, 3))
        menubar.add_cascade(menu=filter_menu, label="Filters")

        border_menu = Menu(menubar, tearoff=0)
        border_menu.add_command(label="Prewit", command=lambda: self.border(border.prewit))
        border_menu.add_command(label="Sobel", command=lambda: self.border(border.sobel))
        border_menu.add_command(label="Multi Prewit", command=lambda: self.border(border.multi_prewit))
        border_menu.add_command(label="Multi Sobel", command=lambda: self.border(border.multi_sobel))
        border_menu.add_command(label="Laplace", command=lambda: self.border(border.laplace))
        border_menu.add_command(label="Param Laplace", command=lambda: self.slider("Laplace", border.laplace))
        border_menu.add_command(label="Intelligent Laplace", command=lambda: self.border(border.intelligent_laplace))

        border_menu.add_command(label="Laplace - Gauss",
                                command=lambda: self.text_box(border.laplace_gauss, "Sigma", 'Laplace - Gauss'))
        menubar.add_cascade(menu=border_menu, label="Border")

        difansi = Menu(menubar, tearoff=0)
        difansi.add_command(label="Leclerc",
                            command=lambda: self.double_text_box(anistropic.leclerc, "Iterations", "Sigma", "Leclerc"))
        difansi.add_command(label="Lorentziano",
                            command=lambda: self.double_text_box(anistropic.lorentziano, "Interations", "Sigma",
                                                                 "Lorentz"))
        difansi.add_command(label="Isotropic",
                            command=lambda: self.text_box(anistropic.isotropic, "Iterations", "Isotropic"))
        menubar.add_cascade(menu=difansi, label="Dif")

        noise_menu = Menu(menubar, tearoff=0)
        noise_menu.add_command(label="Gaussian", command=lambda: actions.percentage_textbox(self, 'gaussian'))
        noise_menu.add_command(label="Rayleigh", command=lambda: actions.percentage_textbox(self, 'rayleigh'))
        noise_menu.add_command(label="Exponential", command=lambda: actions.percentage_textbox(self, 'exponential'))
        noise_menu.add_command(label="Salt & Pepper",
                               command=lambda: actions.percentage_textbox(self, 'salt_and_pepper'))
        menubar.add_cascade(menu=noise_menu, label="Noise")

        threshold_menu = Menu(menubar, tearoff=0)
        threshold_menu.add_command(label="Global Threshold",
                                   command=lambda: self.apply_method(thresholds.global_threshold))
        threshold_menu.add_command(label="Threshold",
                                   command=lambda: self.slider("Umbral", thresholds.threshold, "Threshold"))
        threshold_menu.add_command(label="Otsu", command=lambda: self.apply_method(thresholds.otsu))
        menubar.add_cascade(menu=threshold_menu, label="Threshold")

        master.config(menu=menubar)
        self.menubar = menubar

        self.label_frame = LabelFrame(self.master, text="Operation")
        self.labels = [Label(self.label_frame), Label(self.label_frame), Label(self.label_frame),
                       Label(self.label_frame)]
        self.entries = [Entry(self.label_frame), Entry(self.label_frame), Entry(self.label_frame),
                        Entry(self.label_frame)]
        self.scales = [Scale(self.label_frame, from_=0, to=255, orient="h", length=255),
                       Scale(self.label_frame, from_=0, to=255, orient="h", length=255),
                       Scale(self.label_frame, from_=0, to=255, orient="h", length=255),
                       Scale(self.label_frame, from_=0, to=255, orient="h", length=255)]
        self.accept = Button(self.label_frame, text="OK", width=10, height=1)
        self.cancel = Button(self.label_frame, text="Cancel", width=10, height=1)
        self.apply = Button(self.label_frame, text="Apply", width=10, height=1)

        self.label_frame.grid(column=0, row=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

    def open(self):

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

        def release_left():
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
        self.true_image = image
        width, height = image.size
        self.canvas.configure(width=width, height=height)
        self.canvas.create_image((0, 0), anchor="nw", image=photo)
        self.canvas.bind("<B1-Motion>", set_area)
        self.canvas.bind("<ButtonRelease-1>", release_left)
        self.canvas.grid(column=0, row=0)
        self.canvas.rect = self.canvas.create_rectangle(-1, -1, -1, -1, fill='', outline='#ff0000')

    def save(self):
        filename = filedialog.asksaveasfilename(parent=root)
        image = Image.fromarray(actions.linear_transform(np.array(self.true_image)))
        image.save(filename)
        print(filename)

    def text_box(self, callback, text, name="Operation"):
        def save():
            self.forget()
            if self.entries[0].get() != "":
                self.true_image = Image.fromarray(callback(np.array(self.saved_image), int(self.entries[0].get())))
                self.canvas.image = ImageTk.PhotoImage(self.true_image)
                self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

        def apply():
            self.true_image = Image.fromarray(callback(np.array(self.saved_image), int(self.entries[0].get())))
            self.canvas.image = ImageTk.PhotoImage(self.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

        self.cancel_gui()
        self.saved_image = self.true_image
        self.set_label(text, 0)
        self.apply.grid(row=1, column=1)
        self.accept.grid(row=1, column=2)
        self.cancel.grid(row=1, column=0)
        self.apply.config(command=apply)
        self.accept.config(command=save)
        self.cancel.config(command=self.cancel_gui)
        self.label_frame.config(text=name)

    def double_text_box(self, callback, text1, text2, name="Operation"):
        def save():
            self.forget()
            if self.entries[0].get() != "" and self.entries[1].get() != "":
                self.true_image = Image.fromarray(
                    callback(np.array(self.saved_image), int(self.entries[0].get()), int(self.entries[1].get())))
                self.canvas.image = ImageTk.PhotoImage(self.true_image)
                self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

        def apply():
            self.true_image = Image.fromarray(
                callback(np.array(self.saved_image), int(self.entries[0].get()), int(self.entries[1].get())))
            self.canvas.image = ImageTk.PhotoImage(self.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

        self.cancel_gui()
        self.saved_image = self.true_image
        self.set_label(text1, 0)
        self.set_label(text2, 1)
        self.apply.grid(row=2, column=1)
        self.accept.grid(row=2, column=2)
        self.cancel.grid(row=2, column=0)
        self.apply.config(command=apply)
        self.accept.config(command=save)
        self.cancel.config(command=self.cancel_gui)
        self.label_frame.config(text=name)

    def set_label(self, text, position):
        self.labels[position].config(text=text)
        self.labels[position].grid(row=position, column=0)
        self.entries[position].grid(row=position, column=1)

    def border(self, callback):
        self.true_image = Image.fromarray(callback(np.array(self.true_image)))
        self.canvas.image = ImageTk.PhotoImage(self.true_image)
        self.canvas.create_image((0, 0), image=self.canvas.image, anchor="nw")

    def slider(self, text, callback, name="Operation"):
        def apply(event):
            self.true_image = Image.fromarray(callback(np.array(self.saved_image), self.scales[0].get()))
            self.canvas.image = ImageTk.PhotoImage(self.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

        self.cancel_gui()
        self.saved_image = self.true_image
        self.set_slider(0, text, apply)
        self.accept.grid(row=2, column=0)
        self.cancel.grid(row=2, column=1)
        self.accept.config(command=self.forget)
        self.cancel.config(command=self.cancel_gui)
        self.label_frame.config(text=name)

    def double_slider(self, text1, text2, callback, name="Operation"):
        def apply(event):
            self.true_image = Image.fromarray(
                callback(np.array(self.saved_image), self.scales[0].get(), self.scales[1].get()))
            self.canvas.image = ImageTk.PhotoImage(self.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

        self.cancel_gui()
        self.saved_image = self.true_image
        self.set_slider(0, text1, apply)
        self.set_slider(1, text2, apply)
        self.accept.grid(row=2, column=0)
        self.cancel.grid(row=2, column=1)
        self.accept.config(command=self.forget)
        self.cancel.config(command=self.cancel_gui)
        self.label_frame.config(text=name)

    def set_slider(self, position, text, func):
        self.labels[position].config(text=text)
        self.labels[position].grid(row=0, column=position * 2, columnspan=2)
        self.scales[position].bind("<ButtonRelease-1>", func)
        self.scales[position].set(128)
        self.scales[position].grid(row=1, column=position * 2, columnspan=2)

    def apply_method(self, method, param=-1):
        if param == -1:
            self.true_image = Image.fromarray(method(np.array(self.true_image)))
        else:
            self.true_image = Image.fromarray(method(np.array(self.true_image), param))

        self.canvas.image = ImageTk.PhotoImage(self.true_image)
        self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)

    def cancel_gui(self):
        if self.saved_image is not None:
            self.true_image = self.saved_image
            self.canvas.image = ImageTk.PhotoImage(self.true_image)
            self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)
        self.forget()

    def forget(self):
        for e in self.labels:
            e.grid_forget()
        for e in self.entries:
            e.grid_forget()
        for e in self.scales:
            e.grid_forget()

        self.cancel.grid_forget()
        self.accept.grid_forget()
        self.apply.grid_forget()
        print("Forgotten")


root = Tk()
my_gui = MyFirstGUI(root)

root.iconbitmap('src/ati.ico')
my_gui.release = True
root.mainloop()
