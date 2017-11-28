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
import canny as canny
import susan as susan
import time
import video
import sift
from timeit import default_timer as timer


class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.minsize(width=640, height=480)
        master.title("Adove Fotoyop")
        self.release = True
        self.release2 = True

        self.next_canvas = 1
        self.image_matrix = []
        self.canvas = [Canvas(master, width=200, height=200, cursor="crosshair"),
                       Canvas(master, width=200, height=200, cursor="crosshair"),
                       Canvas(master, width=200, height=200, cursor="crosshair"),
                       Canvas(master, width=200, height=200, cursor="crosshair"),
                       Canvas(master, width=200, height=200, cursor="crosshair"),
                       Canvas(master, width=200, height=200, cursor="crosshair")]
        self.saved_image = None
        self.true_image = None
        menubar = Menu(master)

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=lambda: self.open(None,None))
        filemenu.add_command(label="Open Video", command=lambda: self.open_video())
        filemenu.add_command(label="Save", command=lambda: self.save(None))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Undo", command=self.undo, accelerator="Command+z")
        editmenu.add_command(label="Crop", command=lambda: actions.crop(self, master))
        editmenu.add_command(label="Rotate", command=lambda: self.apply_method(actions.rotate))
        editmenu.add_command(label="Grayscale", command=lambda: self.apply_method(actions.to_grayscale))
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
        gimagemenu.add_command(label="Circle", command=lambda: self.apply_method(gen.generate_circle))
        gimagemenu.add_command(label="Square", command=lambda: self.apply_method(gen.generate_square))
        gimagemenu.add_command(label="Degrade", command=lambda: self.apply_method(gen.generate_degrade))
        gimagemenu.add_command(label="Rayleigh", command=lambda: gen.generate_rayleigh(self))
        gimagemenu.add_command(label="Gauss", command=lambda: gen.generate_gaussian(self))
        gimagemenu.add_command(label="Exponential", command=lambda: gen.generate_exponential(self))
        gimagemenu.add_command(label="Salt and Pepper", command=lambda: gen.generate_SAP(self))
        menubar.add_cascade(label="Generate", menu=gimagemenu)

        filter_menu = Menu(menubar, tearoff=0)
        filter_menu.add_command(label="Mean", command=lambda: self.text_box(mesh.mean_filter, "Size", "Mean Filter"))
        filter_menu.add_command(label="Median", command=lambda: self.text_box(mesh.median_filter, "Size", "Median Filter"))
        filter_menu.add_command(label="Gauss", command=lambda: self.double_text_box(mesh.gauss_filter, "Size", "Sigma", "Gauss Filter"))
        filter_menu.add_command(label="Weighted Mean", command=lambda: self.apply_method(mesh.weighted_mean_filter, 3))
        filter_menu.add_command(label="Weighted Median", command=lambda: self.apply_method(mesh.weighted_median_filter, 3))
        filter_menu.add_command(label="High-Pass", command=lambda: self.apply_method(mesh.highpass_filter, 3))
        menubar.add_cascade(menu=filter_menu, label="Filters")

        border_menu = Menu(menubar, tearoff=0)
        border_menu.add_command(label="Prewit", command=lambda: self.apply_method(border.prewit))
        border_menu.add_command(label="Sobel", command=lambda: self.apply_method(border.sobel))
        border_menu.add_command(label="Multi Prewit", command=lambda: self.apply_method(border.multi_prewit))
        border_menu.add_command(label="Multi Sobel", command=lambda: self.apply_method(border.multi_sobel))
        border_menu.add_separator()
        border_menu.add_command(label="Laplace", command=lambda: self.apply_method(border.laplace))
        border_menu.add_command(label="Param Laplace", command=lambda: self.slider("Laplace", border.laplace))
        border_menu.add_command(label="Intelligent Laplace", command=lambda: self.apply_method(border.intelligent_laplace))
        border_menu.add_command(label="Laplace - Gauss",
                                command=lambda: self.text_box(border.laplace_gauss, "Sigma", 'Laplace - Gauss'))
        border_menu.add_separator()
        border_menu.add_command(label="Hough", command=lambda: self.double_text_box(border.hough, "a", "b", "Hough",self))
        border_menu.add_command(label="Contornos Activos", command=self.single_active_contours)
        border_menu.add_separator()
        border_menu.add_command(label="Harris", command=lambda: self.slider("Threshold", border.harris, "Harris"))
        menubar.add_cascade(menu=border_menu, label="Border")

        difansi = Menu(menubar, tearoff=0)
        difansi.add_command(label="Leclerc", command=lambda: self.double_text_box(anistropic.leclerc, "Iterations", "Sigma", "Leclerc"))
        difansi.add_command(label="Lorentziano", command=lambda: self.double_text_box(anistropic.lorentziano, "Iterations", "Sigma", "Lorentz"))
        difansi.add_command(label="Isotropic", command=lambda: self.text_box(anistropic.isotropic, "Iterations", "Isotropic"))
        menubar.add_cascade(menu=difansi, label="Dif")

        noise_menu = Menu(menubar, tearoff=0)
        noise_menu.add_command(label="Gaussian", command=lambda: actions.percentage_textbox(self, 'gaussian'))
        noise_menu.add_command(label="Rayleigh", command=lambda: actions.percentage_textbox(self, 'rayleigh'))
        noise_menu.add_command(label="Exponential", command=lambda: actions.percentage_textbox(self, 'exponential'))
        noise_menu.add_command(label="Salt & Pepper", command=lambda: actions.percentage_textbox(self, 'salt_and_pepper'))
        menubar.add_cascade(menu=noise_menu, label="Noise")

        threshold_menu = Menu(menubar, tearoff=0)
        threshold_menu.add_command(label="Global Threshold", command=lambda: self.apply_method(thresholds.global_threshold))
        threshold_menu.add_command(label="Threshold", command=lambda: self.slider("Umbral", thresholds.threshold, "Threshold"))
        threshold_menu.add_command(label="Otsu", command=lambda: self.apply_method(thresholds.otsu))
        menubar.add_cascade(menu=threshold_menu, label="Threshold")

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Canny", command=lambda: self.apply_method(canny.canny_function))
        filemenu.add_command(label="Susan All", command=lambda: self.apply_method(susan.susan_function))
        menubar.add_cascade(label="Detector", menu=filemenu)

        video_menu = Menu(menubar, tearoff=0)
        video_menu.add_command(label="Contornos Activos", command=self.active_contours)
        menubar.add_cascade(label="Video", menu=video_menu)

        sift_menu = Menu(menubar, tearoff=0)
        sift_menu.add_command(label="SIFT", command=lambda: sift.main_sift_window(self))
        menubar.add_cascade(label="Compare", menu=sift_menu)



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
        self.apply = Button(self.label_frame, text="Test", width=10, height=1)

        self.label_frame.grid(column=0, row=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        root.bind_all("<Command-z>", self.undo)
        root.bind_all("<Command-c>", self.move_canvas)
        root.bind_all("<Command-n>", self.open)
        root.bind_all("<Command-b>", lambda e: self.open(e, "./src/BARCO.RAW"))
        root.bind_all("<Command-l>", lambda e: self.open(e, "./src/LENA.RAW"))
        root.bind_all("<Command-s>", self.save)

        self.canvas[0].grid(column=0, row=0)
        self.canvas[0].true_image = []

    def open(self, event, name=None):
        if name is None:
            filename = filedialog.askopenfilename(parent=root)
        else:
            filename = name
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

        self.canvas[0].image = photo
        self.canvas[0].true_image = image
        self.true_image = image
        self.image_matrix = []
        self.image_matrix.append(np.array(image))
        width, height = image.size
        self.canvas[0].configure(width=width, height=height)
        self.image_on_canvas = self.canvas[0].create_image((0, 0), anchor="nw", image=photo)
        self.canvas[0].bind("<B1-Motion>", lambda e: self.set_area(e, 0, True))
        self.canvas[0].bind("<ButtonRelease-1>", lambda e: self.release_left(e,0, True))
        self.canvas[0].bind("<B2-Motion>", lambda e: self.set_area(e, 0, False))
        self.canvas[0].bind("<ButtonRelease-2>", lambda e: self.release_left(e, 0, False))

        for i in range(len(self.canvas)):
            self.canvas[i].rect = self.canvas[i].create_rectangle(-1, -1, -1, -1, fill='', outline='#ff0000')
            self.canvas[i].rect2 = self.canvas[i].create_rectangle(-1, -1, -1, -1, fill='', outline='#0000ff')

    def open_video(self):
        filename = filedialog.askopenfilename(parent=root)
        self.video = []
        i = 1
        name = filename.rsplit(".", 1)

        try:
            while True:
                print(name[0][:(len(name[0])-1)] + str(i) + "." + name[1])
                self.video.append(Image.open(name[0][:(len(name[0])-1)] + str(i) + "." + name[1]))
                i += 1
        except FileNotFoundError:
            print(i)
            print(len(self.video))

        self.canvas[0].image = ImageTk.PhotoImage(self.video[0])
        self.canvas[0].true_image = self.video[0]
        self.true_image = self.video[0]
        self.image_matrix = []
        self.image_matrix.append(np.array(self.video[0]))
        width, height = self.video[0].size
        self.canvas[0].configure(width=width, height=height)
        self.image_on_canvas = self.canvas[0].create_image((0, 0), anchor="nw", image=self.canvas[0].image)
        self.canvas[0].bind("<B1-Motion>", lambda e: self.set_area(e, 0, True))
        self.canvas[0].bind("<ButtonRelease-1>", lambda e: self.release_left(e, 0, True))
        self.canvas[0].bind("<B2-Motion>", lambda e: self.set_area(e, 0, False))
        self.canvas[0].bind("<ButtonRelease-2>", lambda e: self.release_left(e, 0, False))

        for i in range(len(self.canvas)):
            self.canvas[i].rect = self.canvas[i].create_rectangle(-1, -1, -1, -1, fill='', outline='#ffff00')
            self.canvas[i].rect2 = self.canvas[i].create_rectangle(-1, -1, -1, -1, fill='', outline='#0000ff')

    def save(self, event):
        filename = filedialog.asksaveasfilename(parent=root)
        image = Image.fromarray(actions.linear_transform(np.array(self.true_image)))
        image.save(filename)
        print(filename)

    def text_box(self, callback, text, name="Operation"):
        def save():
            self.forget()
            if self.entries[0].get() != "":
                self.load_on_canvas(callback(self.image_matrix[-1], int(self.entries[0].get())))

        def apply():
            self.true_image = Image.fromarray(callback(self.image_matrix[-1], int(self.entries[0].get())))
            self.canvas[0].image = ImageTk.PhotoImage(self.true_image)
            self.canvas[0].create_image((0, 0), anchor="nw", image=self.canvas[0].image)

        self.cancel_gui()
        self.saved_image = Image.fromarray(self.image_matrix[-1])
        self.set_label(text, 0)
        self.apply.grid(row=1, column=1)
        self.accept.grid(row=1, column=2)
        self.cancel.grid(row=1, column=0)
        self.apply.config(command=apply)
        self.accept.config(command=save)
        self.cancel.config(command=self.cancel_gui)
        self.label_frame.config(text=name)

    def double_text_box(self, callback, text1, text2, name="Operation", param1 = None):
        def save():
            if self.entries[0].get() != "" and self.entries[1].get() != "":
                if param1 is None:
                    self.load_on_canvas(callback(np.array(self.saved_image), int(self.entries[0].get()), int(self.entries[1].get())))
                elif param1 == self:
                    callback(np.array(self.saved_image), int(self.entries[0].get()), int(self.entries[1].get()), param1)

            self.forget()

        def apply():
            if param1 is None:
                self.true_image = Image.fromarray(
                    callback(np.array(self.saved_image), int(self.entries[0].get()), int(self.entries[1].get())))
                self.canvas[0].image = ImageTk.PhotoImage(self.true_image)
                self.canvas[0].create_image((0, 0), anchor="nw", image=self.canvas[0].image)
            elif param1 == self:
                    callback(np.array(self.saved_image), int(self.entries[0].get()), int(self.entries[1].get()), param1)

        self.cancel_gui()
        self.saved_image = self.canvas[0].true_image
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

    def slider(self, text, callback, name="Operation"):
        def save():
            self.load_on_canvas(np.asarray(self.true_image))
            self.forget()

        def apply(event):
            self.true_image = Image.fromarray(
                callback(np.array(self.saved_image), int(self.scales[0].get())))
            self.canvas[0].image = ImageTk.PhotoImage(self.true_image)
            self.canvas[0].create_image((0, 0), anchor="nw", image=self.canvas[0].image)

        self.cancel_gui()
        self.saved_image = self.canvas[0].true_image
        self.set_slider(0, text, apply)
        self.accept.grid(row=2, column=0)
        self.cancel.grid(row=2, column=1)
        self.accept.config(command=save)
        self.cancel.config(command=self.cancel_gui)
        self.label_frame.config(text=name)

    def double_slider(self, text1, text2, callback, name="Operation"):
        def save():
            self.load_on_canvas(np.asarray(self.true_image))
            self.forget()

        def apply(event):
            self.true_image = Image.fromarray(
                callback(np.array(self.saved_image), self.scales[0].get(), self.scales[1].get()))
            self.canvas[0].image = ImageTk.PhotoImage(self.true_image)
            self.canvas[0].create_image((0, 0), anchor="nw", image=self.canvas[0].image)

        self.cancel_gui()
        self.saved_image = self.canvas[0].true_image
        self.set_slider(0, text1, apply)
        self.set_slider(1, text2, apply)
        self.accept.grid(row=2, column=0)
        self.cancel.grid(row=2, column=1)
        self.accept.config(command=save)
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
            self.load_on_canvas(method(np.array(self.canvas[0].true_image)))
        elif param == self:
            method(np.array(self.canvas[0].true_image), param)
        else:
            self.load_on_canvas(method(np.array(self.canvas[0].true_image), param))

    def active_contours(self):
        # avg = actions.get_area_info(self, np.array(self.canvas[0].true_image))
        self.canvas[0].coords(self.canvas[0].rect, -1, -1, -1, -1)
        self.canvas[0].coords(self.canvas[0].rect2, -1, -1, -1, -1)
        matrixes = []
        self.canvas[0].rect
        for i in range(len(self.video)):
            matrixes.append(np.array(self.video[i], dtype=np.uint8))

        start = timer()
        phi, lin, lout, theta0, theta1 = video.pixel_exchange_border_detect(matrixes[0],
                                                                            (self.x_start, self.x_finish, self.y_start, self.y_finish),
                                                                            (self.x2_start, self.x2_finish, self.y2_start, self.y2_finish))
        end = timer()
        print("First time: " + str(end - start))
        self.paint_pixels(matrixes[0], lout)
        start = timer()
        for i in range(1, len(self.video)):

            phi, lin, lout, theta0, theta1 = video.apply_pixel_exchange(matrixes[i], phi, lin, lout, theta0, theta1)
            self.paint_pixels(matrixes[i], lout)
        end = timer()
        print("Others: " + str(end - start))

    def single_active_contours(self):
        self.canvas[0].coords(self.canvas[0].rect, -1, -1, -1, -1)
        self.canvas[0].coords(self.canvas[0].rect2, -1, -1, -1, -1)

        start = timer()
        phi, lin, lout, theta0, theta1 = video.pixel_exchange_border_detect(np.array(self.canvas[0].true_image),
                                                                            (self.x_start, self.x_finish, self.y_start,
                                                                             self.y_finish),
                                                                            (self.x2_start, self.x2_finish,
                                                                             self.y2_start, self.y2_finish))
        end = timer()
        print(end - start)
        self.paint_pixels(np.array(self.canvas[0].true_image), lout)

    def paint_pixels(self, matrix, pixels):
        for i in pixels:
            matrix[i[0], i[1]] = (0, 0, 255)
        self.canvas[0].ti = Image.fromarray(matrix)
        self.canvas[0].i = ImageTk.PhotoImage(self.canvas[0].ti)
        self.canvas[0].itemconfig(self.image_on_canvas, image=self.canvas[0].i)
        self.canvas[0].update()


    def cancel_gui(self):
        if self.saved_image is not None:
            self.true_image = self.saved_image
            self.canvas[0].image = ImageTk.PhotoImage(self.true_image)
            self.canvas[0].create_image((0, 0), anchor="nw", image=self.canvas[0].image)
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
        self.saved_image = None
        print("Forgotten")

    def load_on_canvas(self, matrix):
        self.image_matrix += [matrix]
        self.canvas[0].true_image = Image.fromarray(self.image_matrix[-1])
        self.canvas[0].image = ImageTk.PhotoImage(self.canvas[0].true_image)
        self.canvas[0].create_image((0, 0), anchor="nw", image=self.canvas[0].image)

    def undo(self,event = None):
        self.image_matrix.pop()
        self.canvas[0].true_image = Image.fromarray(self.image_matrix[-1])
        self.canvas[0].image = ImageTk.PhotoImage(self.canvas[0].true_image)
        self.canvas[0].create_image((0, 0), anchor="nw", image=self.canvas[0].image)

    def move_canvas(self,event):
        self.canvas[self.next_canvas].true_image = Image.fromarray(self.image_matrix[-1])
        self.canvas[self.next_canvas].image = ImageTk.PhotoImage(self.canvas[self.next_canvas].true_image)
        self.canvas[self.next_canvas].create_image((0, 0), anchor="nw", image=self.canvas[self.next_canvas].image)
        self.canvas[self.next_canvas].grid(row=0, column=self.next_canvas)
        self.canvas[self.next_canvas].configure(width=self.image_matrix[-1].shape[1], height=self.image_matrix[-1].shape[0])
        index = self.next_canvas
        self.canvas[self.next_canvas].bind("<B1-Motion>", lambda e: self.set_area(e, index))
        self.canvas[self.next_canvas].bind("<ButtonRelease-1>", lambda e: self.release_left(e,index))
        self.next_canvas += 1

    def set_area(self, event, index, left):
        if(left):
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

            self.canvas[index].coords(self.canvas[index].rect, self.x_start, self.y_start, self.x_finish, self.y_finish)
            self.canvas[index].tag_raise(self.canvas[index].rect)
        else:
            if self.release2:
                self.x2_start = event.x
                self.x2_finish = event.x
            else:
                self.x2_finish = event.x

            if self.release2:
                self.y2_start = event.y
                self.y2_finish = event.y
                self.release2 = False
            else:
                self.y2_finish = event.y

            self.canvas[index].coords(self.canvas[index].rect2, self.x2_start, self.y2_start, self.x2_finish, self.y2_finish)
            self.canvas[index].tag_raise(self.canvas[index].rect2)

    def release_left(self,event,index, left):
        if left:
            self.release = True
            actions.get_area_info(self, np.array(self.canvas[index].true_image),left)
        else:
            self.release2 = True
            actions.get_area_info(self, np.array(self.canvas[index].true_image),left)
root = Tk()
my_gui = MyFirstGUI(root)

root.iconbitmap('src/ati.ico')
my_gui.release = True
root.mainloop()
