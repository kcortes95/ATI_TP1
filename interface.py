from tkinter import Tk, Entry, Label, Button, messagebox, Menu, filedialog, Canvas
from PIL import Image, ImageTk
import numpy as np
import math
import json


class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.minsize(width=640, height=480)
        master.title("Best Image Editor EVAR")

        #self.label = Label(master, text="This is our first GUI!")
        #self.label.pack()
        self.color_text = Entry()
        self.color_text.pack()

        self.x_text = Entry()
        self.x_text.pack()

        self.y_text = Entry()
        self.y_text.pack()

        self.set_button = Button(master, text="Set Pixel", command=self.set_color)
        self.set_button.pack()

        #self.close_button = Button(master, text="Close", command=master.quit)
        #self.close_button.pack()

        self.canvas = Canvas(master, width=200, height=200)


        menubar = Menu(master)

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open)
        filemenu.add_command(label="Save", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Crop", command=self.crop)
        filemenu.add_separator()
        menubar.add_cascade(label="Edit", menu=editmenu)

        gimagemenu = Menu(menubar, tearoff=0)
        gimagemenu.add_command(label="Circle", command=self.generate_circle)
        gimagemenu.add_command(label="Square", command=self.generate_square)
        menubar.add_cascade(label="Images", menu=gimagemenu)

        master.config(menu=menubar)
        self.menubar = menubar;

    def crop(self):
        self.new_window = Tk()
        self.new_window.minsize(width=640,height=480)
        self.new_window.config(menu=self.menubar)
        img = self.canvas.true_image.load()
        print(img)
        new_image = np.zeros((self.y_finish- self.y_start, self.x_finish - self.x_start, 1), dtype=np.uint8)
        k = 0
        l = 0
        for i in range(self.x_start,self.x_finish):
            for j in range(self.y_start, self.y_finish):
                print(str(i) + " " + str(j))
                aux = img[i,j]
                new_image[k,l] = aux
                l = l+1
            k = k+1
            l = 0;
        print(new_image)
        true_croped = Image.fromarray(new_image, 'L')
        cropped = ImageTk.PhotoImage(true_cropped)
        self.new_window.canvas = Canvas(master, width=200, height=200)
        self.new_window.canvas.true_cropped = true_cropped
        self.new_window.canvas.configure(width=true_cropped.width,height=true_cropped.height)
        self.new_window.canvas.create_image((0,0),anchor="nw",image=cropped)

    def set_color(self):
        color = int(self.color_text.get())
        self.canvas.true_image.putpixel((int(self.x_text.get()), int(self.y_text.get())), color)
        self.canvas.image = ImageTk.PhotoImage(self.canvas.true_image)
        self.canvas.create_image((0,0),anchor="nw",image=self.canvas.image)

    def greet(self):
        messagebox.showinfo("hola","kevin");

    def hello(self):
        print("hello")

    def open(self):
        def select_pixel(event):
            print(self.canvas.true_image.getpixel((event.x,event.y)))

        def set_pixel(event):
            self.x_text.delete(0,len(self.x_text.get()))
            self.y_text.delete(0,len(self.y_text.get()))
            self.x_text.insert(0,event.x)
            self.y_text.insert(0,event.y)

        def set_area(event):

            print(str(event.x) + " " + str(event.y))
            if self.release :
                self.x_start = event.x
            else:
                self.x_finish = event.x

            if self.release :
                self.y_start = event.y
                self.release = False;
            else:
                self.y_finish = event.y

            self.canvas.coords(self.canvas.rect,self.x_start,self.y_start,self.x_finish,self.y_finish);
        def release_left(event):
             self.release = True;

        filename = filedialog.askopenfilename(parent=root)
        print(filename)
        if filename.find("RAW") != -1 :
            with open('raw.json') as json_data:
                d = json.load(json_data)
            dim = d['data'][filename.rsplit(".",1)[0].rsplit("/",1)[1]]
            print(dim['x'])
            print(dim['y'])
            image = Image.frombytes('F',(dim['x'],dim['y']),open(filename,"rb").read(),'raw','F;8')
            photo = ImageTk.PhotoImage(image)
        else:
            image = Image.open(filename)
            photo = ImageTk.PhotoImage(image)

        self.canvas.image = photo;
        self.canvas.true_image = image;
        width,height = image.size
        self.canvas.configure(width=width,height=height)
        self.canvas.create_image((0,0),anchor="nw",image=photo)
        self.canvas.bind("<Button-1>", select_pixel)
        self.canvas.bind("<Button-3>", set_pixel)
        self.canvas.bind("<B1-Motion>",set_area)
        self.canvas.bind("<ButtonRelease-1>", release_left)
        self.canvas.pack();
        self.canvas.rect = self.canvas.create_rectangle(-1,-1,-1,-1,fill='')

        print("DONE")


    def save(self):
        filename = filedialog.asksaveasfilename(parent = root)
        self.canvas.true_image.save(filename)
        print(filename)

    #http://www.programcreek.com/python/example/57106/Image.frombytes
    def toImage(arr):
        if arr.type().bytes == 1:
            im = Image.frombytes('L', arr.shape[::-1], arr.tostring())
        else:
            arr_c = arr - arr.min()
            arr_c *= (255./arr_c.max())
            arr = arr_c.astype(UInt8)
            im = Image.frombytes('L', arr.shape[::-1], arr.tostring())
        return im

    def generate_square(self):
        img_size = 200
        default_color = 0
        black = 0
        white = 255

        # img_matrix = [[default_color] * img_size for i in range(img_size)]
        #
        # for x in range(img_size):
        #     for y in range(img_size):
        #         if (y==0 or y == (img_size-1)):
        #             img_matrix[x][y] = white
        #
        #         if (x==0 or x == (img_size-1)):
        #             img_matrix[x][y] = white

        img_array = [0] * (img_size * img_size)

        for i in range(img_size * img_size):
            x = math.floor(i/img_size)
            y = i%img_size

            if (y==0 or y == (img_size-1)):
                img_matrix[i] = white

            if (x==0 or x == (img_size-1)):
                img_matrix[i] = white

        img_ret = toImage(img_array)
        #Luego, guardar en RAW la imagen retornada por la funcion
        print("DONE")


    def generate_circle(self):
        print("TO DO")

root = Tk()
my_gui = MyFirstGUI(root)
my_gui.release = True;
root.mainloop()
