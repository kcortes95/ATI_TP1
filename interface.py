from tkinter import Tk, Label, Button, messagebox, Menu, filedialog, Canvas
from PIL import Image, ImageTk


class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.minsize(width=640, height=480)
        master.title("A simple GUI")

        #self.label = Label(master, text="This is our first GUI!")
        #self.label.pack()

        #self.greet_button = Button(master, text="Greet", command=self.greet)
        #self.greet_button.pack()

        #self.close_button = Button(master, text="Close", command=master.quit)
        #self.close_button.pack()

        self.canvas = Canvas(master, width=200, height=200)


        menubar = Menu(master)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open)
        filemenu.add_command(label="Save", command=self.hello)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        master.config(menu=menubar)


    def greet(self):
        messagebox.showinfo("hola","kevin");

    def hello(self):
        print("hello")

    def open(self):
        filename = filedialog.askopenfilename(parent=root)
        print(filename)
        image = Image.open(filename)
        photo = ImageTk.PhotoImage(image)
        self.canvas.image = photo;
        self.canvas.configure(width=image.width,height=image.height)
        self.canvas.create_image((0,0),anchor="nw",image=photo)
        self.canvas.pack();
        print("DONE")


root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()