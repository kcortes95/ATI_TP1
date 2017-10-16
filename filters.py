import numpy as np
import meshoperations as mesh


def mean_filter(self, size):



def weighted_mean_filter(self, size):
    m = mesh.weighted_mean_filter(np.array(self.true_image), size)
    self.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)


def weighted_median_filter(self, size):
    m = mesh.weighted_median_filter(np.array(self.true_image), size)
    self.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)


def gauss_filter(self, size, sigma):
    m = mesh.gauss_filter(np.array(self.true_image), size, sigma)
    self.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)


def highpass_filter(self,size):
    m = mesh.highpass_filter(np.array(self.true_image), size)
    self.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)


def median_filter(self,size):
    m = mesh.median_filter(np.array(self.true_image), size)
    self.true_image = Image.fromarray(m)
    self.canvas.image = ImageTk.PhotoImage(self.true_image)
    self.canvas.create_image((0, 0), anchor="nw", image=self.canvas.image)
