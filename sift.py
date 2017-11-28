from tkinter import Tk, Toplevel, Scale, Entry, Label, Button, messagebox, Menu, filedialog, Canvas, PhotoImage, LEFT
from PIL import Image, ImageTk
import numpy as np
import cv2

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def main_sift_window(self):
    filenames = []
    self.new_window = Toplevel()
    self.new_window.minsize(width=200, height=70)
    self.new_window.title("SIFT")
    self.l=Label(self.new_window,text="Enter Image 1")
    self.l.pack()
    self.img1 = Button(self.new_window, text="Browse", width=10, height=1, command=lambda: read_file_name(filenames) )
    self.img1.pack()
    self.l2=Label(self.new_window,text="Enter Image 2")
    self.l2.pack()
    self.img2 = Button(self.new_window, text="Browse", width=10, height=1, command=lambda: read_file_name(filenames) )
    self.img2.pack()
    self.per=Label(self.new_window,text="Percentage (0 to 1)")
    self.per.pack()
    self.per_val = Entry(self.new_window)
    self.per_val.pack()

    self.ok = Button(self.new_window, text="SIFT", width=10, height=1, command=lambda: apply_sift(float(self.per_val.get()), filenames) )
    self.ok.pack()


def apply_sift(percentage, filenames):
    img1 = cv2.imread(filenames[0], 0)
    img2 = cv2.imread(filenames[1] ,0)
    del filenames[:]

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    #https://docs.opencv.org/trunk/d3/da1/classcv_1_1BFMatcher.html
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    count = 0

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < percentage * n.distance:
            good.append([m])
            count += 1

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)

    tot_kp1 = len(kp1)
    tot_kp2 = len(kp2)
    print("tot_kp1: " + str(tot_kp1))
    print("tot_kp2: " + str(tot_kp2))
    print("Total de MATCHES por BF: " + str(count))

    match_1 = count / tot_kp1
    match_2 = count / tot_kp2
    match = max(match_1, match_2)

    tolerance = 0.15

    print("Tolerancia de " + str(tolerance) + "%")
    if match > tolerance:
        print("Son la misma imagen")
    else:
        print("No son la misma imagen")

    plt.imshow(img3),plt.show()

def read_file_name(filename):
    file = filedialog.askopenfilename()
    filename.append(file)
    print(filename)
    return file
