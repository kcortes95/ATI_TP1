import matplotlib.pyplot as plt
import numpy as np

names = ["Red","Green","Blue","Alpha"]
def get_histogram(image):
    print(image[0, 0])
    if isinstance(image[0,0], np.ndarray):
        global names
        for i in range(0, len(image[0, 0])):
            plt.subplot(2, 2, i+1)
            plt.title(names[i])
            plt.hist(image[:, :, i].flatten(), 100)

        plt.show()

    else:
        plt.hist(image.flatten(), 100)
        plt.show()
