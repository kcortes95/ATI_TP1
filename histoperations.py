import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np


names = ["Red", "Green", "Blue", "Alpha"]


def equalize(matrix):
    if len(matrix.shape) > 2:
        out = np.zeros(shape=matrix.shape)
        buckets = get_buckets(matrix)
        for i in range(matrix.shape[2]):
            aux = equalize_single(matrix[:, :, i], buckets[i])
            out[:, :, i] = normalize(aux[0], aux[1])
    else:
        buckets = get_bucket(matrix.flatten())
        aux = equalize_single(matrix, buckets)
        out = normalize(aux[0], aux[1])

    return out


def equalize_single(image, bucket):
    acum = get_acum(bucket)
    result = np.zeros(shape=image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = acum[image[i, j]]/image.size

    return result, np.min(result)


def normalize(array, minimum):
    shape = array.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            # array[i, j] = int((array[i, j] - min) / (1 - min) + 0.5) WTF is this transformation
            array[i, j] = int((array[i, j] - minimum)*255 / (1 - minimum))
    return array


def get_acum(array):
    ret = np.array(array)
    for i in range(1, len(array)):
        ret[i] += ret[i-1]
    return ret


def sharpen(matrix):
    return None


def get_buckets(matrix):
    buckets = np.zeros(shape=(len(matrix[0, 0]), 256))
    shape = matrix.shape
    for i in range(shape[2]):
        for j in range(shape[0]):
            for k in range(shape[1]):
                buckets[i, matrix[j, k, i]] += 1

    return buckets


def get_bucket(array):
    arr = np.zeros(256)
    for i in range(len(array)):
        arr[array[i]] += 1
    return arr


def get_histogram(image):
    if isinstance(image[0, 0], np.ndarray):
        image = get_buckets(image)
        global names
        for i in range(0, len(image[:, 1])):
            plt.subplot(2, 2, i + 1)
            plt.title(names[i])
            plt.bar(range(0, 256), image[i])

        plt.show()

    else:
        print(range(256))
        image = get_bucket(image.flatten())
        plt.bar(range(0, 256), image)
        plt.title("Grey")
        plt.show()
