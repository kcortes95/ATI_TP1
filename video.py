import numpy as np
import math
import time

directions = [[1, 0], [1, 1], [1, -1], [0, 1], [0, -1], [-1, 0], [-1, 1], [-1, 1]]

def pixel_exchange_border_detect(mat, coords, outer_coords):
    phi, lin, lout, theta0, theta1 = classify(mat, coords, outer_coords)
    return apply_pixel_exchange(mat, phi, lin, lout, theta0, theta1)


def apply_pixel_exchange(mat, phi, lin, lout, theta0, theta1):
    print('when starting ' + str(len(lout)))
    starting = int(time.time() * 1000)
    F = getF(mat, theta0, theta1)
    change = True

    i = 0
    while change:
        new_lout = []
        new_lin = []
        change = False
        for j in range(len(lout)):
            if F[lout[j][0], lout[j][1]] > 0:
                lin.append(lout[j])
                phi[lout[j][0], lout[j][1]] = -1
                checkNeighbours(phi, new_lout, lout[j], 3, 1)
                change = True
            else:
                new_lout.append(lout[j])

        removeExtra(phi, new_lin, -1, -3)

        for j in range(len(lin)):
            if F[lin[j][0], lin[j][1]] < 0:
                new_lout.append(lin[j])
                phi[lin[j][0], lin[j][1]] = 1
                checkNeighbours(phi, new_lin, lin[j], -3, -1)
                change = True
            else:
                new_lin.append(lin[j])

        removeExtra(phi, new_lout, 1, 3)
        i += 1
        lin = new_lin
        lout = new_lout
    print("Done. Number of iterations: " + str(i) + " Time elapsed:" + str(int(time.time() * 1000) - starting))
    return phi, new_lin, new_lout, theta0, theta1


def classify(mat, coords, outer_coords):  # (min_x,max_x,min_y,max_y)
    lin = []
    lout = []
    theta0 = np.zeros(3)
    theta1 = np.zeros(3)
    w, h, z = mat.shape
    phi = np.empty((w, h), dtype=np.int8)

    for i in range(w):
        for j in range(h):
            if ((j == coords[0] or j == coords[1]) and coords[2] <= i <= coords[3]) or (
                    (i == coords[2] or i == coords[3]) and coords[0] <= j <= coords[1]):
                lout.append([i, j])
                phi[i, j] = 1
            elif ((j == coords[0] + 1 or j == coords[1] - 1) and coords[2] + 1 <= i <= coords[3] - 1) or (
                (i == coords[2] + 1 or i == coords[3] - 1) and coords[0] + 1 <= j <= coords[1] - 1):
                lin.append([i, j])
                phi[i, j] = -1
                theta1 += mat[i, j]
            elif j > coords[0] and j < coords[1] and i > coords[2] and i < coords[3]:
                phi[i, j] = -3
                theta1 += mat[i, j]
            else:
                phi[i, j] = 3

            if j > outer_coords[0] and j < outer_coords[1] and i > outer_coords[2] and i < outer_coords[3]:
                theta0 += mat[i, j]

    pixels_in = (coords[1] - coords[0]) * (coords[3] - coords[2])
    pixels_out = (outer_coords[1] - outer_coords[0]) * (outer_coords[3] - outer_coords[2])

    return phi, lin, lout, theta0 / pixels_out, theta1 / pixels_in


def norm(t1, t2):
    return math.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2 + (t1[2] - t2[2]) ** 2)


def getF(mat, t0, t1):
    w, h, z = mat.shape
    F = np.empty((w, h))
    for i in range(w):
        for j in range(h):
            t = mat[i, j]
            numerator = norm(t0, t)
            denominator = norm(t1, t)
            F[i, j] = math.log(numerator / denominator)

    return F


def checkNeighbours(phi, l, coord, ifThis, thenThat):
    x = coord[0]
    y = coord[1]
    w, h = phi.shape
    for i in range(len(directions)):
        new_x, new_y = x + directions[i][0], y + directions[i][1]
        if 0 <= new_x < w and 0 <= new_y < h:
            if phi[new_x, new_y] == ifThis:
                phi[new_x, new_y] = thenThat
                l.append([new_x, new_y])


def removeExtra(phi, l, surroundedBy, setTo):
    w, h = phi.shape
    for coord in l:
        count = 0
        for d in directions:
            new_x, new_y = coord[0] + d[0], coord[1] + d[1]
            if 0 <= new_x < w and 0 <= new_y < h:
                state = phi[new_x, new_y]
                if state != surroundedBy and state != setTo:
                    count += 1

        if count == 0:
            l.remove(coord)
            phi[coord[0], coord[1]] = setTo
