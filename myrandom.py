import random as rd
import math as m

ga = -1
isGa = 0

def gauss_random(desv, medio):
    return rd.normalvariate
    global ga
    global isGa
    if  isGa == 0:
        eq = m.sqrt(-2*m.log(rd.random()))
        ra = rd.random()
        ga = eq*m.sin(2*m.pi*ra)*desv + medio
        isGa = 1
        return eq*m.cos(2*m.pi*ra)*desv + medio
    else:
        isGa = 0
        return ga


def rayleight_random(xi):
    return xi*m.sqrt(-2*m.log(1-rd.random()))


def exponential_random(lam):
    return (-1 / lam) * m.log(1 - rd.random())

