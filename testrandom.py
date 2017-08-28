import myrandom as r
import matplotlib.pyplot as plt
import actions as a
import numpy as np

arr = []
r1 = 80
r2 = 200
s1 = 20
s2 = 220
for i in range(1, 255):
    if i > r2:
        arr.append(a.getf3(i,r2,s2))
    elif i > r1:
        arr.append(a.getf2(i,r1,r2,s1,s2))
    else:
        arr.append(a.getf1(i, r1, s1))
#plt.plot(arr)
#plt.show()

x = np.array([[1,2,3],[4,5,6]])
print(np.repeat(x,[1,2,3,4,5,6]))