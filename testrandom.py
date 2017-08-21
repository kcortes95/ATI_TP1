import myrandom as r
import matplotlib.pyplot as plt

arr = []
for i in range(1, 100000):
    arr.append(r.rayleight_random(1))

plt.hist(arr, 40)
plt.show()