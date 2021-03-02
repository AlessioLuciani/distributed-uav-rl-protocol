#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Values obtained running the simulation after n episodes training
x = [10, 50, 100, 200, 300, 500]
y1 = np.array([1225.0, 1409.0, 1405.0, 1803.0, 1605.0, 1801.0])
y2 = np.array([1424.0, 1801.0, 1803.0, 1604.0, 1601.0, 1603])
y3 = np.array([888.0, 922.0, 1584.0, 1446.0, 1604.0, 1606.0])

poly_degre = 3
f = np.poly1d(np.polyfit(x, y1, poly_degre))
x_new = np.linspace(x[0], x[-1], 50)
y1_new = f(x_new)

f = np.poly1d(np.polyfit(x, y2, poly_degre))
y2_new = f(x_new)

f = np.poly1d(np.polyfit(x, y3, poly_degre))
y3_new = f(x_new)

fig, ax = plt.subplots()
ax.plot(x_new, y1_new, label=r'$\alpha=0.01$')
ax.plot(x_new, y2_new, label=r'$\alpha=0.001$')
ax.plot(x_new, y3_new, label=r'$\alpha=0.0001$')
plt.legend(loc="lower right")

ax.set(xlabel='Number of training episodes', ylabel='Accumulated AoI [s]')
ax.grid()

fig.savefig("images/accumulated_aoi_plot.png")
plt.show()