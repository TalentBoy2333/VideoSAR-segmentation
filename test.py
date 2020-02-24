import numpy as np 
import torch 
import matplotlib.pyplot as plt

# print(np.e)

x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.e**(-1*x))

plt.figure()
plt.plot(x, y)

ax = plt.gca()
ax.spines['right'].set_color(None)
ax.spines['top'].set_color(None)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# plt.figure()
# plt.plot(x, y1)

# plt.figure(num=3, figsize=(8,5))
# plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--')
# plt.plot(x, y1)

# fig = plt.figure()
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 = fig.add_axes([left, bottom, width, height])
# ax1.plot(x, y1, 'b')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('title')

plt.show()