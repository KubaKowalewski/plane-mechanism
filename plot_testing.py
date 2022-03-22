import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# generate some data
x = np.arange(0, 10, 0.2)
y = np.sin(x)

# plot it
fig = plt.figure(figsize=(8, 6))

gs = gridspec.GridSpec(6, 4)

ax0 = plt.subplot(gs[0:2,1])
# ax0.plot(x, y)
ax0.scatter(x,y)
# ax1 = plt.subplot(gs[2:4,0])
# ax1.plot(y, x)
# ax2 = plt.subplot(gs[4:6,0])
# ax2.plot(y, x)

# ax3 = plt.subplot(gs[0:2,1])
# ax3.plot(x, y)
# ax4 = plt.subplot(gs[2:4,1])
# ax4.plot(y, x)
# ax5 = plt.subplot(gs[4:6,1])
# ax5.plot(y, x)

# ax6 = plt.subplot(gs[0:3,2])
# ax6.plot(x, y)
# ax7 = plt.subplot(gs[3:6,2])
# ax7.plot(y, x)


plt.tight_layout()
plt.savefig('grid_figure.png')

plt.show()