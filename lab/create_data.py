from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from matplotlib import pyplot as plt

x = np.random.normal(1, 1, 1000)
y = np.random.normal(4, 1, 1000)

plt.hist(x, bins=50, color="red")
plt.hist(y, bins=50, color="blue")
plt.show()
