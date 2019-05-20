from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from matplotlib import pyplot as plt

dataset, y = make_blobs(centers=2)
print(dataset.shape)
print(y.shape)
print(dataset[0])
print(dataset[1])
plt.scatter(dataset[:,0], dataset[:,1], c=y)
plt.show()