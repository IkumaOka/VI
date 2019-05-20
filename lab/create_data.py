from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from matplotlib import pyplot as plt

dataset, features = make_blobs(centers=2, random_state=11)

plt.scatter(dataset[:,0], dataset[:,1], c=features)
plt.show()