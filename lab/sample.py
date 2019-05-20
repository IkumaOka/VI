from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from matplotlib import pyplot as plt

a = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1])
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(b[a == 1])