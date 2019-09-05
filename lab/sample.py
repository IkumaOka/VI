import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt
from sklearn import cluster, preprocessing, mixture
from numpy.random import *
import matplotlib.cm as cm
from sklearn.datasets import load_iris


iris = load_iris()
print(iris.data)