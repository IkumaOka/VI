import numpy as np
import matplotlib.pyplot as plt

# 最も近いクラスタのインデックスを求める
def calc_cluster(n_clusters,coordinate, label):
		# 各クラスタごとに平均を計算
		cluster_centers = np.array([coordinate[label == i].mean(axis=0) for i in range(n_clusters)])
		# (center_x - x), (center_y - y)の座標を格納
		array = []
		# クラスタの重心からそれぞれの要素を引く
		for i in range(len(coordinate)):
			sub = cluster_centers - coordinate[i]
			array.append(sub)
		array = np.array(array)
		square_array = array ** 2
		#(center_x - x)**2 + (center_y - y)**2
		sum_array = square_array.sum(axis = 2)
		min_cluster_index = sum_array.argmin(axis=1)

		return min_cluster_index

x = np.random.rand(100)
y = np.random.rand(100)

x_y_coordinate = []
for i in range(100):
	elm  =[x[i], y[i]]
	x_y_coordinate.append(elm)
coordinate = np.array(x_y_coordinate)
n_clusters = 3
pred = np.random.randint(0, n_clusters, len(x_y_coordinate))

for num in range(10):
	pred = calc_cluster(n_clusters, coordinate, pred)
	# 新しいラベルで重心を再計算
	new_cluster_centers = np.array([coordinate[pred == i].mean(axis=0) for i in range(n_clusters)])
	for i in range(n_clusters):
		labels = coordinate[pred == i]
		plt.scatter(labels[:, 0], labels[:, 1])
	plt.scatter(new_cluster_centers[:, 0], new_cluster_centers[:, 1], s=100, facecolor='none', edgecolors='black')
	plt.show()

	



