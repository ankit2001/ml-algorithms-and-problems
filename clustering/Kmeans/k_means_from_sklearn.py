import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

X_train = np.array([[1, 2], [32, 2], [23, 4], [32, 43], [343, 34], [3, 32]])

model = KMeans(n_clusters = 3)

model.fit(X_train)

centroids = model.cluster_centers_
labels = model.labels_

colors = 10 * ["r.", "g.", "c.", "b."]

for i in range(len(X_train)):
	plt.plot(X_train[i][0], X_train[i][1], colors[labels[i]], markersize = 23)

plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", linewidths = 3, s = 124)
plt.show()

