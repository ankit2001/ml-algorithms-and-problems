import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 

from Perceptron import Perceptron

def accuracy(y_predicted, y_true):
	return np.sum(y_predicted == y_true) / len(y_true)

X, Y = datasets.make_blobs(n_features = 10, n_samples = 1000, centers = 2, random_state = 121, cluster_std = 1.3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 121)

per = Perceptron(0.01, 1000)

per.fit(X_train, Y_train)

y_predicted = per.predict(X_test)

print("Accuracy of given model is " + str(accuracy(y_predicted, Y_test) * 100))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker = 'o', c = Y_train)

plt.show()

# 100% accuracy on test data yeah...