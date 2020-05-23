import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from PCA import PCA

pre_data = datasets.load_iris()
X = pre_data.data 
Y = pre_data.target

pca = PCA(2)
pca.fit(X)
X_transformed = pca.transform(X)

print("Dimensions of X is " + str(X.shape))

print("Dimensions of X_transformed is" + str(X_transformed.shape))

print(Y)

x1 = X_transformed[: ,0]
x2 = X_transformed[: ,1]

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.scatter(x1, x2, alpha = 1, cmap = plt.cm.get_cmap('Dark2', 3), c = Y, edgecolor = "red")
plt.colorbar()
plt.show()