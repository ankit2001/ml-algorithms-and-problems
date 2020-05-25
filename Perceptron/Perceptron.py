import numpy as np 

class Perceptron:
	def __init__(self, learning_rate = 0.001, n_iter = 10000):
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.activation_function = unit_step_function
		self.weights = None
		self.bias = None

	def unit_step_function(self, X):
		return np.where(X >= 0, 1, 0)

	def fit(self, X, Y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0
		Y = np.array([1 if x > 0 else 0 for x in Y])
		for i in range(self.n_iter):
			for idx, x in enumerate(X):
				solve = np.dot(x, self.weights) + self.bias
				y_pred = self.activation_function(solve)
				self.weights += self.learning_rate * (Y[idx] - y_pred) * x
				self.bias += self.learning_rate * (Y[idx] - y_pred)
	
	def predict(self, X):
		solve = np.dot(X, self.weights) + self.bias
		return self.activation_function(solve)
