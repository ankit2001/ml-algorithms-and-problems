import numpy as np 

class NaiveBayes:
	def fit(self, X, Y):
		total_samples, total_features = X.shape
		self.Classes = np.unique(Y);
		self.total_classes = Classes.length
		self.Mean = np.zeros(total_samples, total_features, dtype = np.float64)
		self.Var = np.zeros(total_samples, total_features, dtype = np.float64)
		self.Freq = np.zeros(total_classes, dtype = np.float64)

	def predict(self, X_test):
		y_predicted = [self._predict(x)  for x in X]

	def _predict(self, x):
		pass
