import numpy as np 

class NaiveBayes:
	def fit(self, X, Y):
		total_samples, total_features = X.shape
		self.Classes = np.unique(Y)
		total_classes = len(self.Classes)
		self.Mean = np.zeros((total_classes, total_features), dtype = np.float64)
		self.Var = np.zeros((total_classes, total_features), dtype = np.float64)
		self.Freq = np.zeros(total_classes, dtype = np.float64)
		for (i, c) in enumerate(self.Classes):
			X_tmp = X[c == Y]
			self.Mean[i] = X_tmp.mean(axis = 0)
			self.Var[i] = X_tmp.var(axis = 0)
			self.Freq[i] = len(X_tmp) / float(total_samples)

	def predict(self, X_test):
		y_predicted = [self._predict(x) for x in X_test]
		return y_predicted

	def _predict(self, x):
		which_class = []
		for i in range(len(self.Classes)):
			which_class.append(np.log(self.Freq[i]) + np.sum(np.log(self.normal_dist(x, i))))
		class_index = np.argmax(which_class)
		return self.Classes[class_index]

	def normal_dist(self, x, class_index):
		variance = self.Var[class_index]
		mean = self.Mean[class_index]
		num = np.exp(-((x - mean) ** 2) / (2 * variance))
		den = np.sqrt(2 * np.pi * variance)
		return num / den

