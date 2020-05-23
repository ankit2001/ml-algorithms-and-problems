import numpy as np 

class PCA:
	def __init__(self, reduced_to_k):
		self.reduced_to_k = reduced_to_k
		self.Mean = None 
		self.final_components = None

	def fit(self, X):
		self.Mean = np.mean(X, axis = 0)
		X = X - self.Mean
		covariance = np.cov(X.T)
		evalues, evectors = np.linalg.eig(covariance)
		evectors = evectors.T
		indexes_maxi_k = np.argsort(evalues)[ : : -1]
		evectors = evector[indexes_maxi_k]
		self.final_components = evectors[0 : self.reduced_to_k]

	def transform(self, X):
		X = X - self.Mean
		FINAL_COMPONENT_MATRIX = np.dot(X, self.final_components.T)
		return FINAL_COMPONENT_MATRIX


