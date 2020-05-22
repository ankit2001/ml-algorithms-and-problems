import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
#sklearn for just generating random data
import matplotlib.pyplot as plt 
from Naive import NaiveBayes 
#import Naive Bayes class from the file Naive

#Accuracy function
def accuracy(y_true, y_predicted):
	acc = np.sum(y_true == y_predicted) / len(y_true)
	print("Accuracy of given model is : " + str(acc * 100))
	return 

#I have create tested data here
X, Y = datasets.make_classification(n_samples = 100000, n_features = 50, n_classes = 5, random_state = 121, n_informative = 41)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 121)

# You  can check random data through it
print(X)
print(Y)

nb = NaiveBayes()
nb.fit(X_train, Y_train) # fitting test data
final_predictions = nb.predict(X_test) #final predicted values for testing data

#finding accuracy after comparing through original classes through testing data
accuracy(Y_test, final_predictions)


