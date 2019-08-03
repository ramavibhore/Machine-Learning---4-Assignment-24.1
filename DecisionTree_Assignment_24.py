# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
 
# Function importing Dataset
def importdata():
	
	Url='https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
	titanic = pd.read_csv(Url)     
	# Printing the dataswet shape
	print ("Dataset Lenght: ", len(titanic))
	print ("Dataset Shape: ", titanic.shape)
	 
	# Printing the dataset obseravtions
	print('Dataset:')
	with pd.option_context('expand_frame_repr', False):
		print (titanic.head())
	
	# Factorising sex column
	titanic['Sex_fact'], _ = pd.factorize(titanic['Sex'])
	return titanic
 
# Function to split the dataset
def splitdataset(titanic):
 
	# Seperating the target variable
	X = titanic[['Pclass', 'Sex_fact', 'Age', 'SibSp', 'Parch', 'Fare']]
	Y = titanic['Survived']
 
	# We observed that Age column is having null value with 177 records. So we will fill those with 0 value
	print('Count of null value in Age column: ',X['Age'].isnull().sum())
	###### Data manipulation as  ########
	X.fillna(0,inplace=True)
	
	# Spliting the dataset into train and test
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
	
	return X, Y, X_train, X_test, y_train, y_test
	
# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):

	# Creating the classifier object
	clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
 
	# Performing training
	clf_gini.fit(X_train, y_train)
	return clf_gini
     
# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
 
	# Decision tree with entropy
	clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5)
 
	# Performing training
	clf_entropy.fit(X_train, y_train)
	return clf_entropy
 
 
# Function to make predictions
def prediction(X_test, clf_object):
 
	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred
	 
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred,X,Y,dtree):
	
	print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
	 
	print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)
	#Avg accuracy using KFold
	cv = KFold(n=len(X),  # Number of elements
			n_folds=10,            # Desired number of cv folds
			random_state=12)
			
	fold_accuracy = []


	for train_fold, valid_fold in cv:
		train = X.loc[train_fold] # Extract train data with cv indices
		valid = X.loc[valid_fold] # Extract valid data with cv indices
	
		train_y = Y.loc[train_fold]
		valid_y = Y.loc[valid_fold]
	
		model = dtree.fit(X = train, y = train_y)
		valid_acc = model.score(X = valid, y = valid_y)
		fold_accuracy.append(valid_acc)
		
	print("Accuracy per fold: ", fold_accuracy, "\n")
	print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))
	 
	print("Report : ",classification_report(y_test, y_pred))
 
# Driver code
def main():
	 
	# Building Phase
	data = importdata()
	X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
	clf_gini = train_using_gini(X_train, X_test, y_train)
	clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
	 
	# Operational Phase
	print("Results Using Gini Index:")
	 
	# Prediction using gini
	y_pred_gini = prediction(X_test, clf_gini)
	cal_accuracy(y_test, y_pred_gini,X,Y,clf_gini)
	 
	print("Results Using Entropy:")
	# Prediction using entropy
	y_pred_entropy = prediction(X_test, clf_entropy)
	cal_accuracy(y_test, y_pred_entropy,X,Y,clf_entropy)
	 
	 
# Calling main function
if __name__=="__main__":
	main()