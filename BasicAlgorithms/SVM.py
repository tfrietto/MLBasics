import sklearn
from sklearn import datasets
from sklearn import svm

# importing a data set from datasets
cancerData = datasets.load_breast_cancer()

# assigning our X and Y arrays
X = cancerData.data
Y = cancerData.target

# will be used later to replace the 1's and 0's
targetNames = ["Malignant", "Benign"]

# assigning our training data and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)



