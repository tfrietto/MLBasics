import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# importing a data set from datasets
cancerData = datasets.load_breast_cancer()

# assigning our X and Y arrays
X = cancerData.data
Y = cancerData.target

# assigning our training data and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# create and train model
clf = svm.SVC(kernel="linear")
clf.fit(x_train,y_train)

# create predictions and test accuracy
predictions = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, predictions)


