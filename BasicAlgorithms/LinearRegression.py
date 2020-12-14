import numpy
import pandas
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# open and parse csv file
Data = pandas.read_csv("Data_files/student-mat.csv", sep=(";"))

# cut data to include only variables we want
Data = Data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


Predict = "G3"
# declare and assign array X to include all data except "Predict" column
X = numpy.array(Data.drop([Predict], 1))
# declare and assign array Y to include only data in "Predict" column
Y = numpy.array(Data[Predict])

# while loop to get more accurate model
best = 0
while best < 0.95:
    # takes our data from X and Y and splits into 4 groups
    # x_train is values from array X used to create line of best fit
    #       comprised of 90% of data
    # x_test is used to test our line of best fit
    #       comprised of 10% of data
    # same is true of y_train and y_test
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


    # creates and teaches the model the data
    Linear = linear_model.LinearRegression()
    Linear.fit(x_train, y_train)

    # print out accuracy as a percent
    # accuracy is different every run because the declaration of x_train, x_test, etc. chooses random values
    Accuracy = Linear.score(x_test, y_test)
    print("Accuracy: ", Accuracy)

    # if Accuracy of current test is better than all previous tests, update model
    if(Accuracy > best):
        # writing pickle file to save model
        with open("studentG3model.pickle", "wb") as f:
            pickle.dump(Linear, f)
        best = Accuracy

# reading pickle file
pickle_in = open("studentG3model.pickle", "rb")

# using contents of studentG3model.pickle to assign Linear
Linear = pickle.load(pickle_in)

# makes an array of predicted G3 values
Predictions = Linear.predict(x_test)
# loop throught showing what prediction value is based on linear regression
# then shows the data used in calculation
# then shows actual G3 value
for i in range(len(Predictions)):
    print(Predictions[i], x_test[i], y_test[i])