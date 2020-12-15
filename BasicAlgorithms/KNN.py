import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy
from sklearn import linear_model,preprocessing

# opens car.data and copies data into variable Data
Data = pandas.read_csv("Data_files/car.data")
print(Data.head())

# convert all the values in safety to numeric values
# low corresponds to 0, med to 1, and high to 2
def convertSafetyToNum(x):
    if x == "low":
        return 0
    elif x == "med":
        return 1
    else:
        return 2
Data["safety"] = numpy.array(list(map(convertSafetyToNum,Data["safety"])))

