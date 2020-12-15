import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy
from sklearn import linear_model,preprocessing

# opens car.data and copys data into variable Data
Data = pandas.read_csv("Data_files/car.data")
print(Data.head())