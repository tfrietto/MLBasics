import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy
from sklearn import linear_model,preprocessing

# opens car.data and copies data into variable Data
Data = pandas.read_csv("Data_files/car.data")

# convert all the values in safety to numeric values
# low corresponds to 0, med to 1, and high to 2
Safety = Data["safety"]
def convertSafetyToNum(x):
    if x == "low":
        return 0
    elif x == "med":
        return 1
    else:
        return 2
Safety = numpy.array(list(map(convertSafetyToNum,Safety)))

# an alternative way to convert to numeric values using preprocessing
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(Data["buying"]))
maint = le.fit_transform(list(Data["maint"]))
doors = le.fit_transform(list(Data["doors"]))
persons = le.fit_transform(list(Data["persons"]))
lug_boot = le.fit_transform(list(Data["lug_boot"]))
cls = le.fit_transform(list(Data["class"]))

# a variable for the prediction value
predict = "class"

# creating the values to populate our X and Y values
X = list(zip(buying,maint,doors,persons,lug_boot,Safety))
Y = list(cls)

#assigning values for our training variables and testing variables
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

# creating and training our model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)

# getting accuracy of model
accuracy = model.score(x_test,y_test)
print("Accuracy: ", accuracy)

# prints predicted data next to actual data
predictions = model.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i],x_test[i],y_test[i])