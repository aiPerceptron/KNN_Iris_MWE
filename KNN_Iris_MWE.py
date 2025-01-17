"""
This is a file about data generation using skikitlearn.
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# step one: data
iris = load_iris(return_X_y=True, as_frame=True)
# irisSeries = pd.Series(iris)
# irisNpArray = np.array(irisSeries)

# for key in iris:
#    print(key)
# print(type(iris["frame"]))

X = np.array(iris[0]) # X is the features
y = np.array(iris[1]) # y is the targets/labels

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

# step two: picking the algorithm
algo = KNeighborsClassifier()

# step three: Doing the training

algo.fit(X_train, y_train)

# step four: the prediction

y_pred = algo.predict(X_test)

# step five: plotting and analysis

accurate_score = round(accuracy_score(y_test, y_pred)*100, 2) 
print(str(accurate_score) + "% accuracy")

sepal_length_train = X_train[:, 0] # ALL of the rows but only the first column
sepal_width_train = X_train[:, 1] # ALL of the rows but only the second column
petal_length_train = X_train[:, 2] # ALL of the rows but only the third column
petal_width_train = X_train[:, 3] # ALL of the rows but only the fourth column

sepal_length_test = X_test[:, 0]
sepal_width_test = X_test[:, 1]
petal_length_test = X_test[:, 2]
petal_width_test = X_test[:, 3]

# print(X_test[:, 0]) # selects the 0th column
# print(X_test[0, :]) # this one selects the 0th row
# print(X_test[0]) # this one defaults to selecting the 0th row

plt.scatter(sepal_length_train, sepal_width_train, c=y_train, s=15)
plt.title("Sepals")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.scatter(sepal_length_test, sepal_width_test, s=20, marker="x",c=y_test)
plt.savefig("IrisKNN_SKlearn_Sepals.png")

plt.figure()
plt.scatter(petal_length_train, petal_width_train, c=y_train, s=15)
plt.title("Petals")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.scatter(petal_length_test, petal_width_test, s=20, marker="x",c=y_test)
plt.savefig("IrisKNN_SKlearn_Petals.png")
# print(pd.Series(iris))
