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

X = np.array(iris[0])
y = np.array(iris[1])

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25)

# step two: picking the algorithm
algo = KNeighborsClassifier()

# step three: Doing the training

algo.fit(X_train, y_train)

# step four: the prediction

y_pred = algo.predict(X_test)

# step five: plotting and analysis

accurate_score = round(accuracy_score(y_test, y_pred)*100, 2) 
print(str(accurate_score) + "% accuracy")

sepal_length = X[:, 0]
sepal_width = X[:, 1]
petal_length = X[:, 2]
petal_width = X[:, 3]

plt.scatter(sepal_length, sepal_width, c=y, s=15)
plt.title("Sepals")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.savefig("IrisKNN_SKlearn_Sepals.png")
plt.figure()
plt.scatter(petal_length, petal_width, c=y, s=15)
plt.title("Petals")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.savefig("IrisKNN_SKlearn_Petals.png")
# print(pd.Series(iris))
