"""
Dataset Description: https://archive.ics.uci.edu/ml/datasets/iris
columns:
1. s_l => sepal length in cm
2. s_w => sepal width in cm
3. p_l => petal length in cm
4. p_w => petal width in cm
5. class:
-- Iris Setosa 
-- Iris Versicolour 
-- Iris Virginica
"""

import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['s_l', 's_w', 'p_l', 'p_w'])
Y = pd.DataFrame(iris.target, columns=['class'])

# cross validation
LR = linear_model.LogisticRegression(C=1e5)
print("cross_val_score: ", cross_val_score(LR, X, Y.values.reshape(len(Y)), scoring='accuracy'))

# Merger X,Y into a dataframe
Data = pd.concat([X,Y], axis=1)
print(Data.describe())

# split data into train/test
split_index = int(len(X)*0.66)

# shuffle datasets
Data = Data.sample(frac=1)

# split data into train/test
X_train, X_test = Data.iloc[:split_index, :-2], Data.iloc[split_index:, :-2]
Y_train, Y_test = Data.iloc[:split_index, -1], Data.iloc[split_index:, -1]

LR = linear_model.LogisticRegression(C=1e5)
LR.fit(X_train, Y_train)
prediction = LR.predict(X_test)
matrix = confusion_matrix(prediction, Y_test.values.reshape(len(Y_test)))
print(matrix)