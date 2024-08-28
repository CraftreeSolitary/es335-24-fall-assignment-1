from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn


def shape(lst):
    lst = np.array(lst)
    return np.shape(lst)


data.replace('?', np.nan, inplace=True)

# Option 1: Drop rows with missing values
data.dropna(inplace=True)
# unique_ = data['car name'].unique()
# print(shape(unique_))
y = data['mpg']
X = data.drop(['mpg', 'car name'], axis=1)
X = X.astype("float64")

# In this problem, we are supposed to predict the miles per gallon required as per the models and their features given.
# For this, I have dropped the column car name, as that is quite non repititive and also are categorical, thus making them not useful in training a model.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

clf = DecisionTree(criterion="information_gain")
# Applying the decision tree to train the model
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_test = pd.Series(y_test)


def mse(lst1, lst2):
    # lst1 is the list with predicted value
    sum_ = 0
    for a in range(len(lst1)):
        sum_ = sum_+(lst1[a]-lst2[a])**2
    return (sum_/len(lst1))

print("MSE: ", round(np.sqrt(mse(y_pred, np.array(y_test))), 2))

x_plot = np.linspace(1, 118, 118)
plt.figure(figsize=(12, 8))
plt.plot(x_plot, y_test, color="blue", label="Testing Data")
plt.plot(x_plot, y_pred, color="red", label="Predicted Data")
plt.legend()
plt.show()

# We have predicted the values of mpg based on the other features present about the vehicles by training a model, and then predicting the mpg value. And we can see the model correctly tried to predict the mpg for different models.

# refining data
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
y = data['mpg']
X = data.drop(['mpg', 'car name'], axis=1)
X = X.astype("float64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

start_time = time.time()
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)
y_pred_in = reg.predict(X_test)

end_time = time.time()
print(f"Time required for training and testing (Inbuilt model): {
      end_time-start_time}")

# now checking our model and the time required
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

start_time = time.time()
clf = DecisionTree(criterion="information_gain")


clf.fit(X_train, y_train)
y_pred_out = clf.predict(X_test)
end_time = time.time()
print(f"Time required for training and testing (our model): {
      end_time-start_time}")

# Comparing the Root Mean Squared Error

print(f"Root Mean Squared Error by inbuilt model: {
      np.sqrt(mse(y_pred_in, y_test))}")
print(f"Root Mean Squared Error by our model:     {
      np.sqrt(mse(y_pred_out, y_test))}")
