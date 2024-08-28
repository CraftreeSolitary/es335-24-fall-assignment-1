"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from tree.utils import *

np.random.seed(42)


@dataclass
class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.prediction = None
        self.split_value = None
        self.split_attribute = None
        self.children = {}  # the decision tree will be stored here


@dataclass
class DecisionTree:
    """
    criterion -> The criterion to be used for splitting: information_gain or gini_index
    max_depth -> The maximum depth the tree can grow to
    input_type -> The type of input data: real or discrete
    output_type -> The type of output data: real or discrete
    """
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.input_type = None
        self.output_type = None
        self.children = {}  # for the root node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        X -> pd.DataFrame with features as columns and samples as rows
        y -> pd.Series with labels
        """

        # determine the types of input and output data
        dtype_of_each_feature = []
        for i in list(X.columns):
            dtype_of_each_feature.append(X[i].dtype.name)

        self.input_type = dtype_of_each_feature[0]
        # self.input_type = X[0].dtype.name
        self.output_type = y.dtype.name

        # print(self.input_type)
        # print(self.output_type)

        # if the input is discrete, the datatype is "category"
        # if the input is real, the datatype is "float64"
        # category = "category"

        # one hot encode if the input is discrete
        if self.input_type != "float64":
            print("One hot encoding the features!")
            X = one_hot_encoding(X)
            attributes = pd.Series(X.columns)
            # print("attr:", attributes)

        attributes = pd.Series(X.columns)

        if self.input_type != "float64" and self.output_type != "float64":
            print("discrete discrete")
            self.children = self.fit_discrete_discrete(
                X, y, attributes, 0)
        elif self.input_type != "float64" and self.output_type == "float64":
            print("discrete real")
            self.children = self.fit_discrete_real(X, y, attributes, 0)
        elif self.input_type == "float64" and self.output_type != "float64":
            print("real discrete")
            self.children = self.fit_real_discrete(X, y, attributes, 0)
        elif self.input_type == "float64" and self.output_type == "float64":
            print("real real")
            self.children = self.fit_real_real(X, y, attributes, 0)

    def fit_discrete_discrete(self, X: pd.DataFrame, y: pd.Series, attributes: pd.Series, depth: int) -> None:
        """
        Function to train and construct the decision tree for discrete input and discrete output
        X -> pd.DataFrame with features as columns and samples as rows
        y -> pd.Series with labels
        """

        # condition 1: if the depth of the tree is greater than the max_depth, then return the most common class in the current node
        if (depth >= self.max_depth):
            return y.mode().iloc[0]

        # condition 2: if all the samples in the current node belong to the same class, then return that class
        if len(y.unique()) == 1:
            return y.iloc[0]

        # condition 3: if there are no more attributes to split upon, then return the most common class in the current node
        if len(attributes) == 0:
            return y.mode().iloc[0]

        # condition 4: if none of the above conditions are satisfied, then find the best attribute to split upon
        best_attribute = opt_split_discrete(X, y,  self.criterion, attributes)

        # create a new node
        node = Node()
        node.split_attribute = best_attribute

        # remove the best attribute from the list of attributes

        # split the data based on the best attribute
        # get the one hot encoded columns for the best attribute
        encoded_features = [
            col for col in X.columns if col.startswith(str(best_attribute) + '_')]
        # print("encoded features:", encoded_features)
        # attributes = attributes.drop(encoded_features)
        # print(type(attributes))
        for feature in encoded_features:
            # print(type(feature))
            attributes = attributes[attributes != feature]

        # print("attributes:", attributes)
        # print("best attribute:", best_attribute)

        # Iterate over the possible values of the best attribute
        for value in range(len(encoded_features)):
            # Use 1 because it's a one-hot encoding
            X_subset = X[X[encoded_features[value]] == 1]
            y_subset = y[X_subset.index]
            if len(X_subset) == 0:  # just in case
                node.children[value] = y.mode().iloc[0]
                node.prediction = y.mode().iloc[0]
            else:
                node.split_value = value
                # recursively call the function to construct the tree
                node.children[value] = self.fit_discrete_discrete(
                    X_subset, y_subset, attributes, depth + 1)

        return node

    def fit_discrete_real(self, X: pd.DataFrame, y: pd.Series, attributes: pd.Series, depth: int) -> None:
        """
        Function to train and construct the decision tree for discrete input and real output
        X -> pd.DataFrame with features as columns and samples as rows
        y -> pd.Series with labels
        """

        # condition 1: if the depth of the tree is greater than the max_depth, then return the mean of the current node
        if (depth >= self.max_depth):
            return y.mean()

        # condition 2: if there are no more attributes to split upon, then return the mean of the current node
        if len(attributes) == 0:
            return y.mean()

        # condition 3: if all the samples belong to the same value which is unlikely but still, return that value
        if len(y.unique()) == 1:
            return y.iloc[0]

        # condition 4: if none of the above conditions are satisfied, then find the best attribute to split upon
        best_attribute = opt_split_discrete(X, y,  self.criterion, attributes)

        # create a new node
        node = Node()
        node.split_attribute = best_attribute

        encoded_features = [
            col for col in X.columns if col.startswith(str(best_attribute) + '_')]

        # remove the best attribute from the list of attributes
        # attributes = attributes.drop(best_attribute)
        for feature in encoded_features:
            # print(type(feature))
            attributes = attributes[attributes != feature]

        # split the data based on the best attribute
        for value in range(len(encoded_features)):
            X_subset = X[X[encoded_features[value]] == 1]
            y_subset = y[X_subset.index]
            if len(X_subset) == 0:  # just in case
                node.children[value] = y.mean()
                node.prediction = y.mean()
            else:
                node.split_value = value
                node.children[value] = self.fit_discrete_real(
                    X_subset, y_subset, attributes, depth + 1)

        return node

    def fit_real_discrete(self, X: pd.DataFrame, y: pd.Series, attributes: pd.Series, depth: int) -> None:
        """
        Function to train and construct the decision tree for real input and discrete output
        X -> pd.DataFrame with features as columns and samples as rows
        y -> pd.Series with labels
        """

        # condition 1: if the depth of the tree is greater than the max_depth, then return the most common class in the current node
        if (depth >= self.max_depth):
            return y.mode().iloc[0]

        # condition 2: if all the samples in the current node belong to the same class, then return that class
        if len(y.unique()) == 1:
            return y.iloc[0]

        # condition 3: if there are no more attributes to split upon, then return the most common class in the current node
        if len(attributes) == 0:
            return y.mode().iloc[0]

        # condition 4: if none of the above conditions are satisfied, then find the best attribute to split upon
        # print("attributes AJSDLASLD:", type(attributes))
        best_attribute, best_value = opt_split_rido(
            X, y, attributes, self.criterion)

        # create a new node
        node = Node()
        node.split_attribute = best_attribute
        node.split_value = best_value

        # remove the best attribute from the list of attributes
        # attributes = attributes.drop(best_attribute) # ! OLD
        attributes = attributes[~attributes.isin([best_attribute])]  # ! NEW

        # split the data based on the best attribute
        X_left = X[X[best_attribute] <= best_value]
        y_left = y[X_left.index]
        X_right = X[X[best_attribute] > best_value]
        y_right = y[X_right.index]

        # recursively call the function to construct the tree
        node.left = self.fit_real_discrete(
            X_left, y_left, attributes, depth + 1)
        node.right = self.fit_real_discrete(
            X_right, y_right, attributes, depth + 1)

        return node

    def fit_real_real(self, X: pd.DataFrame, y: pd.Series, attributes: pd.Series, depth: int) -> None:
        """
        Function to train and construct the decision tree for real input and real output
        X -> pd.DataFrame with features as columns and samples as rows
        y -> pd.Series with labels
        """

        # print("X: ", X)
        # print("y: ", y)
        # print("attributes: ", attributes)
        # print("depth: ", depth)

        # condition 1: if the depth of the tree is greater than the max_depth, then return the mean of the current node
        if (depth >= self.max_depth):
            return y.mean()

        # condition 2: if there are no more attributes to split upon, then return the mean of the current node
        if len(attributes) == 0:
            return y.mean()

        # condition 3: if all the samples belong to the same value which is unlikely but still, return that value
        if len(y.unique()) == 1:
            return y.iloc[0]

        # condition 4: if none of the above conditions are satisfied, then find the best attribute to split upon
        best_attribute, best_value = opt_split_real(X, y, attributes)

        # print("best_attribute: ", best_attribute)
        # print("best_value: ", best_value)

        # create a new node
        node = Node()
        node.split_attribute = best_attribute
        node.split_value = best_value

        # remove the best attribute from the list of attributes
        # print("attributes:", attributes)
        # print(type(attributes))
        # print("best attribute:", best_attribute)

        # attributes = attributes.drop(best_attribute) # ! OLD
        attributes = attributes[~attributes.isin([best_attribute])]  # ! NEW

        # split the data based on the best attribute
        X_left = X[X[best_attribute] <= best_value]
        y_left = y[X_left.index]
        X_right = X[X[best_attribute] > best_value]
        y_right = y[X_right.index]

        # recursively call the function to construct the tree
        node.left = self.fit_real_real(X_left, y_left, attributes, depth + 1)
        node.right = self.fit_real_real(
            X_right, y_right, attributes, depth + 1)

        return node

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        predictions = []
        for i in range(X.shape[0]):
            # print("Hi")
            # print(self.children)
            current_node = self.children  # start from the root node
            # keep traversing until we reach a leaf node
            while isinstance(current_node, Node):
                split_attribute = current_node.split_attribute
                split_value = current_node.split_value

                if self.input_type == 'category':  # for discrete input
                    # get the value of the split attribute for the current sample
                    value = X[split_attribute].iloc[i]
                    if value in current_node.children:
                        # move to the corresponding child node
                        current_node = current_node.children[value]
                    else:
                        # if the value is not present in the tree, return the majority class
                        current_node = current_node.prediction
                        break
                else:  # for real input
                    value = X[split_attribute].iloc[i]
                    if value <= split_value:
                        current_node = current_node.left  # move to the left child
                    else:
                        current_node = current_node.right  # move to the right child
            # append the prediction for the current sample
            predictions.append(current_node)

        return pd.Series(predictions)

    def plot(self, node=None, indent="") -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if node is None:
            node = self.children  # start from the root node

        # stop when: the current node is a leaf node
        # check if the current node is a leaf node
        if not isinstance(node, Node):
            # if it is a leaf node then print the prediction and exit
            print(indent + "Prediction: " + str(node))
            return

        # do what: print the condition at the current node and recursively call the function for the left and right child
        # if not a leaf node, then print the condition at the current node
        print(indent + f"?(X[{node.split_attribute}] <= {node.split_value})")

        # recurse

        # for real input
        if node.left is not None:
            print(indent + "  Y:")
            self.plot(node.left, indent + "    ")
        if node.right is not None:
            print(indent + "  N:")
            self.plot(node.right, indent + "    ")

        # for discrete input
        if node.children:
            for value, child_node in node.children.items():
                print(indent + f"  Value: {value}")
                self.plot(child_node, indent + "    ")
