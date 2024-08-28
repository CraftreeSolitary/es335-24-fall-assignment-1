"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import math
import pandas as pd


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    # this function will accept a dataframe and return a dataframe with one hot encoding
    encoded_df = pd.DataFrame()

    # iterate over all the columns in the dataframe
    for col in X.columns:
        unique_values = sorted(X[col].unique())

        # iterate over all the unique values in the column
        for val in unique_values:
            # create a new column in the encoded dataframe with the name of the column and the unique value
            # and set the value of the column to 1 if the value in the original dataframe is equal to the unique value
            # otherwise set the value to 0
            # print(type(col)) # int or str
            # print(type(val)) # int
            encoded_df[str(col) + "_" + str(val)] = (X[col] == val).astype(int)

    return encoded_df


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    pass


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    hash = {}

    # calculate the frequency of each value in the series
    for value in Y:
        if value in hash:
            hash[value] += 1
        else:
            hash[value] = 1

    # calculate the entropy
    # entropy = summation of -p(x) * log2(p(x)) for all x in the series
    entropy = 0
    for value in hash:
        prob = hash[value] / len(Y)
        entropy -= prob * math.log2(prob)

    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    The Gini Index measures how often a randomly chosen element from the set would be incorrectly
    labeled if it was randomly labeled according to the distribution of labels in the set.
    """
    hash = {}

    # calculate the frequency of each value in the series
    for value in Y:
        if value in hash:
            hash[value] += 1
        else:
            hash[value] = 1

    # calculate the gini index
    # gini index = 1 - summation of p(x)^2 for all x in the series
    gini = 1
    for value in hash:
        prob = hash[value] / len(Y)
        gini -= prob ** 2

    return gini


def mean_squared_error(y: pd.Series) -> float:
    # print("here")
    mean = y.mean()
    sum = 0
    for x in y:
        sum += (x - mean) ** 2

    return sum / len(y)


# def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
#     """
#     Function to calculate the information gain using criterion (entropy, gini index or MSE)
#     attr -> attribute to split upon, it is in the form of a series which has values of the attribute
#     """
#     # calculate the entropy or gini index of the original series
#     if criterion == "entropy":
#         org_criteria_value = entropy(Y)
#     else:
#         org_criteria_value = gini_index(Y)

    # # calculate the weighted entropy or gini index of the attribute
    # weighted_criteria_value = 0

    # # iterate over all the unique values in the attribute
    # print("attributes:", attr)
    # print("unique:", attr.unique())
    # for value in attr.unique():
    #     # calculate the subset of the series where the attribute is equal to the value
    #     # this will give a boolean series
    #     subset_Y = Y[attr == value]
    #     print("subset_Y:", subset_Y)
    #     # calculate the weight of the subset
    #     subset_weight = len(subset_Y) / len(Y)
    #     if criterion == "entropy":
    #         subset_criteria_value = entropy(subset_Y)
    #         print("subset_criteria_value:", subset_criteria_value)
    #     else:
    #         subset_criteria_value = gini_index(subset_Y)

    #     # calculate the weighted entropy or gini index of the attribute
    #     weighted_criteria_value += (subset_weight * subset_criteria_value)
    #     print("weighted_criteria_value:", weighted_criteria_value)

    # print("org_criteria_value:", org_criteria_value)

    # # now we have

    # # calculate the information gain
    # information_gain = org_criteria_value - weighted_criteria_value

    # return information_gain

def information_gain(Y: pd.Series, attr: pd.DataFrame, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index)
    attr -> attribute to split upon, it is in the form of a series which has values of the attribute
    """
    # print(type(attr)) # dataframe
    # calculate the entropy or gini index of the original series
    if criterion == "entropy":
        org_criteria_value = entropy(Y)  # ye sahi se aa raha hai
        # print("Y: ", Y)
        # print("org_criteria_value: ", org_criteria_value)
    else:
        org_criteria_value = gini_index(Y)

    # calculate the weighted entropy or gini index of the attribute
    weighted_criteria_value = 0

    # Group features by their original feature (before one-hot encoding)
    original_features = {}
    for feature in attr.columns.unique():
        # print("feature: ", feature)
        # Extract the original feature name
        original_feature = feature.split('_')[0]
        if original_feature not in original_features:
            original_features[original_feature] = []
        original_features[original_feature].append(feature)

    # print("original features: ", original_features)
    # print("attr", attr)

    # Iterate through each group of one-hot encoded features
    # print("orignial feature items: ", original_features.items())
    # print(list(original_features.values())) # 2d list
    encoded_features_2d = list(original_features.values())
    encoded_features = [
        item for sublist in encoded_features_2d for item in sublist]
    # print("encoded_features: ", encoded_features)
    for encoded_feature in encoded_features:
        # Calculate the subset of the series where the attribute is equal to the value
        # Use .any() to check if any of the one-hot encoded features is True
        # print("original feature: ", original_feature)
        # print("encoded features: ", encoded_features)
        indices = attr[attr[encoded_feature] == 1].index
        subset_Y = Y.loc[indices]
        # print("--------------------")
        # print(subset_Y)
        # print("--------------------")

        # Calculate the weight of the subset
        subset_weight = len(subset_Y) / len(Y)
        if criterion == "entropy":
            subset_criteria_value = entropy(subset_Y)
            # print("subset_criteria_value:", subset_criteria_value)
        else:
            subset_criteria_value = gini_index(subset_Y)

        # Calculate the weighted entropy or gini index of the attribute
        weighted_criteria_value += (subset_weight * subset_criteria_value)

    # Calculate the information gain
    information_gain = org_criteria_value - weighted_criteria_value

    # print("===============")
    # print("information_gain: ", information_gain)
    # print("===============")

    return information_gain


def most_reduction_in_mse(X: pd.DataFrame, y: pd.Series, feature: pd.Series) -> float:
    # this function will find the best split for real valued output
    best_value = None
    best_score = -float('inf')

    # print("X: ", X)
    # print("y: ", y)
    # print("feature: ", feature)
    # print(X[feature].sort_values().unique())

    # Sort the feature values and get unique values
    sorted_feature = X[feature].sort_values().unique()
    # print("Sorted Feature: ", sorted_feature)

    # Iterate over possible split points between consecutive values
    for i in range(1, len(sorted_feature)):
        # print("HERE")
        # Calculate the split value as the average of consecutive values
        split_value = (sorted_feature[i] + sorted_feature[i - 1]) / 2

        # print("Split Value: ", split_value)

        # Split the data based on the split value
        left = y[X[feature] <= split_value]
        right = y[X[feature] > split_value]

        # Calculate the Mean Squared Error for the split
        # print("left: ", X[feature] <= split_value)
        # print("right: ", X[feature] > split_value)
        # mean_squared_error expects a series
        # print(type(left)) # series
        # whats the error then?
        left_mse = mean_squared_error(y=left)  # ! ERROR HERE
        right_mse = mean_squared_error(right)
        weighted_mse = (len(left) / len(y) * left_mse) + \
            (len(right) / len(y) * right_mse)
        reduction_in_mse = mean_squared_error(y) - weighted_mse

        # Update the best split value and score if the reduction in MSE is higher
        if reduction_in_mse > best_score:
            best_value = split_value
            best_score = reduction_in_mse

    return best_value, best_score


def opt_split_discrete(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    best_attribute = None
    best_score = -float('inf')

    # # iterate over all the features
    # for feature in features:
    #     # get the unique values of the feature
    #     unique_values = X[feature].unique()

    #     # iterate over all the unique values
    #     for value in unique_values:
    #         # calculate the information gain of the feature
    #         info_gain = information_gain(y, X[feature] == value, criterion)

    #         # update the best attribute and value if the information gain is higher
    #         if info_gain > best_score:
    #             best_attribute = feature
    #             best_value = value
    #             best_score = info_gain

    # return best_attribute, best_value

    # Group features by their original feature (before one-hot encoding)
    original_features = {}
    for feature in features:
        # Extract the original feature name
        try:
            original_feature = int(feature.split('_')[0])
        except ValueError:
            original_feature = feature.split('_')[0]
        if original_feature not in original_features:
            original_features[original_feature] = []
        original_features[original_feature].append(feature)

    # print("Original Features: ", original_features)

    # Iterate through each group of one-hot encoded features
    for original_feature, encoded_features in original_features.items():
        # print("passing this into information gain:", pd.Series(encoded_features))
        # print(encoded_features)
        # print(type(encoded_features))
        # print(X[encoded_features])
        # print(X[encoded_features].columns)
        # Calculate the information gain for the group of features
        info_gain = information_gain(
            y, X[encoded_features], criterion)
        # print("Information Gain: ", info_gain, "for feature: ", original_feature)

        # Update the best attribute and value if the information gain is higher
        if info_gain > best_score:
            best_attribute = original_feature  # Store the original feature name
            best_score = info_gain

    return best_attribute


def opt_split_real(X: pd.DataFrame, y: pd.Series, features: pd.Series):
    # this function will find the best split for real valued output

    # print("X: ", X)
    # print("y: ", y)
    # print("features: ", features)

    best_attribute = None
    best_value = None
    best_score = -float('inf')

    # iterate over all the features
    # print(features)
    for feature in features:
        # find the best split for the feature
        # print(feature)
        value, score = most_reduction_in_mse(X, y, feature)

        # update the best attribute and value if the reduction in MSE is higher
        if score > best_score:
            best_attribute = feature
            best_value = value
            best_score = score

    return best_attribute, best_value


def information_gain_without_one_hot_encoding(Y: pd.Series, attr: pd.Series, criterion: str, split_value) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index)
    attr -> attribute to split upon, it is in the form of a series which has values of the attribute
    """
    # calculate the entropy or gini index of the original series
    if criterion == "entropy":
        org_criteria_value = entropy(Y)
    else:
        org_criteria_value = gini_index(Y)

    # calculate the weighted entropy or gini index of the attribute
    weighted_criteria_value = 0

    left = Y[attr <= split_value]
    right = Y[attr > split_value]

    if criterion == "entropy":
        weighted_criteria_value = (
            len(left) / len(Y)) * entropy(left) + (len(right) / len(Y)) * entropy(right)
    else:
        weighted_criteria_value = (
            len(left) / len(Y)) * gini_index(left) + (len(right) / len(Y)) * gini_index(right)

    information_gain = org_criteria_value - weighted_criteria_value

    return information_gain


def opt_split_rido(X: pd.DataFrame, y: pd.Series, features: pd.Series, criterion: str):
    # we need to find the best split for real input and discrete output
    # iterate over all the features
    # for each feature, sort the values and find the best split
    # get the information gain for each feature using entropy or gini index
    # return the feature with the highest information gain
    best_attribute = None
    best_score = -float('inf')
    best_split = None

    for feature in features:
        # first sort the values of the feature
        sorted_values = X[feature].sort_values().unique()

        # iterate over all the possible split points
        for i in range(1, len(sorted_values)):
            split = (sorted_values[i] + sorted_values[i - 1]) / 2

            # calculate the information gain
            info_gain = information_gain_without_one_hot_encoding(
                y, X[feature], criterion, split)

            # update the best attribute and score if the information gain is higher
            if info_gain > best_score:
                best_attribute = feature
                best_score = info_gain
                best_split = split

    return best_attribute, best_split


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    pass
