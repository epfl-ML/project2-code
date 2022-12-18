import numpy as np
import functions_ml as impl
import pandas as pd
from sklearn.linear_model import RidgeClassifier

def split_label_features(df):
    """
    Split a dataframe into features and labels.
    Args:
        df: dataframe to split

    Returns:
        a tuple containing the features and labels.
    """
    return df["state"], df.drop("state", axis=1)



def drop_invalid_features(df, max_invalid_ratio):
    """
    Remove all the columns with more than 50% of the data missing or zero.
    Args:
        df: the dataframe to clean

    Returns:
        df: the cleaned dataframe
    """
    df = df.copy()
    for col in df.columns:
        if not df[col].any():
            df.drop(col, axis=1, inplace=True)
    return df

def standardize(df, features=[]): 
    """
        Normalize the data using the mean and standard deviation of the training data
        Args:
            X: the data to normalize
        Returns:
            df: the normalized data
    """
    for col in df.columns:
        if col in features:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df


def log_features(df, features=[]):
    """
    Take the log of the features.
    Args:
        df: the dataframe to transform

    Returns:
        df: the transformed dataframe
    """
    df = df.copy()
    for col in df.columns:
        if col in features:
            df[col] = np.log(df[col])
    return df
  

def expand_features_poly(df, max_degree, features=None):
        """
        Expand the dataframe by adding polynomial features.

        Args:
            degree: maximum degree of the polynomial
            ignored: features to ignore

        Returns:
            df: the expanded dataframe
        """
        if features is None:
            features = ['EEGv', 'EMGv']
        df = df.copy()
        for feature in df.columns:
            if feature not in features:
                continue
            for degree in range(2, max_degree + 1):
                df[f"{feature}^{degree}"] = df[feature] ** degree
        # add bias
        df["bias"] = 1
        return df

def expand_features_trigonometric(df, columns=None):
    """
    Expand the features of a dataframe by adding trigonometric features. The original features are kept with the same
    name, and the new features are named as "feature_name_sin" and "feature_name_cos".
    Args:
        df: dataframe to expand
        columns: list of columns to expand

    Returns:
        the expanded dataframe
    """
    pi_upper_bound = 3.2  # Handles bad rounding in the data sets

    df = df.copy()
    for feature in df.columns:
        if feature not in columns:
            continue
        # Compute the sine and cosine of the feature, but only if the feature value is in the range [-pi, pi]. Values
        # outside this range are set to -999.
        df[f"{feature}_sin"] = np.where(
            np.logical_and(
                df[feature] >= -pi_upper_bound, df[feature] <= pi_upper_bound
            ),
            np.sin(df[feature]),
            -999,
        )
        df[f"{feature}_cos"] = np.where(
            np.logical_and(
                df[feature] >= -pi_upper_bound, df[feature] <= pi_upper_bound
            ),
            np.cos(df[feature]),
            -999,
        )

    return df


def clean_data(df, degree, expand_trig, max_invalid_ratio, features=['EEGv', 'EMGv']):
    """
    Clean the data, removing invalid columns and changing invalid values remaining.
    Args:
        df: dataframe to clean
        degree: degree of the polynomial expansion
        expand_trig: whether to expand the trigonometric features or not
        ignored: list of features to ignore
        max_invalid_ratio: maximum ratio of invalid values in a column
        features: list of features to expand
    Returns:
        the cleaned dataframe
    """
    df = df.copy()

    df = drop_invalid_features(df, max_invalid_ratio)
    df = log_features(df, features=features)
    df = standardize(df, features=features)
    df = expand_features_poly(
        df, degree, features=features,
    )

    if expand_trig:
        df = expand_features_trigonometric(df, columns=['EEGv', 'EMGv'])

    return df
