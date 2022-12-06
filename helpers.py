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

def standardize(df, ignored=[]): 
    """
        Normalize the data using the mean and standard deviation of the training data
        Args:
            X: the data to normalize
        Returns:
            df: the normalized data
    """
    for col in df.columns:
        if col not in ignored:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df
  

def expand_features_poly(df, max_degree, ignored=None):
        """
        Expand the dataframe by adding polynomial features.

        Args:
            degree: maximum degree of the polynomial
            ignored: features to ignore

        Returns:
            df: the expanded dataframe
        """
        if ignored is None:
            ignored = ["state", 'bias']
        df = df.copy()
        for feature in df.columns:
            if feature in ignored:
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


def clean_data(df_raw, max_invalid_ratio, degree, expand_trig):
    """
    Clean the data, removing invalid columns and changing invalid values remaining.
    Args:
        df: dataframe to clean
        max_invalid_ratio: maximum ratio of invalid values in a feature
        degree: degree of the polynomial expansion
        expand_trig: whether to expand the trigonometric features or not
        ignored: list of features to ignore

    Returns:
        the cleaned dataframe
    """
    df = df_raw.copy()

    df = drop_invalid_features(df, max_invalid_ratio)
    df = standardize(df, ignored=["state", "bias"])
    df = expand_features_poly(
        df, degree, ignored=["state", "bias"],
    )

    if expand_trig:
        df = expand_features_trigonometric(df, columns=['EEGv', 'EMGv'])

    return df
