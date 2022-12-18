import enum
import typing

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

def load_features(file: str) -> pd.DataFrame:
    """
    Load features from a csv files.
    Args:
        file: path to the dataset.
    Returns:
        data: numpy array of shape (N, D).
    """
    return pd.read_csv(file, sep=',', header=0)


class WindowOperationFlag(enum.IntFlag):
    """
    The different features that can be extracted from a window. If multiple features are selected,
    they will all be extracted from the window.
    """
    MEAN = enum.auto()
    MEDIAN = enum.auto()
    VAR = enum.auto()


def features_window(
        df: pd.DataFrame,
        window_size: int,
        op: WindowOperationFlag = 0,
        features: typing.Union[list[str], None] = None,
        center=False,
) -> pd.DataFrame:
    """
    Smooth features by performing a set of operations over a window of size `window_size`. The implementations must
    select the features to smooth from the `features` argument. If `features` is None, all features will be smoothed.
    Args:
        df: numpy array of shape (N, D).
        window_size: size of the window.
        op: the operation(s) to perform on the window.
        features: the list of features
        center: whether the window is centered or not.
    Returns:
        data: numpy array of shape (N, D).
    """
    if op == 0:
        op = WindowOperationFlag.MEAN | WindowOperationFlag.MEDIAN | WindowOperationFlag.VAR
    if features is None:
        features = list(df.columns)

    # For each operation, add the computed aggregated value.
    if WindowOperationFlag.MEAN & op == WindowOperationFlag.MEAN:
        df[[f + f"_mean{window_size}" for f in features]] = df[features].rolling(window_size, center=center).mean()
    if WindowOperationFlag.MEDIAN & op == WindowOperationFlag.MEDIAN:
        df[[f + f"_median{window_size}" for f in features]] = df[features].rolling(window_size, center=center).median()
    if WindowOperationFlag.VAR & op == WindowOperationFlag.VAR:
        df[[f + f"_var{window_size}" for f in features]] = df[features].rolling(window_size, center=center).var()

    return df

def log_features(df, features=[]):
    """
    Take the log of the features.
    Args:
        df: the dataframe to transform

    Returns:
        df: the transformed dataframe
    """
    df1 = df.copy()
    for feature in features:
        df1[f"{feature}_log"] = np.log(df1[feature])
    return df1

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

def filter_days(df, days):
    df1 = df.copy()
    df1["day"] = df1.index // 21600
    return df[df1["day"].isin(days)]

def states(rawState):
    if rawState:
        return 'rawState', 'state'
    return 'state', 'rawState'

def split_labels(df, useRaw=False):
    skeep, sdrop = states(useRaw)

    df1 = df.copy()

    x = df1.drop([skeep], axis=1)
    y = df1[skeep]

    return (x, y)
    
def encode_labels(y, cat_matrix=False):
    le = LabelEncoder()
    le.fit(y)
    y1 = le.transform(y)
    if cat_matrix:
        y1 = tf.keras.utils.to_categorical(y1)
    return (y1, le)

def decode_labels(le, y):
    return le.inverse_transform(y)

def standardize(df, features=[]): 
    """
        Normalize the data using the mean and standard deviation of the training data
        Args:
            X: the data to normalize
        Returns:
            df: the normalized data (dataframe)
    """
    for col in df.columns:
        if col in features:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df

def split_encode_scale_data(df, useRaw, test_size, seed, cat_matrix):
    """
    Good for using same mice for train and test
    """
    x, y_raw = split_labels(df, useRaw=useRaw)
    y, le = encode_labels(y_raw, cat_matrix=cat_matrix)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, le

def split_encode_scale_data_kfold(df, useRaw, seed, cat_matrix):
    """
    Good for using same mice for kfold
    """
    x, y_raw = split_labels(df, useRaw=useRaw)
    y, le = encode_labels(y_raw, cat_matrix=cat_matrix)

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    return x, y, le

def encode_scale_data(df_train, df_test, useRaw, seed, cat_matrix):
    """
    Good for using different mice for train and test
    """
    x_train, y_train_raw = split_labels(df_train, useRaw=useRaw)
    y_train, le = encode_labels(y_train_raw, cat_matrix=cat_matrix)

    x_test, y_test_raw = split_labels(df_test, useRaw=useRaw)
    y_test = le.transform(y_test_raw)
    if cat_matrix:
        y_test = tf.keras.utils.to_categorical(y_test)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, le

def clean_data(data_folder, data_files, days, window_sizes, window_features, dropBins, useRaw, balance=True, standardize_df=False, standardize_features=[]):
    df = pd.DataFrame()
    # iterate over several files
    for file in data_files:
        df_temp = load_features(data_folder + file)

        # filter days
        df_temp = filter_days(df_temp, days)

        # compute bins features
        # drop raw bins
        if dropBins:
            for i in range(401):
                df_temp = df_temp.drop([f"bin{i}"], axis=1)

        # add feature window
        window_names = ["EEGv", "EMGv"]
        for window_size in window_sizes:
            df_temp = features_window(df_temp, window_size=window_size, op=WindowOperationFlag.MEAN, features=window_features)
            df_temp = features_window(df_temp, window_size=window_size, op=WindowOperationFlag.VAR, features=window_features)

            for feature in window_features:
                window_names.append(feature + "_mean" + str(window_size))
                window_names.append(feature + "_var" + str(window_size))
        # drop nan from feature window
        df_temp = df_temp.dropna()

        # add logs features
        # drop zeroes for EEGv and EMGv
        size_before = df_temp.shape[0]
        for feature in window_names:
            df_temp = df_temp[df_temp[feature] > 0]
        size_after = df_temp.shape[0]
        if size_before != size_after:
            print(f"Removed {size_before - size_after} rows with invalid log values in {file}")
        df_temp = log_features(df_temp, window_names)
        
        # add polynomial expansion
        # add trigonometric expansion ?

        df = pd.concat([df, df_temp])

    # standardize
    if standardize_df:
        df = standardize(df, features=features)

    # balance classes
    skeep, sdrop = states(useRaw)
    if balance:
        balance = df[skeep].value_counts().min()
        print(f"Balancing classes to {balance} samples per class (total: {balance * len(df[skeep].unique())})")
        df = df.groupby(skeep).apply(lambda x: x.sample(balance)).reset_index(drop=True)

    # drop unwanted features
    df = df.drop([sdrop, "temp"], axis=1)
    return df
