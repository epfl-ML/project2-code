import enum
import typing

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

def load_features(folder: str, files: list[str]) -> pd.DataFrame:
    """
    Load features from a list of csv files.
    Args:
        file: path to the dataset.
    Returns:
        data: numpy array of shape (N, D).
    """
    df = pd.DataFrame()
    for file in files:
        df1 = pd.read_csv(folder + file, sep=',', header=0)
        df = pd.concat([df, df1], axis=0)
        
    return df


class WindowOperationFlag(enum.IntFlag):
    """
    The different features that can be extracted from a window. If multiple features are selected,
    they will all be extracted from the window.
    """
    MEAN = enum.auto()
    MEDIAN = enum.auto()
    MODE = enum.auto()
    VAR = enum.auto()


def features_window(
        df: pd.DataFrame,
        window_size: int,
        op: WindowOperationFlag = 0,
        features: typing.Union[list[str], None] = None,
) -> pd.DataFrame:
    """
    Smooth features by performing a set of operations over a window of size `window_size`. The implementations must
    select the features to smooth from the `features` argument. If `features` is None, all features will be smoothed.
    Args:
        df: numpy array of shape (N, D).
        window_size: size of the window.
        op: the operation(s) to perform on the window.
        features: the list of features
    Returns:
        data: numpy array of shape (N, D).
    """
    if op == 0:
        op = WindowOperationFlag.MEAN | WindowOperationFlag.MEDIAN | WindowOperationFlag.MODE | WindowOperationFlag.VAR
    if features is None:
        features = list(df.columns)

    # For each operation, add the computed aggregated value.
    if WindowOperationFlag.MEAN & op == WindowOperationFlag.MEAN:
        df[[f + f"_mean{window_size}" for f in features]] = df[features].rolling(window_size).mean()
    if WindowOperationFlag.MEDIAN & op == WindowOperationFlag.MEDIAN:
        df[[f + f"_median{window_size}" for f in features]] = df[features].rolling(window_size).median()
    if WindowOperationFlag.MODE & op == WindowOperationFlag.MODE:
        # TODO : This is particularly slow, since it requires to iterate over the whole window for each step.
        #        Moreover, it may not be particularly relevant for real-valued features. Should we remove it ?
        df[[f + f"_mode{window_size}" for f in features]] = df[features].rolling(window_size).apply(lambda x: pd.Series.mode(x)[0])
    if WindowOperationFlag.VAR & op == WindowOperationFlag.VAR:
        df[[f + f"_var{window_size}" for f in features]] = df[features].rolling(window_size).var()

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
            df[f"{col}_log"] = np.log(df[col])
    return df

def add_times(df) -> pd.DataFrame:
    """
    Add times to the dataframe.
    Args:
        df: numpy array of shape (N, D).
    Returns:
        data: numpy array of shape (N, D+2).
    """
    df1 = df.copy()
    df1["time"] = df1.index
    df1["day"] = df1["time"] // 21600
    return df1


def encode_labels(y, cat_matrix=False):
    le = LabelEncoder()
    le.fit(y)
    y1 = le.transform(y)
    if cat_matrix:
        y1 = tf.keras.utils.to_categorical(y1)
    return (y1, le)

def decode_labels(le, y):
    return le.inverse_transform(y)

def filter_days(df, days):
    return df[df["day"].isin(days)]

def states(rawState):
    if rawState:
        return 'rawState', 'state'
    return 'state', 'rawState'

def split_labels(df, useRaw=False):
    skeep, sdrop = states(useRaw)

    df1 = df.copy()
    df1 = df1.drop([sdrop, "temp", "time", "day"], axis=1)

    x = df1.drop([skeep], axis=1)
    y = df1[skeep]

    return (x, y)

def split_data(df, useRaw, test_size, seed, cat_matrix):
    x, y_raw = split_labels(df, useRaw=useRaw)
    y, le = encode_labels(y_raw, cat_matrix=cat_matrix)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    # Scale on train, or whole data?
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, le