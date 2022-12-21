import enum
import typing

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

from scipy import stats

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

def add_mean_variance_feature_windows(df, window_sizes, window_features):
    """
    Add mean and variance rolling windows for each window size.
    Args:
        df: the dataframe to transform
        window_sizes: the list of window sizes
        window_features: the list of features to apply the window to
    
    Returns:
        df: the transformed dataframe
    """

    window_names = ['EEGv', 'EMGv']
    for window_size in window_sizes:
        df = features_window(df, window_size=window_size, op=WindowOperationFlag.MEAN, features=window_features)
        df = features_window(df, window_size=window_size, op=WindowOperationFlag.VAR, features=window_features)
        for feature in window_features:
            window_names.append(feature + "_mean" + str(window_size))
            window_names.append(feature + "_var" + str(window_size))

    # drop nan from feature window
    df = df.dropna()

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

    # drop zeroes
    size_before = df1.shape[0]
    for feature in features:
        df1 = df1[df1[feature] > 0]
    size_after = df1.shape[0]
    if size_before != size_after:
        print(f"Removed {size_before - size_after} rows with invalid log values in {file}")

    # apply log
    for feature in features:
        df1[f"{feature}_log"] = np.log(df1[feature])
    return df1

def expand_features_poly(df, max_degree, features=None):
    """
    Expand the dataframe by adding polynomial features.

    Args:
        df: the dataframe to expand
        max_degree: maximum degree of the polynomial
        features: the list of features to expand

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
    """
    Filter the dataframe to keep only the days in the list.
    Args:
        df: the dataframe to filter
        days: the list of days to keep
    
    Returns:
        df: the filtered dataframe
    """

    df1 = df.copy()
    df1["day"] = df1.index // 21600
    return df[df1["day"].isin(days)]

def states(rawState):
    """
    Return the state and rawState columns depending on the rawState flag.
    Args:
        rawState: whether to use the rawState column or not

    Returns:
        skeep: the column to keep
        sdrop: the column to drop
    """

    if rawState:
        return 'rawState', 'state'
    return 'state', 'rawState'

def split_labels(df, useRaw=False):
    """
    Split the dataframe into features and labels.
    Args:
        df: the dataframe to split
        useRaw: whether to use the rawState column or not
    
    Returns:
        x: the features
        y: the labels
    """

    skeep, sdrop = states(useRaw)

    df1 = df.copy()

    x = df1.drop([skeep], axis=1)
    y = df1[skeep]

    return (x, y)
    
def encode_labels(y, cat_matrix=False):
    """
    Encode the labels.
    Args:
        y: the labels to encode
        cat_matrix: whether to return a categorical matrix or not
    
    Returns:
        y1: the encoded labels
        le: the label encoder
    """

    le = LabelEncoder()
    le.fit(y)
    y1 = le.transform(y)
    if cat_matrix:
        y1 = tf.keras.utils.to_categorical(y1)
    return (y1, le)

def decode_labels(le, y):
    """
    Decode the labels.
    Args:
        le: the label encoder
        y: the labels to decode
    
    Returns:
        y1: the decoded labels
    """

    return le.inverse_transform(y)

def split_encode_scale_data(df, useRaw, test_size, seed, cat_matrix):
    """
    Split the dataframe into features and labels, encode the labels, and scale the features. 
    Good for using same mice for train and test

    Args:
        df: the dataframe to split
        useRaw: whether to use the rawState column or not
        test_size: the size of the test set
        seed: the seed for the random state
        cat_matrix: whether to return a categorical matrix or not
    
    Returns:
        x_train: the training features
        x_test: the test features
        y_train: the training labels
        y_test: the test labels
        le: the label encoder
    """

    x, y_raw = split_labels(df, useRaw=useRaw)
    y, le = encode_labels(y_raw, cat_matrix=cat_matrix)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    # Standardize the data
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, le

def split_encode_scale_data_kfold(df, useRaw, seed, cat_matrix):
    """
    Split the dataframe into features and labels, encode the labels, and scale the features.
    Good for using same mice for kfold

    Args:
        df: the dataframe to split
        useRaw: whether to use the rawState column or not
        seed: the seed for the random state
        cat_matrix: whether to return a categorical matrix or not
    
    Returns:
        x: the features
        y: the labels
        le: the label encoder
    """

    x, y_raw = split_labels(df, useRaw=useRaw)
    y, le = encode_labels(y_raw, cat_matrix=cat_matrix)

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    return x, y, le

def encode_scale_data(df_train, df_test, useRaw, seed, cat_matrix):
    """
    Encode the labels, and scale the features.
    Good for using different mice for train and test

    Args:
        df_train: the training dataframe
        df_test: the test dataframe
        useRaw: whether to use the rawState column or not
        seed: the seed for the random state
        cat_matrix: whether to return a categorical matrix or not

    Returns:
        x_train: the training features
        x_test: the test features
        y_train: the training labels
        y_test: the test labels
        le: the label encoder
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

def rebalance_labels(df, label_column = "state"):
    """
        Rebalance the labels in the dataframe
        Args:
            df: dataframe with labels
            label_column: column with labels
        Returns:
            df: dataframe with balanced labels
    """

    balance = df[label_column].value_counts().min()
    df = df.groupby(label_column).apply(lambda x: x.sample(balance)).reset_index(drop=True)
    return df

def remove_outliers_quantile(df, my_features, threshold=0.95):
    """
    Remove outliers from dataframe
    Args:
        df: dataframe with features
        features: list of features to remove outliers from
        threshold: threshold for quantile
    Returns:
        df: dataframe without outliers
    """
    alpha = 1 - threshold
    for feature in my_features:
        q_lower = df[feature].quantile(alpha / 2)
        q_upper = df[feature].quantile(1 - alpha / 2)
        df = df[(df[feature] > q_lower) & (df[feature] < q_upper)]
    
    return df

def spectral_flatness(dataframe):
    """
    Calculate the spectral flatness of the dataframe
    Args:
        dataframe: the dataframe to calculate the spectral flatness of
    Returns:
        dataframe: the dataframe with the spectral flatness column added
    """

    df = dataframe.copy()
    bins = [f"bin{i}" for i in range(401)]

    # sum log of bins
    sum_log = df[bins].apply(lambda x: np.log(x), axis=1).sum(axis=1)
    # divide by number of bins
    mean_log = sum_log / 401
    # exponentiate
    exp = mean_log.apply(lambda x: np.exp(x))
    # divide by sum
    res =  401 * (exp / df[bins].sum(axis=1))

    df['spectral_flatness'] = res
    return df

def spectral_rolloff(dataframe, p):
    """
    Calculate the spectral rolloff of the dataframe
    Args:
        dataframe: the dataframe to calculate the spectral rolloff of
        p: the percentage of the spectral rolloff
    Returns:
        dataframe: the dataframe with the spectral rolloff column added
    """

    df = dataframe.copy()

    bins = [f"bin{i}" for i in range(401)]

    df[f'spectral_rolloff_{p}'] = df[bins].apply(lambda x: np.argmax(x.cumsum() >= x.sum() * p) * 0.25, axis=1)
    
    return df

def spectral_centroid(dataframe):
    """
    Calculate the spectral centroid of the dataframe
    Args:
        dataframe: the dataframe to calculate the spectral centroid of
    Returns:
        dataframe: the dataframe with the spectral centroid column added
    """

    df = dataframe.copy()
    bins = [f"bin{i}" for i in range(401)]

    # weighted sum
    weighted_sum = df[bins].apply(lambda x: np.sum(x * np.arange(401) * 0.25), axis=1)
    sum = df[bins].sum(axis=1)
    
    df['spectral_centroid'] = weighted_sum / sum

    return df

def spectral_entropy(dataframe):
    """
    Calculate the spectral entropy of the dataframe
    Args:
        dataframe: the dataframe to calculate the spectral entropy of
    Returns:
        dataframe: the dataframe with the spectral entropy column added
    """
    
    df = dataframe.copy()
    bins = [f"bin{i}" for i in range(401)]

    # normalize bins
    df2 = df[bins].apply(lambda x: x / x.sum(), axis=1)
    
    def entropy(x):
        return np.sum(x * np.log2(x))

    # calculate entropy
    df['spectral_entropy'] = df2.apply(lambda x: entropy(x), axis=1)

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


def clean_data(data_folder, data_files, days, window_sizes, window_features, rolloffs, dropBins, useRaw, balance=True, standardize_df=False, standardize_features=[]):
    """
    Load, clean, and standardize the data.
    It abstracts away the whole process of loading, cleaning, and standardizing the data.
    
    Args:
        data_folder: the folder containing the data
        data_files: the files to load
        days: the days to keep
        window_sizes: the window sizes to use
        window_features: the features to use for the window
        rolloffs: the rolloffs to use
        dropBins: whether to drop the raw bins or not
        useRaw: whether to use the rawState column or not
        balance: whether to balance the data or not
        standardize_df: whether to standardize the dataframe or not
        standardize_features: the features to standardize

    Returns:
        df: the cleaned dataframe
    """

    df = pd.DataFrame()
    # iterate over several files
    for file in data_files:
        df_temp = load_features(data_folder + file)

        # filter days
        df_temp = filter_days(df_temp, days)

        # add spectral features
        for rolloff in rolloffs:
            df_temp = spectral_rolloff(df_temp, p=rolloff)
        df_temp = spectral_flatness(df_temp)
        df_temp = spectral_centroid(df_temp)
        df_temp = spectral_entropy(df_temp)

        # drop raw bins
        if dropBins:
            for i in range(401):
                df_temp = df_temp.drop([f"bin{i}"], axis=1)

        # add feature window
        df_temp, window_names = add_mean_variance_feature_windows(df, window_sizes, window_features)

        df_temp = log_features(df_temp, window_names)
        
        # add polynomial expansion
        # add trigonometric expansion ?

        df = pd.concat([df, df_temp])

    # balance classes
    skeep, sdrop = states(useRaw)
    if balance:
        df = rebalance_labels(df, skeep)

    # drop unwanted features
    df = df.drop([sdrop, "temp"], axis=1)
    return df

