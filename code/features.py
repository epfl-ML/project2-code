import enum
import pandas as pd
import typing


def load_features(file: str) -> pd.DataFrame:
    """
    Load features from a csv file.
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
        df[[f + "_mean" for f in features]] = df[features].rolling(window_size).mean()
    if WindowOperationFlag.MEDIAN & op == WindowOperationFlag.MEDIAN:
        df[[f + "_median" for f in features]] = df[features].rolling(window_size).median()
    if WindowOperationFlag.MODE & op == WindowOperationFlag.MODE:
        # TODO : This is particularly slow, since it requires to iterate over the whole window for each step.
        #        Moreover, it may not be particularly relevant for real-valued features. Should we remove it ?
        df[[f + "_mode" for f in features]] = df[features].rolling(window_size).apply(lambda x: pd.Series.mode(x)[0])
    if WindowOperationFlag.VAR & op == WindowOperationFlag.VAR:
        df[[f + "_var" for f in features]] = df[features].rolling(window_size).var()

    return df


data = load_features("data/out.csv")
aggregated = features_window(data, window_size=1, features=["EEGv", "EMGv", "temp"])
print(aggregated)
