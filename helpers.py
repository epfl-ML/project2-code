import numpy as np
import functions_ml as impl
import pandas as pd

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

def cross_validation(df, k_fold, lambda_, seed):
    """
    Perform a cross validation on the data.
    Args:
        df: a cleaned dataframe to use
        k_fold: number of folds
        lambda_: regularization parameter
        expand_trig: whether to expand the trigonometric features or not
        seed: seed for the random number generator

    Returns:
        the loss and accuracy of the model
    """

    losses_tr = []
    losses_te = []
    precisions = []
    recalls = []
    f1_scores = []
    accuracys = []
    tps = []
    tns = []
    fps = []
    fns = []

    splits = build_k_splits(df, k_fold, seed)

    for k in range(k_fold):
        train = pd.concat(splits[:k] + splits[k + 1 :])
        test = splits[k]

        # Split the data
        y_tr, tx_tr = split_label_features(train)
        y_te, tx_te = split_label_features(test)


        # train the model
        weights, _ = impl.ridge_regression(y_tr, tx_tr, lambda_)

        # compute the scores
        loss_tr = np.sqrt(2 * impl.compute_loss_mse(y_tr, tx_tr, weights))
        loss_te = np.sqrt(2 * impl.compute_loss_mse(y_te, tx_te, weights))
        ((precision, recall, f1_score, accuracy), (tp, tn, fp, fn)) = compute_scores(
            y_te, tx_te, weights
        )


        # save the scores
        losses_tr.append(loss_tr)
        losses_te.append(loss_te)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        accuracys.append(accuracy)
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)

    # compute the mean scores
    loss_tr = np.mean(losses_tr)
    loss_te = np.mean(losses_te)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1_score = np.mean(f1_scores)
    accuracy = np.mean(accuracys)
    tps = np.mean(tps)
    tns = np.mean(tns)
    fps = np.mean(fps)
    fns = np.mean(fns)
    total = tps + tns + fps + fns

    tps = tps / total
    tns = tns / total
    fps = fps / total
    fns = fns / total

    return loss_tr, loss_te, precision, recall, f1_score, accuracy, (tps, tns, fps, fns)


def run_cross_validation(
    df_raw,
    max_invalid_ratio,
    k_fold,
    lambdas,
    degrees,
    expand_trig,
    seed,
    print_best,
):
    """
    Run cross validation on the data.
    Args:
        df_raw: dataframe to use
        max_invalid_ratio: maximum ratio of invalid values in a feature
        k_fold: number of folds
        lambdas: list of regularization parameters
        degrees: list of degrees of the polynomial expansion
        expand_trig: whether to expand the trigonometric features or not
        seed: seed for the random number generator
        print_best: whether to print the best results or not to the console

    Returns:
        a tuple containing the best lambda and degree, and the corresponding scores
    """

    # Define the variables to store the best results
    best_lambda = None
    best_degree = None
    best_scores = None

    # For each lambda and degree, run the cross validation
    accs = np.zeros((len(lambdas), len(degrees)), dtype=np.float64)
    for i, lambda_ in enumerate(lambdas):
        for j, degree in enumerate(degrees):
            # Clean data with given hyper-parameters
            train = clean_data(
                df_raw, max_invalid_ratio, degree, expand_trig
            )

            # Compute the loss and accuracy for each fold
            (
                loss_tr,
                loss_te,
                precision,
                recall,
                f1_score,
                accuracy,
                confusion,
            ) = cross_validation(train, k_fold, lambda_, seed)

            accs[i, j] = accuracy

            # Print the results
            print(
                f"lambda={lambda_:.2e}, degree={degree:2d}, loss_tr={loss_tr:.2e}, loss_te={loss_te:.2e}, precision={precision:.3f}, recall={recall:.3f}, f1_score={f1_score:.3f}, accuracy={accuracy:.5f}"
            )

            # Update the best results if necessary
            if best_scores is None or accuracy > best_scores[5]:
                best_lambda = lambda_
                best_degree = degree
                best_scores = (
                    loss_tr,
                    loss_te,
                    precision,
                    recall,
                    f1_score,
                    accuracy,
                    confusion,
                )

    if print_best:
        loss_tr, loss_te, precision, recall, f1_score, accuracy, confusion = best_scores
        print(f"Best lambda:    {best_lambda:.2e}")
        print(f"Best degree:    {best_degree}")
        print(f"Loss_tr:        {loss_tr:.2e}")
        print(f"Loss_te:        {loss_te:.2e}")
        print(f"Precision:      {precision:.5f}")
        print(f"Recall:         {recall:.5f}")
        print(f"F1 score:       {f1_score:.5f}")
        print(f"Accuracy:       {accuracy:.5f}")
        print(
            f"tp, tn, fp, fn: {confusion[0]:.1%}, {confusion[1]:.1%}, {confusion[2]:.1%}, {confusion[3]:.1%}"
        )

    return accs


def build_k_splits(df, k, seed):
    """
    Split a dataframe into k folds.
    Args:
        df: dataframe to split
        k: number of folds
        seed: seed for the random number generator

    Returns:
        a list of k dataframes
    """
    splits = []
    size = len(df) // k
    d = df.copy(deep=True)
    for _ in range(0, k):
        sample = d.sample(n=min(size, len(d)), random_state=seed)
        splits += [sample]
        d = d.drop(index=sample.index)
    return splits

def compute_scores(y, tx, w):
    """
    Compute the loss and accuracy of a model.
    Args:
        y: true labels
        tx: data
        w: model weights

    Returns:
        a tuple containing the scores
    """
    #  create prediction
    predictions = tx @ w
    predictions = np.sign(predictions)

    # Compute the scores
    tp = np.sum(np.logical_and(predictions == 1, y == 1))
    tn = np.sum(np.logical_and(predictions == -1, y == -1))
    fp = np.sum(np.logical_and(predictions == 1, y == -1))
    fn = np.sum(np.logical_and(predictions == -1, y == 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return ((precision, recall, f1_score, accuracy), (tp, tn, fp, fn))
