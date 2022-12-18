import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history['loss'])
    axs[0].plot(history['val_loss'])
    axs[0].set_title('Model loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_yscale('log')
    axs[0].legend(['train', 'test'], loc='upper right')

    axs[1].plot(history['accuracy'])
    axs[1].plot(history['val_accuracy'])
    axs[1].set_title('Model accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['train', 'test'], loc='upper left')

def plot_df(data, day, log=False, state="state"):
    df = data[data["day"] == day]

    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    eeg = sns.lineplot(data = df, x = "hours", y = "EEGv", hue = state, ax=axs[0])
    axs[0].set_title("EEGv")

    emg = sns.lineplot(data = df, x = "hours", y = "EMGv", hue = state, ax=axs[1])
    axs[1].set_title("EMGv")

    if log:
        eeg.set(yscale="log")
        emg.set(yscale="log")

    plt.show()

def scatter(data, state="state"):
    g = sns.FacetGrid(data, col="day", hue=state, col_wrap=2, height=6)
    g.map(sns.scatterplot, "EEGv", "EMGv", alpha=.50).add_legend()
    g.set(xscale="log", yscale="log")

    plt.show()

def density(df, state="state"):
    sns.set_theme(style="ticks")
    fig = plt.figure(dpi=600)
    g = sns.FacetGrid(df, col="day", hue=state, col_wrap=2, height=6)
    g.map(sns.kdeplot, "EEGv", "EMGv", fill=False, log_scale=True, alpha=.50).add_legend()

    plt.show()

def plot_confusion(model, x_test, y_test, le, cat_matrix, normalize='true'):
    y_pred = model.predict(x_test)

    if cat_matrix:
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

    cm = np.array(confusion_matrix(y_test, y_pred))
    print(cm)

    cm = np.array(confusion_matrix(y_test, y_pred, normalize=normalize)) # normalize = 'true' or 'pred'
    confusion = pd.DataFrame(cm, index=le.classes_, columns=le.classes_ + ' (pred)')


    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    sns.heatmap(confusion, annot=True, cmap="Blues", fmt='.2f')
    plt.title(f'Confusion matrix (normalize = {normalize}')
    plt.show()

def confusion(model, x_test, y_test, le, cat_matrix, normalize='true'):
    y_pred = model.predict(x_test)

    if cat_matrix:
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

    cm = np.array(confusion_matrix(y_test, y_pred))

    return cm