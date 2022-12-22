from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam


def nn_baseline_model(input_dim, num_classes, lr=0.0001):
    """
    Baseline model for the neural network
    Args:
        input_dim: number of features
	num_classes: number of classes
	lr: learning rate
    Returns:
        model: keras model
    """
    model = Sequential()
    model.add(Dense(3, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

    return model
