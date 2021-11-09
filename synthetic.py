"""
# Venkat Sasank Chavalam
# Tensor Flow
# 11/9/2021
"""

import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf


# Defining functions
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # A sequential model may contain more than one layers.
    model = tf.keras.models.Sequential()

    # Describing the topography of model.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # training to minimize the model's mean squared errors.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, feature, label, epoch_s, batch_size):
    """Train the model by feeding it data."""

    # Feed the feature values and the label values to the model
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=epoch_s)

    # Gather the trained model's weight and bias.
    trained_model_w = model.get_weights()[0]
    trained_model_b = model.get_weights()[1]

    # The list of epoch_s is stored separately from the
    # rest of history.
    epoch_s = history.epoch

    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

    # Specifically gather the model's root mean
    # squared error at each epoch.
    rms_error = hist["root_mean_squared_error"]

    return trained_model_w, trained_model_b, epoch_s, rms_error


# title Define the plotting functions
def plot_the_model(trained_model_w, trained_model_b, feature, label):
    """Plot the trained model against the training feature and label."""

    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")

    # Plot the feature values vs. label values.
    plt.scatter(feature, label)

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    xpoint0 = 0
    ypoint0 = trained_model_b
    xpoint1 = feature[-1]
    ypoint1 = trained_model_b + (trained_model_w * xpoint1)
    plt.plot([xpoint0, xpoint1], [ypoint0, ypoint1], c='r')

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epoch_s, rms_error):
    """
    Plot the loss curve, which shows loss vs. epoch.
    """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epoch_s, rms_error, label="Loss")
    plt.legend()
    plt.ylim([rms_error.min() * 0.97, rms_error.max()])
    plt.show()


my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0,
               11.0, 12.0])
my_label = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0,
             33.8, 38.2])

LEARNING_RATE = 0.01
EPOCH_S = 500
MY_BATCH_SIZE = 12

my_model = build_model(LEARNING_RATE)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, EPOCH_S,
                                                         MY_BATCH_SIZE)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
