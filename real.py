"""
# Venkat Sasank Chavalam
# real.py
# 11/9/2021
"""

# Importing modules
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

# Adjusting the granularity of the report.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Importing dataset.
training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc."
                                             "google.com/mledu-datasets"
                                             "/california_housing_train.csv")

training_df["median_house_value"] /= 1000.0

training_df.head()

training_df.describe()


# Defining functions to train model
def build_model(my_learning_rate):
    """
    Create and compile a simple linear regression model.
    """
    model = tf.keras.models.Sequential()

    # Describing the topography
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, definition, feature, label, epoch_s, size_of_batch):
    """Train the model by feeding it data."""

    # Feeding the model
    history = model.fit(x=definition[feature],
                        y=definition[label],
                        batch_size=size_of_batch,
                        epochs=epoch_s)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    epoch_s = history.epoch

    hist = pd.DataFrame(history.history)

    # Tracking the progression of training
    rms_error = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epoch_s, rms_error


# Defining the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against 200 random training examples."""

    plt.xlabel(feature)
    plt.ylabel(label)

    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    # Creating a red line representing the model.
    x_pointz = 0
    y_pointz = trained_bias
    x_pointo = 10000
    y_pointo = trained_bias + (trained_weight * x_pointo)
    plt.plot([x_pointz, x_pointo], [y_pointz, y_pointo], c='r')

    # Rendering the scatter plot
    plt.show()


def plot_the_loss_curve(epoch_s, rms_error):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epoch_s, rms_error, label="Loss")
    plt.legend()
    plt.ylim([rms_error.min() * 0.97, rms_error.max()])
    plt.show()


LEARNING_RATE = 2
EPOCH_S = 3
BATCH_SIZE = 120

# Specify the feature and the label.
MY_FEATURE = "total_rooms"
# the total number of rooms on a specific city block.
MY_LABEL = "median_house_value"
# the median value of a house on a specific city block.


my_model = build_model(LEARNING_RATE)
weight, bias, epochs, rmse = train_model(my_model, training_df,
                                         MY_FEATURE, MY_LABEL,
                                         EPOCH_S, BATCH_SIZE)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias)

plot_the_model(weight, bias, MY_FEATURE, MY_LABEL)
plot_the_loss_curve(epochs, rmse)


def predict_house_values(number, feature, label):
    """Predict house values based on a feature."""

    batch = training_df[feature][10000:10000 + number]
    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(number):
        print("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                      training_df[label][10000 + i],
                                      predicted_values[i][0]))
