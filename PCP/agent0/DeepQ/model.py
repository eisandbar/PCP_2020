"""In this file we define our model structure

CNN was used as it makes the best use of the patterns present on the board
As the board is quite small only a few layers are possible, and their is also limited (2x2)
"""

from tensorflow import keras
from tensorflow.keras import layers

num_actions = 7


def create_q_model():
    inputs = layers.Input(shape=(6, 7, 1))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 2, strides=1, activation="relu")(inputs)
    layer2 = layers.Dropout(0.5)(layer1)
    layer3 = layers.Conv2D(64, 2, strides=1, activation="relu")(layer2)
    layer4 = layers.Dropout(0.5)(layer3)
    layer5 = layers.Conv2D(64, 2, strides=1, activation="relu")(layer4)

    layer6 = layers.Flatten()(layer5)

    layer7 = layers.Dense(256, activation="relu")(layer6)
    action = layers.Dense(num_actions, activation="linear")(layer7)

    return keras.Model(inputs=inputs, outputs=action)
