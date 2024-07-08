import keras
from keras import layers


__all__ = ["lenet"]


def lenet(input_shape, num_classes):
    if not isinstance(input_shape, (tuple, list, keras.KerasTensor)):
        raise TypeError("input_shape must be a tuple or list or keras.KerasTensor")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    input_tensor = keras.Input(input_shape)
    x = layers.Conv2D(filters=6, kernel_size=(5, 5), activation="tanh")(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = layers.Conv2D(filters=16, kernel_size=(5, 5), activation="tanh", strides=(1, 1))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=120, activation="tanh")(x)
    x = layers.Dense(units=84, activation="tanh")(x)
    output = layers.Dense(units=num_classes, activation="softmax")(x)

    model = keras.Model(input_tensor, output)
    return model


lenet((16, 16, 1), 10).summary()
