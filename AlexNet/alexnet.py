import keras
from keras import layers

__all__ = ["alexnet"]

def alexnet(input_shape, num_classes):
    """
    Constructs the AlexNet model using the Functional API of Keras.

    Parameters:
    input_shape (tuple): Shape of the input tensor, e.g., (227, 227, 3)
    num_classes (int): Number of output classes for classification

    Returns:
    keras.Model: AlexNet model
    """
    if not isinstance(input_shape, (tuple, list)):
        raise TypeError("input_shape must be a tuple or list")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    input_tensor = keras.Input(shape=input_shape)
    # First convolutional layer
    x = layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="valid", activation="relu")(input_tensor)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
    # Second convolutional layer
    x = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
    # Third, Fourth, and Fifth Convolutional Layers
    x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x = layers.Dense(units=num_classes, activation="softmax")(x)

    # Create model
    model = keras.Model(input_tensor, x)
    return model