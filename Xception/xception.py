import keras
from keras import layers

__all__ = ["xception"]

def __entry_flow(inputs):
    # Initial convolutional layers
    x = layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Save a backend of previous blocks
    previous_block = x
    for val in [128, 256, 728]:
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(val, kernel_size=3, strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(val, kernel_size=3, strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(val, kernel_size=3, strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=(2, 2), padding="same")(x)
        # Add residual blocks
        y = layers.Conv2D(val, kernel_size=1, strides=(2, 2), padding="same")(previous_block)
        x = layers.Add()((x, y))
        previous_block = x

    return x


def __middle_flow(x):
    previous_block = x
    # Repeat eight times
    for i in range(8):
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(728, kernel_size=3, strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(728, kernel_size=3, strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(728, kernel_size=3, strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()((x, previous_block))
        previous_block = x
    return x


def __exit_flow(x):
    previous_block = x
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(728, kernel_size=3, strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(1024, kernel_size=3, strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Max pooling layer
    x = layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same")(x)

    # Add residual blocks
    y = layers.Conv2D(1024, kernel_size=1, strides=(2, 2), padding="same")(previous_block)
    x = layers.Add()((x, y))
    x = layers.SeparableConv2D(1536, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(2048, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def xception(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = __entry_flow(inputs)
    x = __middle_flow(x)
    x = __exit_flow(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.models.Model(inputs, outputs)
    return model