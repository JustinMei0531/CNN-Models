import keras
from keras import layers


__all__ = ["vgg11", "vgg13", "vgg16", "vgg19"]

# The letter 'm' represents a max pooling layer
# In some vgg implementations, capital M is used to represent the maximum pooling layer, but this is not conducive to program maintenance, 
# so here -1 is used to represent the maximum pooling layer, so that the element type in the list is unique
vgg_config = dict(
    vgg11 = [64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1],
    vgg13 = [64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1],
    vgg16 = [64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1],
    vgg19 = [64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1],
)


def __vgg(vgg_name, input_shape, num_classes):
    """
    Constructs a VGG model using the configuration specified by `vgg_name`. Do not use this function directly.

    Parameters:
    vgg_name (str): Name of the VGG model configuration to use (e.g., 'vgg11', 'vgg13', 'vgg16', 'vgg19').
    input_shape (tuple or list): Shape of the input tensor (e.g., (224, 224, 3)).
    num_classes (int): Number of output classes for classification.

    Returns:
    keras.Model: Constructed VGG model.

    Raises:
    TypeError: If `input_shape` is not a tuple or list, or if `num_classes` is not an integer.
    ValueError: If `vgg_name` is not a string or not found in the VGG configuration.
    """
    if not isinstance(input_shape, (tuple, list, keras.KerasTensor)):
        raise TypeError("input_shape must be a tuple or list or keras.KerasTensor")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    if not isinstance(vgg_name, str) or vgg_name not in vgg_config:
        raise ValueError("Can not find model configuration from vgg config")
    vgg_layers = vgg_config[vgg_name]
    input_tensor = keras.Input(input_shape)
    x = input_tensor
    for layer in vgg_layers:
        if layer == -1:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        else:
            x = layers.Conv2D(filters=layer, kernel_size=(3, 3), padding="same", activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x= layers.Dropout(.5)(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x= layers.Dropout(.5)(x)
    outputs = layers.Dense(units=num_classes, activation="softmax")(x)

    model = keras.Model(input_tensor, outputs)

    return model


def vgg11(input_shape, num_classes):
    return __vgg("vgg11", input_shape, num_classes)

def vgg13(input_shape, num_classes):
    return __vgg("vgg13", input_shape, num_classes)

def vgg16(input_shape, num_classes):
    return __vgg("vgg16", input_shape, num_classes)

def vgg19(input_shape, num_classes):
    return __vgg("vgg19", input_shape, num_classes)

