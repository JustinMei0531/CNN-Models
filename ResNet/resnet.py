from keras import layers, KerasTensor
import keras
from blocks import residual_block, bottleneck_block


__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]

resnet_config = {
    "18": (2, 2, 2, 2),
    "34": (3, 4, 6, 3),
    "50": (3, 4, 6, 3),
    "101": (3, 4, 23, 3),
    "152": (3, 8, 36, 3),
}

# Defining two shallow residual networks
def __basic_resnet(version, input_shape, num_classes):
    version_control = ("18", "34")
    if version not in version_control:
        raise ValueError("There is no corresponding version of the structure.")
    if not isinstance(input_shape, (tuple, list, KerasTensor)):
        raise TypeError("input_shape must be a tuple or list or keras.KerasTensor")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    # Get resnet configuration
    config = resnet_config.get(version)
    
    # Initial convolutional and max pooling layers
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=7, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    

    # Stack residual blocks
    for index, numbers in enumerate(config):
        filters = 64 * (2 ** index)
        for j in range(numbers):
            if j == 0 and index > 0:
                x = residual_block(x, filters, stride=2, need_shortcut=True)
            else:
                x = residual_block(x, filters, stride=1, need_shortcut=False)
    
    # Global average pooling and output layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs, outputs)
    return model

# Define three deeper resnets
def __bottleneck_resnet(version, input_shape, num_classes):
    version_control = ("50", "101", "152")
    if version not in version_control:
        raise ValueError("There is no corresponding version of the structure.")
    if not isinstance(input_shape, (tuple, list, KerasTensor)):
        raise TypeError("input_shape must be a tuple or list or keras.KerasTensor")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    
     # Get resnet configuration
    config = resnet_config.get(version)
    
    # Initial convolution and max pooling
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Stack bottleneck blocks
    for index, number in enumerate(config):
        filters = 64 * (2 ** index)
        for j in range(number):
            if j == 0 and index > 0:
                x = bottleneck_block(x, filters, stride=2, need_shortcut=True)
            else:
                x = bottleneck_block(x, filters, stride=1, need_shortcut=False)
    
    # Global average pooling and output layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs, outputs)
    return model

def resnet18(input_shape, num_classes):
    return __basic_resnet("18", input_shape, num_classes)

def resnet34(input_shape, num_classes):
    return __basic_resnet("34", input_shape, num_classes)

def resnet50(input_shape, num_classes):
    return __bottleneck_resnet("50", input_shape, num_classes)

def resnet101(input_shape, num_classes):
    return __bottleneck_resnet("101", input_shape, num_classes)

def resnet152(input_shape, num_classes):
    return __bottleneck_resnet("152", input_shape, num_classes)