import keras
from keras import layers

__all__ = [
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn"
]


vgg_config = dict(
    vgg11 = [64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1],
    vgg13 = [64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1],
    vgg16 = [64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1],
    vgg19 = [64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1],
)



def __vggbn(vggbn_name, input_shape, num_classes):
    if not isinstance(input_shape, (tuple, list, keras.KerasTensor)):
        raise TypeError("input_shape must be a tuple or list or keras.KerasTensor")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    if not isinstance(vggbn_name, str) or vggbn_name not in vgg_config:
        raise ValueError("Can not find model configuration from vgg config")
    
    input_tensor = keras.Input(input_shape)
    vgg_layers = vgg_config[vggbn_name]
    input_tensor = keras.Input(shape=input_shape)
    x = input_tensor
    
    for layer in vgg_layers:
        if layer == -1:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        else:
            x = layers.Conv2D(filters=layer, kernel_size=(3, 3), padding="same", activation=None)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=num_classes, activation="softmax")(x)

    model = keras.Model(input_tensor, outputs)
    return model


def vgg11_bn(input_shape, num_classes):
    return __vggbn("vgg11", input_shape, num_classes)


def vgg13_bn(input_shape, num_classes):
    return __vggbn("vgg13", input_shape, num_classes)


def vgg16_bn(input_shape, num_classes):
    return __vggbn("vgg16", input_shape, num_classes)

def vgg19_bn(input_shape, num_classes):
    return __vggbn("vgg19", input_shape, num_classes)


vgg19_bn((224, 224, 3), 10).summary()