from keras import layers

__all__ = ["residual_block", "bottleneck_block"]

def residual_block(x, filters, kernel_size=3, stride=1, need_shortcut=False):
    shortcut = x
    if need_shortcut == True:
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=stride)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding="same", strides=stride)(x)
    x  = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()

    # Combine two branches
    x = layers.Add()((x, shortcut))
    x = layers.ReLU()(x)
    return x


def bottleneck_block(x, filters, stride=1, need_shortcut=False):
    shortcut = x
    if need_shortcut == True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride)(x)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Conv2D(filters, 1, strides=stride)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(4 * filters, 1)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x