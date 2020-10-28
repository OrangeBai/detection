from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
import tensorflow as tf


def dense_layer(input_layer, units, activation, batch_norm=True, **kwargs):
    x = Dense(units, **kwargs)(input_layer)
    if batch_norm:
        x = BatchNormalization()(x)
    x = activation(x)
    return x


def dense_block(x, structure):
    for layer_setting in structure:
        x = dense_layer(x, **layer_setting)

    return x


def conv_layer(x, filters, kernel_size, strides=(1, 1), padding='same', activation=None, batch_norm=True, **kwargs):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, **kwargs)(x)
    if batch_norm:
        x = BatchNormalization()(x)

    if activation:
        if activation == 'LeakyReLU':
            x = LeakyReLU(alpha=0.1)(x)
        else:
            x = Activation(activation)(x)
    return x


def bn_block(input_tensor, filters, stride, activation, block_name, identity_block, **kwargs):
    """
    bottle net block for ResNet
    Bottle net block for ResNet
    :param input_tensor: Input tensor
    :param filters:
    :param stride:
    :param activation:
    :param block_name: block name
    :param identity_block: If True, set as identity block
    :return:
    """
    shortcut = input_tensor
    x = conv_layer(input_tensor,
                   filters=filters,
                   kernel_size=(1, 1),
                   strides=stride,
                   activation=activation,
                   name=block_name + '_Conv1',
                   **kwargs)
    x = conv_layer(x,
                   filters=filters,
                   kernel_size=(3, 3),
                   activation=activation,
                   name=block_name + '_Conv2',
                   **kwargs)
    x = conv_layer(x,
                   filters=filters * 4,
                   kernel_size=(1, 1),
                   activation=None,
                   name=block_name + '_Conv3',
                   **kwargs)

    if not identity_block:
        shortcut = conv_layer(input_tensor,
                              filters=filters * 4,
                              kernel_size=(1, 1),
                              strides=stride,
                              activation=None,
                              name=block_name + '_shortcut',
                              **kwargs)

    x = Add()([x, shortcut])

    if activation == 'LeakyReLU':
        x = LeakyReLU(alpha=0.1)(x)
    else:
        x = Activation(activation)(x)

    return x


def res_block(input_tensor, filters, stride, activation, block_name, identity_block, **kwargs):
    """
    Residual block for ResNet
    :param activation:
    :param stride:
    :param filters:
    :param input_tensor: Input tensor
    :param block_name: block name
    :param identity_block:  if True, the block is set to be identity block
                            else, the block is set to be bottleneck block
    :return:Output tensor
    """
    shortcut = input_tensor
    x = conv_layer(input_tensor,
                   filters=filters,
                   kernel_size=(3, 3),
                   strides=stride,
                   activation=activation,
                   name=block_name + '_Conv1',
                   **kwargs)

    x = conv_layer(x,
                   filters=filters,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation=None,
                   name=block_name + '_Conv2',
                   **kwargs)

    if not identity_block:
        shortcut = conv_layer(shortcut,
                              filters=filters,
                              kernel_size=(1, 1),
                              strides=stride,
                              activation=None,
                              name=block_name + 'shortcut',
                              **kwargs)

    x = Add()([x, shortcut])
    x = Activation(activation)(x)

    return x


def darknet_res_block(input_tensor, filters, activation, block_name, **kwargs):
    """
        Residual block for ResNet
        :param activation:
        :param filters:
        :param input_tensor: Input tensor
        :param block_name: block name
        :return:Output tensor
        """
    shortcut = input_tensor
    x = conv_layer(input_tensor,
                   filters=filters,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation=activation,
                   name=block_name + '_Conv1',
                   **kwargs)

    x = conv_layer(x,
                   filters=2 * filters,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation=None,
                   name=block_name + '_Conv2',
                   **kwargs)

    x = Add()([x, shortcut])
    x = Activation(activation)(x)

    return x


def darknet_conv(x, filters, activation, **kwargs):
    x = conv_layer(x,
                   filters=filters,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation=activation,
                   **kwargs)
    x = conv_layer(x,
                   filters=filters * 2,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation=activation,
                   **kwargs)
    x = conv_layer(x,
                   filters=filters * 2,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation=activation,
                   **kwargs)
    x = conv_layer(x,
                   filters=filters,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation=activation,
                   **kwargs)
    x = conv_layer(x,
                   filters=filters * 2,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation=activation,
                   **kwargs)
    return x


def yolo_v3_output(x, filters, anchors, classes, activation, name=None):
    x = conv_layer(x, filters * 2, (3,3), activation=activation)
    x = conv_layer(x, anchors * (classes + 5), (3,3), activation=activation, batch_norm=False)
    x = Reshape((-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5))(x)
    return x

