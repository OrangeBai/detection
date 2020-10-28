from models.nets import *


def res50(input_tensor, activation, **kwargs):
    x = conv_layer(input_tensor, 64, (7, 7), strides=(2, 2), activation=activation)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = bn_block(x, 64, (1, 1), activation, 'block_1_1', False, **kwargs)
    x = bn_block(x, 64, (1, 1), activation, 'block_1_2', False, **kwargs)
    x = bn_block(x, 64, (1, 1), activation, 'block_1_3', False, **kwargs)

    x = bn_block(x, 128, (2, 2), activation, 'block_2_1', False, **kwargs)
    x = bn_block(x, 128, (1, 1), activation, 'block_2_2', False, **kwargs)
    x = bn_block(x, 128, (1, 1), activation, 'block_2_3', False, **kwargs)
    x = bn_block(x, 128, (1, 1), activation, 'block_2_4', False, **kwargs)

    x = bn_block(x, 256, (2, 2), activation, 'block_3_1', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_2', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_3', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_4', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_5', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_6', False, **kwargs)

    x = bn_block(x, 512, (2, 2), activation, 'block_4_1', False, **kwargs)
    x = bn_block(x, 512, (1, 1), activation, 'block_4_2', False, **kwargs)
    x = bn_block(x, 512, (1, 1), activation, 'block_4_3', False, **kwargs)

    return x


def darknet_v1(input_tensor, activation, **kwargs):
    x = conv_layer(input_tensor, 64, (7, 7), strides=(2, 2), activation=activation)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = conv_layer(x, 192, (3, 3), activation=activation, name='block1_1')
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = conv_layer(x, 128, (1, 1), activation=activation, name='block2_1')
    x = conv_layer(x, 256, (3, 3), activation=activation, name='block2_2')
    x = conv_layer(x, 256, (1, 1), activation=activation, name='block2_3')
    x = conv_layer(x, 512, (3, 3), activation=activation, name='block2_4')
    x = MaxPooling2D((2, 2), padding='same', strides=(2, 2))(x)

    for i in range(4):
        x = conv_layer(x, 256, (1, 1), activation=activation, name='block3_' + str(2 * i + 1))
        x = conv_layer(x, 512, (3, 3), activation=activation, name='block3_' + str(2 * i + 2))
    x = conv_layer(x, 512, (1, 1), activation=activation, name='block3_9')
    x = conv_layer(x, 1024, (3, 3), activation=activation, name='block3_10')
    x = MaxPooling2D((2, 2), padding='same', strides=(2, 2))(x)

    for i in range(2):
        x = conv_layer(x, 512, (1, 1), activation=activation, name='block4_' + str(2 * i + 1))
        x = conv_layer(x, 1024, (3, 3), activation=activation, name='block4_' + str(2 * i + 2))
    x = conv_layer(x, 1024, (3, 3), activation=activation, name='block4_5')
    x = conv_layer(x, 1024, (3, 3), strides=(2, 2), activation=activation, name='block4_6')

    x = conv_layer(x, 1024, (3, 3), activation=activation, name='block5_1')
    x = conv_layer(x, 1024, (3, 3), activation=activation, name='block5_2')

    return x


def darknet_v3(input_tensor, activation, **kwargs):
    x = conv_layer(input_tensor, 32, (3, 3), activation=activation, **kwargs)
    x = conv_layer(x, 64, (3, 3), strides=(2, 2), padding='same', activation=activation, **kwargs)

    x = darknet_res_block(x, 32, activation=activation, block_name='block1_1', **kwargs)
    x = conv_layer(x, 128, (3, 3), strides=(2, 2), padding='same', activation=activation, **kwargs)

    for i in range(2):
        x = darknet_res_block(x, 64, activation=activation, block_name='block2_' + str(i), **kwargs)
    x = conv_layer(x, 256, (3, 3), strides=(2, 2), padding='same', activation=activation, **kwargs)

    for i in range(8):
        x = darknet_res_block(x, 128, activation=activation, block_name='block3_' + str(i), **kwargs)
    scale1 = x
    x = conv_layer(x, 512, (3, 3), strides=(2, 2), padding='same', activation=activation, **kwargs)

    for i in range(8):
        x = darknet_res_block(x, 256, activation=activation, block_name='block4_' + str(i), **kwargs)
    scale2 = x
    x = conv_layer(x, 1024, (3, 3), strides=(2, 2), padding='same', activation=activation, **kwargs)

    for i in range(4):
        x = darknet_res_block(x, 512,  activation=activation, block_name='block5_' + str(i), **kwargs)

    return [scale1, scale2, x]
