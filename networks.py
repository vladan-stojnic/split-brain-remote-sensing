from keras.layers import Input, ZeroPadding2D, Conv2D, Activation, BatchNormalization, MaxPooling2D
from keras.regularizers import l2
from keras.models import Model

def AlexNet(input_shape, output_channels, kernel_initializer = 'glorot_uniform'):
    input = Input(input_shape, name = 'input')
    x = ZeroPadding2D(padding = 5)(input)
    x = Conv2D(48, 11, strides = (4, 4), dilation_rate = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = l2(0.0001), bias_regularizer = l2(0.0001), name = 'conv1')(x)
    x = BatchNormalization(name = 'bn1')(x)
    x = Activation('relu', name = 'act1')(x)
    x = ZeroPadding2D(padding = 1)(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'pool1')(x)

    x = ZeroPadding2D(padding = 2)(x)
    x = Conv2D(128, 5, strides = (1, 1), dilation_rate = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = l2(0.0001), bias_regularizer = l2(0.0001), name = 'conv2')(x)
    x = BatchNormalization(name = 'bn2')(x)
    x = Activation('relu', name = 'act2')(x)
    x = ZeroPadding2D(padding = 1)(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'pool2')(x)

    x = ZeroPadding2D(padding = 1)(x)
    x = Conv2D(192, 3, strides = (1, 1), dilation_rate = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = l2(0.0001), bias_regularizer = l2(0.0001), name = 'conv3')(x)
    x = BatchNormalization(name = 'bn3')(x)
    x = Activation('relu', name = 'act3')(x)

    x = ZeroPadding2D(padding = 1)(x)
    x = Conv2D(192, 3, strides = (1, 1), dilation_rate = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = l2(0.0001), bias_regularizer = l2(0.0001), name = 'conv4')(x)
    x = BatchNormalization(name = 'bn4')(x)
    x = Activation('relu', name = 'act4')(x)

    x = ZeroPadding2D(padding = 1)(x)
    x = Conv2D(128, 3, strides = (1, 1), dilation_rate = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = l2(0.0001), bias_regularizer = l2(0.0001), name = 'conv5')(x)
    x = BatchNormalization(name = 'bn5')(x)
    x = Activation('relu', name = 'act5')(x)
    x = ZeroPadding2D(padding = 1)(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (1, 1), name = 'pool5')(x)

    x = ZeroPadding2D(padding = 5)(x)
    x = Conv2D(2048, 6, strides = (1, 1), dilation_rate = (2, 2), kernel_initializer = kernel_initializer, kernel_regularizer = l2(0.0001), bias_regularizer = l2(0.0001), name = 'fc6')(x)
    x = Activation('relu', name = 'act6')(x)

    x = Conv2D(2048, 1, strides = (1, 1), dilation_rate = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = l2(0.0001), bias_regularizer = l2(0.0001), name = 'fc7')(x)
    x = Activation('relu', name = 'act7')(x)

    x = Conv2D(output_channels, 1, strides = (1, 1), dilation_rate = (1, 1), kernel_initializer = kernel_initializer, kernel_regularizer = l2(0.0001), bias_regularizer = l2(0.0001), name = 'fc8')(x)
    x = Activation('softmax', name = 'act8')(x)

    model = Model(inputs = input, outputs = x)

    return model
