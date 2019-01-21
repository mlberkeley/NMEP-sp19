from keras.models import Model
from keras.layers import Input, add
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout
from keras.regularizers import l2


def rnpa_bottleneck_layer(input_tensor, nb_filters, filter_sz, stage, init='glorot_normal', reg=0.01, use_shortcuts=True):

    nb_in_filters, nb_bottleneck_filters = nb_filters

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = '+' + str(stage)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage>1: # first activation is just after conv1
        x = BatchNormalization(name=bn_name+'a')(input_tensor)
        x = Activation('relu', name=relu_name+'a')(x)
        x = Dropout(0.3)(x)
    else:
        x = input_tensor
    
    x = Conv2D(
            filters=nb_bottleneck_filters, 
            kernel_size=(1,1),
            kernel_initializer=init,
            kernel_regularizer=l2(reg),
            use_bias=False,
            name=conv_name+'a'
        )(x)
    x = Dropout(0.3)(x)

    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = BatchNormalization(name=bn_name+'b')(x)
    x = Activation('relu', name=relu_name+'b')(x)
    x = Conv2D(
            filters=nb_bottleneck_filters, 
            kernel_size=(filter_sz,filter_sz),
            padding='same',
            kernel_initializer=init,
            kernel_regularizer=l2(reg),
            use_bias = False,
            name=conv_name+'b'
        )(x)
    x = Dropout(0.3)(x)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = BatchNormalization(name=bn_name+'c')(x)
    x = Activation('relu', name=relu_name+'c')(x)
    x = Conv2D(
            filters=nb_in_filters, 
            kernel_size=(1,1),
            kernel_initializer=init, 
            kernel_regularizer=l2(reg),
            name=conv_name+'c'
        )(x)
    x = Dropout(0.3)(x)
    # merge
    if use_shortcuts:
        x = add([x, input_tensor])

    return x




def ResNetPreAct(input_shape=(32, 32, 3), layer1_params=(3,128,2), res_layer_params=(3,32,25),
        final_layer_params=None, init='glorot_normal', reg=0.01, use_shortcuts=True):
    
    """
    https://gist.github.com/JefferyRPrice/c1ecc3d67068c8d9b3120475baba1d7e#gistcomment-2321581
    
    Parameters
    ----------
    input_dim : tuple of (C, H, W)
    nb_classes: number of scores to produce from final affine layer (input to softmax)
    layer1_params: tuple of (filter size, num filters, stride for conv)
    res_layer_params: tuple of (filter size, num res layer filters, num res stages)
    final_layer_params: None or tuple of (filter size, num filters, stride for conv)
    init: type of weight initialization to use
    reg: L2 weight regularization (or weight decay)
    use_shortcuts: to evaluate difference between residual and non-residual network
    """

    sz_L1_filters, nb_L1_filters, stride_L1 = layer1_params
    sz_res_filters, nb_res_filters, nb_res_stages = res_layer_params
    
    use_final_conv = (final_layer_params is not None)
    if use_final_conv:
        sz_fin_filters, nb_fin_filters, stride_fin = final_layer_params
        sz_pool_fin = input_shape[1] / (stride_L1 * stride_fin)
    else:
        sz_pool_fin = input_shape[1] / (stride_L1)


    from keras import backend as K


    img_input = Input(shape=input_shape, name='cifar')

    x = Conv2D(
            filters=nb_L1_filters, 
            kernel_size=(sz_L1_filters,sz_L1_filters),
            padding='same',
            strides=(stride_L1, stride_L1),
            kernel_initializer=init,
            kernel_regularizer=l2(reg),
            use_bias=False,
            name='conv0'
        )(img_input)
    
    x = BatchNormalization(name='bn0')(x)
    x = Activation('relu', name='relu0')(x)

    for stage in range(1,nb_res_stages+1):
        x = rnpa_bottleneck_layer(
                x,
                (nb_L1_filters, nb_res_filters),
                sz_res_filters, 
                stage,
                init=init, 
                reg=reg, 
                use_shortcuts=use_shortcuts
            )


    x = BatchNormalization(name='bnF')(x)
    x = Activation('relu', name='reluF')(x)

    if use_final_conv:
        x = Conv2D(
                filters=nb_L1_filters, 
                kernel_size=(sz_L1_filters,sz_L1_filters),
                padding='same',
                strides=(stride_fin, stride_fin),
                kernel_initializer=init,
                kernel_regularizer=l2(reg),
                name='convF'
            )(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(10, activation='softmax', name='fc10')(x)

    return Model(img_input, x, name='rnpa')
  
model = ResNetPreAct()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

import keras.utils as np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

num_classes = 10
epochs = 90

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

x_train = x_train / 255.
x_test = x_test / 255.

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs, validation_data=(x_test, y_test))