import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *


def identity_block(x, filters, kernels, strides, conv_modes, training=True):
    """
    Identity block in Course 4 Week 2 of Deep Learning Specialization by Andrew NG.
    Args:
        x: output of previous activation of shape (m, n_H_prev, n_W_prev, n_C_prev)
        filters: a list of number of filters for Conv2d layers (the shape of the following
        arguments are the same as this one) e.g) [64, 64, 256]
        kernels: a list of kernels e.g) [(1, 1), (3, 3), (1, 1)]
        strides: a list of strides e.g) [(1, 1), (2, 2), (1, 1)]
        conv_modes: a list of conv modes e.g) ['same', 'valid', 'valid']
        training:

    Returns:
        keras layer
    """
    x_shortcut = x
    L = len(filters)
    for i in range(L):
        x = tf.keras.layers.Conv2D(filters[i], kernels[i], strides[i], padding=conv_modes[i])(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x, training=training)
        if i != L - 1:
            x = tf.keras.layers.ReLU()(x)
        else:
            x = tf.keras.layers.Add()([x_shortcut, x])
    x = tf.keras.layers.ReLU()(x)
    return x


def convolutional_block(x, filters, kernels, strides, conv_modes, training=True):
    """
    Convolutional block in Course 4 Week 2 of Deep Learning Specialization by Andrew NG.
    Args:
        x: output of previous activation of shape (m, n_H_prev, n_W_prev, n_C_prev)
        filters: a list of number of filters for Conv2d layers (the shape of the following
        arguments are the same as this one) e.g) [64, 64, 256]
        notice that the first element is for the shortcut path
        kernels: a list of kernels e.g) [(1, 1), (3, 3), (1, 1)]
        strides: a list of strides e.g) [(1, 1), (2, 2), (1, 1)]
        conv_modes: a list of conv modes e.g) ['same', 'valid', 'valid']
        training:

    Returns:
        keras layer
    """
    x_shortcut = tf.keras.layers.Conv2D(filters[0], kernels[0], strides[0], padding=conv_modes[0])(x)
    x_shortcut = tf.keras.layers.BatchNormalization(axis=3)(x_shortcut, training=training)
    L = len(filters)
    for i in range(L - 1):
        x = tf.keras.layers.Conv2D(filters[i + 1], kernels[i + 1], strides[i + 1], padding=conv_modes[i + 1])(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x, training=training)
        if i + 1 != L - 1:
            x = tf.keras.layers.ReLU()(x)
        else:
            x = tf.keras.layers.Add()([x_shortcut, x])
    x = tf.keras.layers.ReLU()(x)
    return x


def ResNet50(input_shape, output_shape):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE
    Args:
        input_shape: shape of the images of the dataset
        output_shape: integer, number of classes

    Returns:
        keras model
    """
    x_in = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_in)

    x = tf.keras.layers.Conv2D(64, (7, 7), (2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((3, 3), (2, 2))(x)

    x = convolutional_block(x,
                            [256, 64, 64, 256],
                            [(1, 1), (1, 1), (3, 3), (1, 1)],
                            [(1, 1), (1, 1), (1, 1), (1, 1)],
                            ['valid', 'valid', 'same', 'valid'])
    for i in range(2):
        x = identity_block(x,
                           [64, 64, 256],
                           [(1, 1), (3, 3), (1, 1)],
                           [(1, 1), (1, 1), (1, 1)],
                           ['valid', 'same', 'valid']
                           )

    x = convolutional_block(x,
                            [512, 128, 128, 512],
                            [(1, 1), (1, 1), (3, 3), (1, 1)],
                            [(2, 2), (2, 2), (1, 1), (1, 1)],
                            ['valid', 'valid', 'same', 'valid'])
    for i in range(3):
        x = identity_block(x,
                           [128, 128, 512],
                           [(1, 1), (3, 3), (1, 1)],
                           [(1, 1), (1, 1), (1, 1)],
                           ['valid', 'same', 'valid']
                           )

    x = convolutional_block(x,
                            [1024, 256, 256, 1024],
                            [(1, 1), (1, 1), (3, 3), (1, 1)],
                            [(2, 2), (2, 2), (1, 1), (1, 1)],
                            ['valid', 'valid', 'same', 'valid'])
    for i in range(5):
        x = identity_block(x,
                           [256, 256, 1024],
                           [(1, 1), (3, 3), (1, 1)],
                           [(1, 1), (1, 1), (1, 1)],
                           ['valid', 'same', 'valid']
                           )

    x = convolutional_block(x,
                            [2048, 512, 512, 2048],
                            [(1, 1), (1, 1), (3, 3), (1, 1)],
                            [(2, 2), (2, 2), (1, 1), (1, 1)],
                            ['valid', 'valid', 'same', 'valid'])
    for i in range(2):
        x = identity_block(x,
                           [512, 512, 2048],
                           [(1, 1), (3, 3), (1, 1)],
                           [(1, 1), (1, 1), (1, 1)],
                           ['valid', 'same', 'valid']
                           )
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)

    model = tf.keras.models.Model(x_in, x)
    return model


if __name__ == '__main__':
    clf = ResNet50((64, 64, 3), 6)
    clf.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    print(clf.summary())

    # Data preparation and loading
    x_train, y_train, x_test, y_test, _ = load_dataset("./datasets/train_signs.h5",
                                                       "./datasets/test_signs.h5")
    x_train, x_test = x_train / 255., x_test / 255.
    y_train = tf.one_hot(y_train.T.squeeze(), depth=6).numpy()
    y_test = tf.one_hot(y_test.T.squeeze(), depth=6).numpy()

    # Training model for 20 epochs
    r = clf.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

    # Plot training and validation loss and accuracy plots
    plot_loss_accuracy_vs_epoch(r, './3-ResNet_plots.png')

    # Test with arbitrary image (Uncomment the following lines)
    # img_path = './2.jpg'  # change this to your image path
    # img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
    # plt.imshow(img)
    #
    # d = tf.keras.utils.img_to_array(img)
    # d = np.expand_dims(d, axis=0)
    # d = tf.keras.applications.resnet50.preprocess_input(d)
    # plt.title(f'Predicted: {np.argmax(clf.predict(d)[0])}')
    # plt.show()
    # print(clf.predict(d)[0])
