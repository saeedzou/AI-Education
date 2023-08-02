import math
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt


def load_dataset(train_path, test_path):
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def AlexNet(input_shape, output_shape):
    """

    Args:
        output_shape: number of output classes
        input_shape: shape of input image without batch size e.g) (64, 64, 3)

    Returns:
        model: AlexNet architecture

    """
    i = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu')(i)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=2)(x)
    x = tf.keras.layers.Conv2D(256, (5, 5), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=2)(x)
    x = tf.keras.layers.Conv2D(384, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(384, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.models.Model(i, x)
    return model


def plot_loss_accuracy_vs_epoch(model_fit_output, path_to_save_figure):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(model_fit_output.history['loss'], label='train_loss')
    ax1.plot(model_fit_output.history['val_loss'], label='val_loss')
    ax1.legend()

    ax2.plot(model_fit_output.history['accuracy'], label='train_accuracy')
    ax2.plot(model_fit_output.history['val_accuracy'], label='val_accuracy')
    ax2.legend()
    plt.savefig(path_to_save_figure, bbox_inches='tight')
    plt.show()
