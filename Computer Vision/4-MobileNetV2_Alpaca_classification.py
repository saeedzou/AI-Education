import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *


# Augmenter helper function
def augmenter(x):
    x = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(x)
    return x


def alpaca_model(input_shape=(160, 160, 3), data_augmentation=augmenter):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')  # From imageNet
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, x)
    return model


if __name__ == '__main__':
    # load, shuffle, split data into train, val sets
    train = tf.keras.utils.image_dataset_from_directory("./datasets/alpaca_dataset",
                                                        batch_size=32,
                                                        image_size=(160, 160),
                                                        shuffle=True,
                                                        validation_split=0.2,
                                                        subset='training',
                                                        seed=99)
    val = tf.keras.utils.image_dataset_from_directory("./datasets/alpaca_dataset",
                                                      batch_size=32,
                                                      image_size=(160, 160),
                                                      shuffle=True,
                                                      validation_split=0.2,
                                                      subset='validation',
                                                      seed=99)

    # Display 16 images from train set
    for img, label in train.take(1):
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for i in range(4):
            for j in range(4):
                axes[i, j].imshow(img[i + j].numpy().astype('uint8'))
                axes[i, j].axis('off')
                axes[i, j].set_title("Alpaca" if label[i + j].numpy() == 0 else "Not Alpaca")
        plt.show()

    # Show augmented images of a single image
    for img, _ in train.take(1):
        plt.figure(figsize=(10, 10))
        first_image = img[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = augmenter(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')
        plt.show()

    # MobileNetV2 preprocessor
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Define model and train for 5 epochs
    clf = alpaca_model()
    clf.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

    r = clf.fit(train, validation_data=val, epochs=5)

    # Plot training and validation loss and accuracy plots
    plot_loss_accuracy_vs_epoch(r, './4-MobileNetV2_plots.png')
