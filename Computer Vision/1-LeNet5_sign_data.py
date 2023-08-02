import numpy as np
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt


def LeNet5(input_shape, num_classes):
    i = tf.keras.layers.Input(shape=x_train.shape[1:])
    x = tf.keras.layers.Conv2D(6, (5, 5), padding='same', activation='tanh')(i)
    x = tf.keras.layers.AveragePooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(16, (5, 5), activation='tanh')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120, activation='tanh')(x)
    x = tf.keras.layers.Dense(84, activation='tanh')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    clf = tf.keras.models.Model(i, x)
    return clf


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, classes = load_dataset("./datasets/train_signs.h5",
                                                             "./datasets/test_signs.h5")
    x_train, x_test = x_train / 255., x_test / 255.
    y_train = tf.one_hot(y_train.T.squeeze(), depth=6).numpy()
    y_test = tf.one_hot(y_test.T.squeeze(), depth=6).numpy()

    # Train with LeNet architecture
    model = LeNet5((64, 64, 3), 6)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

    # Plot training and validation loss and accuracy plots
    plot_loss_accuracy_vs_epoch(r, './1-LeNet5_plots.png')
    # Test with arbitrary image (Uncomment the following lines)
    # img_path = './2.jpg'  # change this to your image path
    # img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
    # plt.imshow(img)
    #
    # d = tf.keras.utils.img_to_array(img)
    # d = np.expand_dims(d, axis=0)
    # d = tf.keras.applications.resnet50.preprocess_input(d)
    # plt.title(f'Predicted: {np.argmax(model.predict(d)[0])}')
    # plt.show()
    # print(model.predict(d)[0])
