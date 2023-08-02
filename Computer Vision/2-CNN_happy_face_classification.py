import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *


def happy_model(input_shape):
    i = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(i)
    x = tf.keras.layers.Conv2D(32, (7, 7), 1, name='conv0')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn0')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), name='max_pool')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(x)
    return tf.keras.models.Model(i, x)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, classes = load_dataset("./datasets/train_happy.h5",
                                                             "./datasets/test_happy.h5")
    x_train, x_test = x_train / 255., x_test / 255.
    y_train, y_test = y_train.T.squeeze(), y_test.T.squeeze()
    model = happy_model((64, 64, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=40, batch_size=16)
    print(model.summary())
    # Plot training and validation loss and accuracy plots
    plot_loss_accuracy_vs_epoch(r, './2-CNN_plots.png')
    # # Test with arbitrary image
    # img_path = './2.jpg'  # change this to your image path
    # img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
    # plt.imshow(img)
    # d = tf.keras.utils.img_to_array(img)
    # d = np.expand_dims(d, axis=0)
    # plt.title('Happy' if model.predict(d)[0, 0] == 1 else "Sad")
    # plt.show()
    # print(model.predict(d))
