import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


def new_model(train_x, train_y, val_x, val_y):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))

    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=3, validation_data=(val_x, val_y))
    model.save("model.h5")


def ask_retrain_model():
    try:
        tf.keras.models.load_model("model.h5")
        retrain = input("Model found. Would you like to retrain (Y/N)? ").lower() == "y"
        if retrain:
            new_model(x_train, y_train, x_test, y_test)

    except IOError:
        print("No model found. Starting to train...")
        new_model(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    ask_retrain_model()
    model = tf.keras.models.load_model("model.h5")
    predict = model.predict(x_test)
    for i in range(10):
        plt.figure(i)
        plt.imshow(x_test[i])
        plt.title(f"Predict: {np.argmax(predict[i])}")

    plt.show()
