import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy

if __name__ == '__main__':
    (train_input, train_expected), (test_input, test_expected) = mnist.load_data()

    # Data Normalization (from [0..255] to [0..1])
    train_input = train_input.astype("float32") / 255
    test_input = test_input.astype("float32") / 255

    # Make sure that images have shape (28, 28, 1)
    train_input = numpy.expand_dims(train_input, -1)
    test_input = numpy.expand_dims(test_input, -1)

    # class vectors => binary class matrices
    train_expected = to_categorical(train_expected, 10)
    test_expected = to_categorical(test_expected, 10)

    # Build model
    model = Sequential([
        keras.Input(shape=(28, 28, 1)),
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(),  # learning_rate = 0.001
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # train the model
    model.fit(train_input, train_expected, epochs=10, batch_size=128, validation_data=(test_input, test_expected))

    # test the model
    test_loss, test_acc = model.evaluate(test_input, test_expected)
    print(f"Accuracy: {test_acc}")
    print(f"Loss: {test_loss}")
