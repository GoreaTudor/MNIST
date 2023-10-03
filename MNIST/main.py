import keras.src.optimizers.optimizer_v1
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense

if __name__ == '__main__':
    (train_input, train_expected), (test_input, test_expected) = mnist.load_data()

    # Data Normalization (from [0..255] to [0..1])
    train_input = train_input / 255.0
    test_input = test_input / 255.0

    # Transform from matrix to array
    train_input = train_input.reshape((-1, 28, 28, 1))
    test_input = test_input.reshape((-1, 28, 28, 1))

    # Build model
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))         # IL
    model.add(Dense(units=128, activation="tanh"))      # HL
    model.add(Dense(units=10, activation="sigmoid"))    # OL

    model.compile(
        # optimizer="SGD",  # learning_rate = 0.01
        optimizer="adam",  # learning_rate = 0.001

        # loss="mean_squared_error",
        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    # train the model
    model.fit(train_input, train_expected, epochs=5, validation_data=(test_input, test_expected))

    # test the model
    test_loss, test_acc = model.evaluate(test_input, test_expected)
    print(f"Accuracy: {test_acc}")
    print(f"Loss: {test_loss}")


# BEST Configs:
# 1. HL(128, relu), OL(softmax), "adam", "sparse_categorical_crossentropy"  --  97.6%
# 2. HL(128, tanh), OL(sigmoid), "adam", "sparse_categorical_crossentropy"  --  97.4%
# 3. HL(128, sigmoid), OL(sigmoid), "adam", "sparse_categorical_crossentropy"  --  97.0%
# 4. HL(128, tanh), OL(sigmoid), "SGD", "sparse_categorical_crossentropy"  --  92.7%
# 5. HL(128, sigmoid), OL(sigmoid), "SGD", "sparse_categorical_crossentropy"  -- 89.8%

# WORST Configs:
# 3. HL(128, tanh), HL(64, tanh), OL(sigmoid), "SGD", "mse"  --  9.3%
# 1. 2xHL(16, sigmoid), OL(sigmoid), "SGD", "mse"  --  9.8%
# 2. 2xHL(16, tanh), OL(sigmoid), "SGD", "mse"  --  10.0%

