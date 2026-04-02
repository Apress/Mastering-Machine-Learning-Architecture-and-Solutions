import tensorflow as tf
import numpy as np


def test_model_reproducibility():
    np.random.seed(42)
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, size=100)

    history_1 = model.fit(X_train, y_train, epochs=1, verbose=0)
    history_2 = model.fit(X_train, y_train, epochs=1, verbose=0)

    assert np.isclose(history_1.history['loss'][0], history_2.history['loss'][0],
                      atol=0.01), "Training is not reproducible"
    model.save("saved_model.h5")
    print("Model saved as saved_model.h5")


test_model_reproducibility()
