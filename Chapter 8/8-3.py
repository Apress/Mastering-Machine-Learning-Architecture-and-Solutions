import numpy as np
import tensorflow as tf
def test_model_inference():
    np.random.seed(42)
    X_test = np.random.rand(10, 5)  # Simulating test samples

    model = tf.keras.models.load_model("saved_model.h5")  # Load a trained model
    predictions = model.predict(X_test)

    assert np.all(predictions >= 0) and np.all(predictions <= 1), "Inference outputs are not valid probabilities"
    print("Random test passed — predictions are valid probabilities.")
    print("Random predictions:\n", predictions)


    # =============================
    # 2. MANUAL NUMERICAL TEST INPUT
    # =============================
    manual_input = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [1.0, 0.0, 0.5, 0.2, 0.1],
        [0.9, 0.9, 0.9, 0.9, 0.9],
    ])

    manual_predictions = model.predict(manual_input)

    # Validate range again
    assert np.all(manual_predictions >= 0) and np.all(manual_predictions <= 1), \
        "Manual inference outputs are not valid probabilities"

    print("\nManual numerical test passed.")
    print("Manual inputs:\n", manual_input)
    print("Manual predictions:\n", manual_predictions)test_model_inference()
