import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def train_model(model, X_train, y_train, epochs, batch_size, learning_rate):
    """
    Trains the deep learning model.
    Implements Algorithm 6 from the thesis. [cite: 981]

    Args:
        model (tf.keras.Model): The compiled Keras model.
        X_train (np.array): Training data (longitudinal MRI sequences).
                            Shape: (num_samples, time_points, H, W, D, Channels)
        y_train (np.array): Training labels (integer-encoded).
        epochs (int): Number of training epochs. [cite: 1151]
        batch_size (int): Batch size for training. [cite: 1151]
        learning_rate (float): Initial learning rate for the Adam optimizer. [cite: 1146]

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    # Ensure y_train is one-hot encoded if using 'categorical_crossentropy'
    # The thesis mentions 'sparse_categorical_crossentropy'[cite: 894, 961], so integer labels are fine.
    
    # The thesis mentions a "scheduler learning rate" but doesn't specify which one.
    # For now, we use a fixed learning rate. You can add a callback for learning rate scheduling.
    # callbacks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate * (0.95 ** epoch))]

    # For managing overfitting[cite: 986, 1153]:
    # L2 regularization is handled in model.py at Dense layer.
    # Dropout is handled in model.py by adding Dropout layers where appropriate.
    # The optimizer (Adam) is set during model compilation in model.py with the specified learning rate.

    # Model is already compiled with Adam optimizer and Cross-Entropy Loss
    # as per build_densenet_bilstm_model in model.py. [cite: 964, 894, 961]

    print(f"Training with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}.")

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        # validation_split=0.1, # Thesis mentions 90% training/validation split, 10% test [cite: 1155]
        # callbacks=callbacks, # Uncomment if using learning rate scheduling
        verbose=1
    )
    return history
