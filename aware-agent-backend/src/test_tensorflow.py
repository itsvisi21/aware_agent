import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def test_tensorflow():
    """Test TensorFlow installation and basic functionality."""
    print(f"TensorFlow version: {tf.__version__}")
    
    # Create a simple model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(10,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("TensorFlow test completed successfully!")

if __name__ == "__main__":
    test_tensorflow() 