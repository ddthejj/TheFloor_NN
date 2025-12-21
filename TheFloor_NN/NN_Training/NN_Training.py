import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Define your problem's parameters (example: 2D grid, 4 actions)
# You'll need to adjust this based on your game
input_dim = 10  # Example input size (number of features, such as position, score, etc.)
output_dim = 4  # Number of possible actions (e.g., up, down, left, right)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=input_dim, activation='relu'))  # Hidden layer
model.add(Dense(64, activation='relu'))  # Another hidden layer
model.add(Dense(output_dim, activation='softmax'))  # Output layer (probabilities of actions)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dummy input data (replace with your actual game states and corresponding actions)
X_train = np.random.random((1000, input_dim))  # 1000 samples, 10 features
y_train = np.random.randint(0, output_dim, 1000)  # Random action indices

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('floor_game_model.h5')