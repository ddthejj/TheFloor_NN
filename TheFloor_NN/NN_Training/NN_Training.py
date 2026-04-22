import numpy as np
import tensorflow as tf
from tensorflow import keras

MAX_NEIGHBORS = 50
FEATURES = 4
INPUT_SIZE = MAX_NEIGHBORS * FEATURES

data = np.loadtxt("replay_buffer.csv", delimiter=",")

states = data[:, :INPUT_SIZE]
actions = data[:, INPUT_SIZE].astype(int)
rewards = data[:, INPUT_SIZE + 1]
dones = data[:, INPUT_SIZE + 2]

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(INPUT_SIZE,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(MAX_NEIGHBORS)
])

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="mse"
)

gamma = 0.99

for i in range(len(states) - 1):
    target = rewards[i]
    if not dones[i]:
        target += gamma * np.max(model(states[i+1:i+2]))

    q_vals = model(states[i:i+1]).numpy()
    q_vals[0][actions[i]] = target

    model.train_on_batch(states[i:i+1], q_vals)

model.save("model/floor_ai")
