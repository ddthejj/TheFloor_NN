import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

MAX_NEIGHBORS = 50
FEATURES = 4
INPUT_SIZE = MAX_NEIGHBORS * FEATURES
DEFAULT_DATA_PATH = Path("ml/replay_buffer.csv")
DEFAULT_MODEL_PATH = Path("model/floor_ai.keras")


def load_replay_buffer(csv_path: Path):
    """Load and split replay buffer rows into (s, a, r, done, s')."""
    data = np.loadtxt(csv_path, delimiter=",", ndmin=2)

    if data.shape[1] < INPUT_SIZE + 3:
        raise ValueError(
            f"Replay buffer has {data.shape[1]} columns, but expected at least {INPUT_SIZE + 3}."
        )

    states = data[:, :INPUT_SIZE].astype(np.float32)
    actions = data[:, INPUT_SIZE].astype(np.int32)
    rewards = data[:, INPUT_SIZE + 1].astype(np.float32)
    dones = data[:, INPUT_SIZE + 2].astype(np.bool_)

    return states, actions, rewards, dones


def build_q_network(input_size: int, num_actions: int):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_size,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(num_actions),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_dqn(
    model,
    states,
    actions,
    rewards,
    dones,
    gamma=0.99,
    batch_size=128,
    epochs=10,
):
    """Offline fitted Q iteration over replay buffer."""
    num_samples = len(states)
    if num_samples < 2:
        raise ValueError("Need at least 2 replay rows to build s -> s' transitions.")

    # s_t, a_t, r_t, done_t, s_(t+1)
    s_t = states[:-1]
    a_t = actions[:-1]
    r_t = rewards[:-1]
    d_t = dones[:-1]
    s_tp1 = states[1:]

    for epoch in range(epochs):
        # Bootstrap targets from current model
        q_next = model.predict(s_tp1, verbose=0)
        max_q_next = np.max(q_next, axis=1)

        q_target = model.predict(s_t, verbose=0)
        td_target = r_t + (1.0 - d_t.astype(np.float32)) * gamma * max_q_next
        q_target[np.arange(len(s_t)), a_t] = td_target

        history = model.fit(s_t, q_target, batch_size=batch_size, epochs=1, verbose=0)
        print(f"epoch {epoch + 1:>2}/{epochs}: loss={history.history['loss'][0]:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train The Floor Q-network from replay buffer CSV.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to replay_buffer.csv")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_MODEL_PATH, help="Output model path (.keras preferred)"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Replay buffer not found at '{args.data}'. Run the C++ game to generate data first."
        )

    states, actions, rewards, dones = load_replay_buffer(args.data)
    print(f"Loaded {len(states)} replay rows from {args.data}")

    model = build_q_network(INPUT_SIZE, MAX_NEIGHBORS)
    train_dqn(
        model,
        states,
        actions,
        rewards,
        dones,
        gamma=args.gamma,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    # Set deterministic-ish behavior for easier debugging
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
