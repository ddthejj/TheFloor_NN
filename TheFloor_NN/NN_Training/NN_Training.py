import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

MAX_NEIGHBORS = 50
FEATURES = 5
INPUT_SIZE = MAX_NEIGHBORS * FEATURES
DEFAULT_DATA_PATH = Path("ml/replay_buffer.csv")
DEFAULT_MODEL_PATH = Path("model/floor_ai.keras")
DEFAULT_SAVED_MODEL_PATH = Path("model/floor_ai_savedmodel")
DEFAULT_TFLITE_MODEL_PATH = Path("model/floor_ai.tflite")
DEFAULT_NORM_PATH = Path("model/floor_ai.norm.json")


def load_replay_buffer(csv_path: Path):
    """Load and split replay buffer rows into (s, a, r, done, s')."""
    skip_rows = 0
    with csv_path.open("r", encoding="utf-8") as csv_file:
        first_line = csv_file.readline().strip()
        if first_line and any(ch.isalpha() for ch in first_line):
            skip_rows = 1

    data = np.loadtxt(csv_path, delimiter=",", ndmin=2, skiprows=skip_rows)

    if data.shape[1] < INPUT_SIZE + 3:
        raise ValueError(
            f"Replay buffer has {data.shape[1]} columns, but expected at least {INPUT_SIZE + 3}."
        )

    states = data[:, :INPUT_SIZE].astype(np.float32)
    actions = data[:, INPUT_SIZE].astype(np.int32)
    rewards = data[:, INPUT_SIZE + 1].astype(np.float32)
    dones = data[:, INPUT_SIZE + 2].astype(np.bool_)

    if np.any((actions < 0) | (actions >= MAX_NEIGHBORS)):
        invalid = int(np.sum((actions < 0) | (actions >= MAX_NEIGHBORS)))
        raise ValueError(
            f"Replay buffer contains {invalid} actions outside [0, {MAX_NEIGHBORS - 1}]"
        )

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


def normalize_states(states: np.ndarray):
    """Z-normalize states and return (normalized_states, mean, std)."""
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (states - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_dqn(
    model,
    states,
    actions,
    rewards,
    dones,
    gamma=0.99,
    batch_size=128,
    epochs=10,
    val_split=0.1,
    target_update=0.05,
):
    """Offline fitted Q iteration over replay buffer with target network."""
    num_samples = len(states)
    if num_samples < 2:
        raise ValueError("Need at least 2 replay rows to build s -> s' transitions.")

    # s_t, a_t, r_t, done_t, s_(t+1)
    s_t = states[:-1]
    a_t = actions[:-1]
    r_t = rewards[:-1]
    d_t = dones[:-1]
    s_tp1 = states[1:]

    split_idx = int(len(s_t) * (1.0 - val_split))
    split_idx = max(1, min(split_idx, len(s_t) - 1))

    train_slice = slice(0, split_idx)
    val_slice = slice(split_idx, len(s_t))

    s_train, a_train, r_train, d_train, s_tp1_train = (
        s_t[train_slice],
        a_t[train_slice],
        r_t[train_slice],
        d_t[train_slice],
        s_tp1[train_slice],
    )
    s_val, a_val, r_val, d_val, s_tp1_val = (
        s_t[val_slice],
        a_t[val_slice],
        r_t[val_slice],
        d_t[val_slice],
        s_tp1[val_slice],
    )

    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    for epoch in range(epochs):
        # Bootstrap targets from current model
        q_next = target_model.predict(s_tp1_train, verbose=0)
        max_q_next = np.max(q_next, axis=1)

        q_target = model.predict(s_train, verbose=0)
        td_target = r_train + (1.0 - d_train.astype(np.float32)) * gamma * max_q_next
        q_target[np.arange(len(s_train)), a_train] = td_target

        val_next = target_model.predict(s_tp1_val, verbose=0)
        val_max_q_next = np.max(val_next, axis=1)
        val_q_target = model.predict(s_val, verbose=0)
        val_td_target = r_val + (1.0 - d_val.astype(np.float32)) * gamma * val_max_q_next
        val_q_target[np.arange(len(s_val)), a_val] = val_td_target

        history = model.fit(
            s_train,
            q_target,
            validation_data=(s_val, val_q_target),
            batch_size=batch_size,
            epochs=1,
            verbose=0,
        )
        loss = history.history["loss"][0]
        val_loss = history.history["val_loss"][0]
        print(f"epoch {epoch + 1:>2}/{epochs}: loss={loss:.6f} val_loss={val_loss:.6f}")

        online_weights = model.get_weights()
        target_weights = target_model.get_weights()
        blended = [
            target_update * online + (1.0 - target_update) * target
            for online, target in zip(online_weights, target_weights)
        ]
        target_model.set_weights(blended)


def parse_args():
    parser = argparse.ArgumentParser(description="Train The Floor Q-network from replay buffer CSV.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to replay_buffer.csv")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_MODEL_PATH, help="Output model path (.keras preferred)"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--target-update", type=float, default=0.05)
    parser.add_argument(
        "--saved-model-out",
        type=Path,
        default=DEFAULT_SAVED_MODEL_PATH,
        help="TensorFlow SavedModel directory for C++ inference",
    )
    parser.add_argument(
        "--tflite-out",
        type=Path,
        default=DEFAULT_TFLITE_MODEL_PATH,
        help="TensorFlow Lite model file used by C++ inference",
    )
    parser.add_argument(
        "--norm-out",
        type=Path,
        default=DEFAULT_NORM_PATH,
        help="Normalization JSON path used by C++ inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Replay buffer not found at '{args.data}'. Run the C++ game to generate data first."
        )

    states, actions, rewards, dones = load_replay_buffer(args.data)
    print(f"Loaded {len(states)} replay rows from {args.data}")
    states, mean, std = normalize_states(states)
    print("Applied z-normalization to state features")

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
        val_split=args.val_split,
        target_update=args.target_update,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}")

    if args.saved_model_out.exists():
        import shutil

        shutil.rmtree(args.saved_model_out)
    tf.saved_model.save(model, str(args.saved_model_out))
    print(f"Saved TensorFlow SavedModel to {args.saved_model_out}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    args.tflite_out.parent.mkdir(parents=True, exist_ok=True)
    args.tflite_out.write_bytes(tflite_model)
    print(f"Saved TensorFlow Lite model to {args.tflite_out}")

    stats_path = args.norm_out
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_payload = {
        "input_size": INPUT_SIZE,
        "num_actions": MAX_NEIGHBORS,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
    }
    stats_path.write_text(json.dumps(stats_payload), encoding="utf-8")
    print(f"Saved normalization stats to {stats_path}")


if __name__ == "__main__":
    # Set deterministic-ish behavior for easier debugging
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
