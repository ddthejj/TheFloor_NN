import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser(description="Predict best action for The Floor state")
    parser.add_argument("--model", type=Path, required=True, help="Path to .keras model")
    parser.add_argument(
        "--norm",
        type=Path,
        required=False,
        help="Optional path to normalization .json (if omitted, raw features are used).",
    )
    parser.add_argument(
        "--valid-count",
        type=int,
        required=True,
        help="Number of valid neighbor actions (action range is [0, valid-count))",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_state = sys.stdin.read().strip()
    if not raw_state:
        raise ValueError("Expected flattened state values on stdin")

    flat = np.fromstring(raw_state, sep=",", dtype=np.float32)

    if args.norm is not None:
        stats = json.loads(args.norm.read_text(encoding="utf-8"))
        mean = np.array(stats["feature_mean"], dtype=np.float32)
        std = np.array(stats["feature_std"], dtype=np.float32)

        if flat.shape[0] != mean.shape[0]:
            raise ValueError(
                f"State size mismatch. Got {flat.shape[0]} values, expected {mean.shape[0]} from norm file."
            )
        normalized = (flat - mean) / std
    else:
        normalized = flat

    model = keras.models.load_model(args.model)
    q_values = model.predict(normalized.reshape(1, -1), verbose=0)[0]

    valid_count = max(1, min(int(args.valid_count), q_values.shape[0]))
    valid_scores = q_values[:valid_count]
    action = int(np.argmax(valid_scores))

    sys.stdout.write(str(action))


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
