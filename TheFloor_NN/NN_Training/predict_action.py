import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import sys
import time
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
    parser.add_argument(
        "--state",
        type=str,
        required=False,
        help="Optional flattened state string. If omitted, state is read from stdin.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as a long-lived prediction server that polls a request file.",
    )
    parser.add_argument(
        "--request-file",
        type=Path,
        required=False,
        help="Request file used in --serve mode (format: request_id|valid_count|state_csv).",
    )
    parser.add_argument(
        "--response-file",
        type=Path,
        required=False,
        help="Response file used in --serve mode (format: request_id|action).",
    )
    return parser.parse_args()


def load_norm_stats(norm_path: Path | None):
    if norm_path is not None:
        stats = json.loads(norm_path.read_text(encoding="utf-8"))
        mean = np.array(stats["feature_mean"], dtype=np.float32)
        std = np.array(stats["feature_std"], dtype=np.float32)
        return mean, std
    return None, None


def normalize_state(flat: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        return flat
    if flat.shape[0] != mean.shape[0]:
        raise ValueError(
            f"State size mismatch. Got {flat.shape[0]} values, expected {mean.shape[0]} from norm file."
        )
    return (flat - mean) / std


def predict_action(model, flat: np.ndarray, valid_count: int, mean: np.ndarray | None, std: np.ndarray | None) -> int:
    normalized = normalize_state(flat, mean, std)
    q_values = model.predict(normalized.reshape(1, -1), verbose=0)[0]
    bounded_count = max(1, min(int(valid_count), q_values.shape[0]))
    valid_scores = q_values[:bounded_count]
    return int(np.argmax(valid_scores))


def run_single_shot(args, model, mean, std):
    raw_state = args.state.strip() if args.state is not None else sys.stdin.read().strip()
    if not raw_state:
        raise ValueError("Expected flattened state values on stdin")
    flat = np.fromstring(raw_state, sep=",", dtype=np.float32)
    action = predict_action(model, flat, args.valid_count, mean, std)
    sys.stdout.write(str(action))


def run_server(args, model, mean, std):
    if args.request_file is None or args.response_file is None:
        raise ValueError("--serve requires --request-file and --response-file")

    args.response_file.write_text("READY\n", encoding="utf-8")
    last_processed_id = -1

    while True:
        try:
            raw_request = args.request_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raw_request = ""

        if not raw_request:
            time.sleep(0.005)
            continue

        if raw_request == "QUIT":
            return

        request_id_text, valid_count_text, raw_state = raw_request.split("|", 2)
        request_id = int(request_id_text)

        if request_id <= last_processed_id:
            time.sleep(0.002)
            continue

        valid_count = int(valid_count_text)
        flat = np.fromstring(raw_state, sep=",", dtype=np.float32)
        action = predict_action(model, flat, valid_count, mean, std)
        args.response_file.write_text(f"{request_id}|{action}\n", encoding="utf-8")
        last_processed_id = request_id


def main():
    args = parse_args()
    model = keras.models.load_model(args.model)
    mean, std = load_norm_stats(args.norm)

    if args.serve:
        run_server(args, model, mean, std)
        return

    run_single_shot(args, model, mean, std)


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
