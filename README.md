# TheFloor_NN

This project has two parts:

- `TheFloor_NN/Project1`: C++ game simulation for **The Floor**.
- `TheFloor_NN/NN_Training`: Python training script that learns from logged games.

## 1) Generate training data

The C++ game logs transitions to:

- `ml/replay_buffer.csv`

Each row is:

- flattened state (`50 neighbors * 5 features = 250 floats`)
- `action` (neighbor index chosen)
- `reward`
- `done` (`0` or `1`)

Run many games first so the replay buffer has enough variety.

### Replay log location

By default the C++ logger writes to `ml/replay_buffer.csv` relative to your current working directory.

You can force an exact output path by setting env var `THE_FLOOR_REPLAY_PATH` before running the game:

```bash
# Linux/macOS
export THE_FLOOR_REPLAY_PATH=/absolute/path/to/replay_buffer.csv
```

```powershell
# Windows PowerShell
$env:THE_FLOOR_REPLAY_PATH = "C:\path\to\replay_buffer.csv"
```

The logger now prints the full path it writes to on startup so you can confirm it immediately.


## 2) Train the neural net

From repo root:

```bash
python TheFloor_NN/NN_Training/NN_Training.py --data ml/replay_buffer.csv --out model/floor_ai.keras --epochs 20 --batch-size 256
```

The script performs offline DQN-style fitted Q updates with:

- z-normalized input features
- a soft-updated target network
- train/validation loss tracking per epoch

### Useful options

- `--data`: path to replay CSV
- `--out`: model output file (`.keras` recommended)
- `--epochs`: number of passes over data
- `--batch-size`: fit batch size
- `--gamma`: discount factor
- `--val-split`: holdout ratio for validation
- `--target-update`: soft target-network update factor (tau)

## 3) Practical training advice

If you're new to NN/RL, start simple:

1. **Collect data first**: at least tens of thousands of rows.
2. **Reward shaping**: give small positive rewards for good attacks and penalties for bad ones.
3. **Normalize features**: keep features in similar ranges (for example, all in `[0, 1]`).
4. **Train/evaluate loop**:
   - train model
   - run games with model-guided actions
   - append new replay data
   - retrain
5. **Track metrics**: win rate, average score delta, and invalid/poor action rate.

## 4) Next step suggestions

- Split a validation set from replay buffer and monitor validation loss.
- Add a target network for more stable DQN training.

## 5) Let the trained model play the simulator

`ChooseNeighbor()` can call the Python predictor to pick the action instead of always choosing randomly.

### Important: current model paths are fixed in C++

The simulator currently **does not** read environment variables for model inference.  
It uses these hard-coded relative paths from `TheFloor_NN/Project1/Tile.cpp`:

- Predictor script: `NN_Training/predict_action.py`
- Model file: `NN_Training/artifacts/model.keras`
- Normalization file: `NN_Training/artifacts/norm.json` *(optional; only used if present)*

So, before running the simulator, make sure those files exist at exactly those locations relative to the process working directory.

### What each file does

- `model.keras`: trained Keras model that outputs one Q-value per possible action index (`0..49`).
- `norm.json`: normalization statistics with:
  - `feature_mean`: per-feature mean
  - `feature_std`: per-feature std dev
  If this file is missing, inference still runs on raw features.
- `predict_action.py`: reads one flattened state from `stdin`, applies optional normalization, runs the model, and prints one integer action.

### Action-selection behavior

At decision time, the simulator:

1. Flattens the current state to 250 features (`50 neighbors * 5 features`).
2. Calls `predict_action.py --model ... [--norm ...] --valid-count N`.
3. Restricts candidate actions to the first `N` outputs, where `N = current neighbor count`.
4. Uses `argmax` over those valid actions.
5. Falls back to random neighbor choice if prediction fails or returns an out-of-range index.
