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

The simulator can now query a trained model during `ChooseNeighbor()`.

Set these environment variables before running the C++ simulator:

- `THE_FLOOR_MODEL_PATH`: path to the trained `.keras` model.
- `THE_FLOOR_MODEL_NORM_PATH` *(optional but recommended)*: path to the normalization JSON (`.norm.json`) saved during training.

Example:

```bash
export THE_FLOOR_MODEL_PATH=/absolute/path/to/model/floor_ai.keras
export THE_FLOOR_MODEL_NORM_PATH=/absolute/path/to/model/floor_ai.norm.json
```

If only `THE_FLOOR_MODEL_PATH` is set, the simulator still uses model inference (with raw, unnormalized features).

If both variables are set, each move uses `TheFloor_NN/NN_Training/predict_action.py` with normalized features and chooses the best valid neighbor action. If prediction fails, the simulator falls back to random choice.
