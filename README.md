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

The C++ logger writes to `ml/replay_buffer.csv` relative to your current working directory.

## 2) Train the neural net

From repo root:

```bash
python TheFloor_NN/NN_Training/NN_Training.py --data ml/replay_buffer.csv --out model/floor_ai.keras --saved-model-out model/floor_ai_savedmodel --norm-out model/floor_ai.norm.json --epochs 20 --batch-size 256
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
- `--saved-model-out`: SavedModel directory consumed by the C++ runtime
- `--norm-out`: normalization stats path consumed by the C++ runtime

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

## 5) Let the trained model play the simulator (TensorFlow C++ runtime)

The simulator now performs inference directly in C++ with the TensorFlow C API (no Python subprocess).

> Note: training and replay-data loading are still handled in Python (`NN_Training.py`).
> The C++ TensorFlow integration is for runtime inference inside the simulator only.

### Required runtime artifacts

Train with `NN_Training.py` so these are created under `model/`:

- `floor_ai_savedmodel/` (SavedModel directory used by C++)
- `floor_ai.norm.json` (normalization stats, optional but recommended)

### Build configuration (Visual Studio)

`Project1.vcxproj` uses a fixed TensorFlow root under the solution directory:

- headers: `$(SolutionDir)third_party\tensorflow\include`
- libs: `$(SolutionDir)third_party\tensorflow\lib`
- linked library: `tensorflow.lib`

At runtime, place `tensorflow.dll` next to the executable (or in a standard loader path).

### Action-selection behavior

At decision time, the simulator:

1. Flattens the current state to 250 features (`50 neighbors * 5 features`).
2. Runs the SavedModel in-process via TensorFlow C API.
3. Restricts candidate actions to the first `N` outputs, where `N = current neighbor count`.
4. Uses `argmax` over those valid actions.
5. Falls back to random neighbor choice if model loading/inference fails.
