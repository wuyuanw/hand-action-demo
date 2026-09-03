# Hand Action Demo

Real-time local-camera demo: **OpenCV → MediaPipe 21-point hand skeleton → normalized 66-D temporal feature → PyTorch LSTM → action + confidence**.

## Actions
`idle`, `grab`, `release`, `move`, `rotate`, `wipe`.

> `grab` in V1 means an open→closed hand-motion pattern, **not proof that an object was physically grasped**. Object-aware semantics require an object detector / hand-object relation stage later.

## Environment
Python 3.10–3.12 recommended. MediaPipe is pinned to `1.0.1` in this demo.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python tools/download_models.py
```

The MediaPipe Tasks HandLandmarker requires a local `.task` model. `download_models.py` fetches the official model into `models/hand_landmarker.task`.

## Collect data

```bash
python tools/collect_data.py --action idle --samples 200
python tools/collect_data.py --action grab --samples 200
python tools/collect_data.py --action release --samples 200
python tools/collect_data.py --action move --samples 200
python tools/collect_data.py --action rotate --samples 200
python tools/collect_data.py --action wipe --samples 200
```

Keys: `SPACE` record/pause, `S` save current 30-frame sample, `R` reset buffer, `Q` quit. Add `--auto` for automatic samples; default stride is 30 frames.

Suggested collection diversity: change position, distance, speed, amplitude and direction. For `wipe`, perform 2–3 reciprocating sweeps; for `move`, keep hand shape relatively stable; for `rotate`, rotate palm/wrist clearly.

## Train

```bash
python tools/inspect_dataset.py
python tools/train.py
```

Best checkpoint: `models/hand_action_lstm.pth`. Training curve: `outputs/training_curve.png`.

## Assembly101 one-command training

For the Assembly101 primitive-action model, the easiest entry point is now:

```bash
python train.py
```

Default actions are `pick up`, `put down`, `position`, `screw`, and `unscrew`. The script automatically:

1. reuses/downloads public `pablovela5620/assembly101-720p` `v8` videos;
2. selects multiple recordings from each official train/validation/test split;
3. extracts MediaPipe Hands features with deterministic mixed-recording sampling;
4. builds balanced train/validation/test samples;
5. trains `HandActionLSTM`, evaluates the best checkpoint, and saves it to `models/hand_action_assembly101_lstm.pth`.

The normal preset uses up to 4 recordings per split with 100 train, 30 validation, and 30 test samples per class. Existing raw videos are never deleted and are reused on later runs. If a previously generated feature dataset already satisfies the target size, `train.py` skips MediaPipe preprocessing and trains directly.

Quick pipeline check:

```bash
python train.py --quick
```

Force a clean feature rebuild while still keeping downloaded videos:

```bash
python train.py --force-rebuild
```

Train exactly the existing `dataset_assembly101` without download/preprocessing:

```bash
python train.py --reuse-dataset
```

Customize the training size when needed:

```bash
python train.py \
  --recordings 6 \
  --train-per-class 150 \
  --validation-per-class 40 \
  --test-per-class 40 \
  --epochs 80
```

## DHG-2016 / DHG-14 baseline

This branch can train the same LSTM from the DHG-14/28 skeleton data. The 14-class labels are:
`grab`, `tap`, `expand`, `pinch`, `rotation_cw`, `rotation_ccw`, `swipe_right`, `swipe_left`, `swipe_up`, `swipe_down`, `swipe_x`, `swipe_v`, `swipe_plus`, `shake`.

Expected raw dataset layout:

```text
DHG14-28_dataset/
  informations_troncage_sequences.txt
  gesture_1/
    finger_1/
      subject_1/
        essai_1/
          skeleton_world.txt
  ...
```

Convert the variable-length 22-joint DHG sequences to the demo's fixed `30 x 66` representation:

```bash
python tools/prepare_dhg2016.py --dhg-root /path/to/DHG14-28_dataset
```

Train with a subject-wise split. By default subject 20 is held out and subjects 1–19 are used for training:

```bash
python tools/train_dhg2016.py --test-subject 20
```

The checkpoint is saved as `models/hand_action_dhg14_lstm.pth`.

Run the DHG-trained checkpoint from the local camera:

```bash
python main.py --camera 0 \
  --checkpoint models/hand_action_dhg14_lstm.pth \
  --feature-mode dhg
```

DHG has 22 hand joints (`wrist + palm + 20 finger joints`) while MediaPipe exposes 21 landmarks without an explicit palm joint. The adapter drops DHG palm joint 1, keeps wrist and the 20 finger joints, linearly resamples the gesture to 30 frames, wrist-centers/scales the local pose, and uses scale-normalized wrist velocity. This feature mode is isolated from the original `legacy` feature mode, so existing checkpoints remain compatible.

For benchmark reporting, DHG-14/28 is normally evaluated with leave-one-subject-out cross-validation. `train_dhg2016.py` implements one held-out subject at a time; run subjects 1–20 if you want the full 20-fold protocol.

> Domain-gap note: DHG skeletons come from Intel RealSense world coordinates while live inference uses MediaPipe landmarks. The adapter makes their topology and scale compatible, but this remains cross-sensor transfer. For production SOP actions, fine-tuning with your own camera skeleton sequences is recommended.

## Run

```bash
python main.py --camera 0
python main.py --camera 0 --debug
```

The model waits until a 30-frame sequence is available. Predictions run every 3 frames and use confidence threshold + recent majority smoothing. Five consecutive no-hand frames clear temporal state.

## Feature definition
Each legacy frame produces 66 float values:
- 63 = 21×XYZ landmarks centered at wrist and scaled by wrist→middle-MCP distance.
- 3 = raw normalized-image wrist displacement `(dx, dy, dz)` versus previous frame.

The extra velocity is intentional: wrist-centering alone removes global motion that is essential for `move` and `wipe`.

## Tests

```bash
pytest -q
```

Unit tests do not require a camera or MediaPipe model file.

## Known limitations
- Skeleton-only `grab` can confuse a plain fist-closing gesture with a true grasp.
- `move`, `wipe`, and `rotate` depend on camera viewpoint and training diversity.
- A 30-frame window adds temporal observation latency.
- V1 tracks one hand only.
- DHG-trained camera inference has a RealSense-to-MediaPipe domain gap; use it as a pretrained baseline, then fine-tune on target-camera data.

## Next upgrade
Add object detection/tracking and hand-object geometric relations so predictions become semantic events such as `GRAB(cloth)` and `WIPE(surface, cloth)`.
