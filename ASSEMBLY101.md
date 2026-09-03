# Assembly101 -> MediaPipe -> Hand Action Dataset

This project now supports an end-to-end Assembly101 pipeline:

```text
official Assembly101 videos + annotations
        |
        v
MediaPipe Hands (21 landmarks)
        |
        v
30 x 66 temporal features
        |
        v
HandActionLSTM
```

Assembly101 fine-grained annotations provide `video`, `start_frame`, `end_frame`, `action_cls`, `verb_cls`, and `noun_cls`. Temporal boundaries are defined at 30 fps.

## Recommended label strategy

For an SOP primitive-action library, use verb labels. This merges object-specific actions into reusable primitives. For example:

```text
pick up bumper
pick up screwdriver
pick up wheel
        -> pick_up
```

Object identity should later come from YOLO / hand-object relation logic.

## Dataset access

The official `assembly-101/assembly101-download-scripts` repository historically downloads recordings from Google Drive. As of May 2026, the maintainers also distribute Assembly101 through the gated Hugging Face dataset `cvml-nus/assembly101` and state that Hugging Face is the long-term distribution path.

This project supports both video sources:

- `--source hf` (default, recommended)
- `--source gdrive` (wraps the official GitHub downloader)

For Hugging Face downloads, the project also supports:

- `--mirror official` (default, `https://huggingface.co`)
- `--mirror cn` (`https://hf-mirror.com`)

The annotation CSV files are downloaded from the official gated Hugging Face Assembly101 release. The public `assembly-101/assembly101-annotations` GitHub repository is useful for annotation-format documentation, but it does not contain the actual `train.csv`, `validation.csv`, and `test.csv` files in Git.

## 1. Install dependencies and MediaPipe model

```bash
pip install -r requirements.txt
python tools/download_models.py
```

First accept the Assembly101 dataset access terms in your Hugging Face account, then either log in with the Hugging Face CLI or set `HF_TOKEN`. This is required for the annotation CSVs as well as Hugging Face video downloads.

## 2. Recommended smoke test

Do this before downloading a large portion of the dataset:

```bash
python tools/assembly101_pipeline.py \
  --source hf \
  --views v8 \
  --labels "pick up,put down,position,screw,unscrew" \
  --max-recordings 1 \
  --limit 5
```

For mainland China, add `--mirror cn`:

```bash
python tools/assembly101_pipeline.py \
  --source hf \
  --mirror cn \
  --views v8 \
  --labels "pick up,put down,position,screw,unscrew" \
  --max-recordings 1 \
  --limit 5
```

Windows PowerShell:

```powershell
python tools/assembly101_pipeline.py `
  --source hf `
  --mirror cn `
  --views v8 `
  --labels "pick up,put down,position,screw,unscrew" `
  --max-recordings 1 `
  --limit 5
```

This command performs:

1. download the official annotation CSVs;
2. find recordings containing the selected verb labels;
3. choose recordings that maximize target-verb coverage in each official split;
4. download fixed RGB view `v8` (`C10404_rgb.mp4`);
5. sample each annotated action interval;
6. run MediaPipe Hands;
7. generate `30 x 66` `.npy` samples.

`--limit N` limits accepted samples per class in each split. Add `--train` if you also want to train the LSTM immediately.

## 3. Full selected-action pipeline

After the smoke test succeeds, remove `--max-recordings` and `--limit`:

```bash
python tools/assembly101_pipeline.py \
  --source hf \
  --views v8 \
  --labels "pick up,put down,position,screw,unscrew" \
  --train
```

In mainland China:

```bash
python tools/assembly101_pipeline.py \
  --source hf \
  --mirror cn \
  --views v8 \
  --labels "pick up,put down,position,screw,unscrew" \
  --train
```

The default `v8` setting intentionally downloads only one fixed RGB camera per recording. Assembly101 has eight fixed RGB views and four egocentric views, so downloading all views is much larger.

Useful view options:

```text
v1 ... v8     one fixed RGB camera
fixed         all 8 fixed RGB cameras
e1 ... e4     one egocentric camera
egocentric    all 4 egocentric cameras
all           all 12 views
```

For the MediaPipe SOP baseline, start with `v8`. Use `fixed` later when you specifically want cross-view robustness.

## 4. Download only

If you want to separate downloading from preprocessing:

```bash
python tools/download_assembly101.py \
  --source hf \
  --mirror cn \
  --views v8 \
  --labels "pick up,screw,unscrew" \
  --max-recordings 5
```

Default raw-data layout:

```text
data/assembly101/
  annotations/
    fine-grained-annotations/
      actions.csv
      train.csv
      validation.csv
      test.csv
  recordings/
    <recording_name>/
      C10404_rgb.mp4
```

Then run:

```bash
python tools/prepare_assembly101.py \
  --video-root data/assembly101/recordings \
  --annotation-dir data/assembly101/annotations/fine-grained-annotations \
  --labels "pick up,screw,unscrew"
```

## 5. Hugging Face source and mirror

The Hugging Face release is gated. You must accept the dataset terms before files can be downloaded. A mirror changes the endpoint only; it does not remove the need for Hugging Face access permission or a token.

You can provide the token through the environment:

```bash
export HF_TOKEN=hf_xxx
python tools/download_assembly101.py --source hf --mirror cn --views v8 --max-recordings 1
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_xxx"
python tools/download_assembly101.py --source hf --mirror cn --views v8 --max-recordings 1
```

`--mirror cn` sets `HF_ENDPOINT=https://hf-mirror.com` inside the downloader before `huggingface_hub` is imported. You therefore do not need to export `HF_ENDPOINT` manually. Use `--mirror official` to force the normal Hugging Face endpoint.

Or pass `--hf-token` directly. Environment variables are preferred so the token does not appear in shell history.

## 6. Google Drive video compatibility

The `gdrive` source automatically clones and invokes the official repository:

```text
https://github.com/assembly-101/assembly101-download-scripts
```

The official downloader requires Google Drive access, `client_secrets.json`, and browser authentication at least once. Annotation CSVs are still obtained from the official Hugging Face release so the rest of this project's directory layout is identical for both video sources.

Example:

```bash
python tools/download_assembly101.py \
  --source gdrive \
  --views v8 \
  --client-secrets /path/to/client_secrets.json \
  --authenticate \
  --resume
```

For a remote/headless machine, generate `credentials.json` using the official authentication flow on a machine with a browser and then pass:

```bash
--credentials /path/to/credentials.json
```

The wrapper preserves the official downloader's view selection and resume behavior while keeping its helper files under:

```text
data/assembly101/_official_downloader/
```

## 7. MediaPipe feature generation

Each accepted annotated segment is processed as:

```text
Assembly101 annotated segment
        |
        v
sample 30 timestamps across the action interval
        |
        v
MediaPipe Hands (21 x XYZ)
        |
        v
interpolate occasional missed-hand frames
        |
        v
wrist-centered + scale-normalized pose
+ wrist displacement
        |
        v
30 x 66 .npy
```

Segments are discarded when MediaPipe detects a hand in fewer than 60% of sampled frames. Change this with `--min-hand-ratio`.

Generated layout:

```text
dataset_assembly101/
  manifest.json
  metadata.csv
  train/
    pick_up/*.npy
    screw/*.npy
    ...
  validation/
    ...
  test/
    ...
```

`metadata.csv` retains the source video, temporal bounds, verb/noun/action IDs, and hand-detection ratio.

## 8. Train

```bash
python tools/train_assembly101.py
```

Checkpoint:

```text
models/hand_action_assembly101_lstm.pth
```

The training script uses the official Assembly101 train/validation/test splits.

## 9. Camera inference

Because both training videos and live input are processed through this project's MediaPipe feature extractor, use normal feature mode:

```bash
python main.py \
  --camera 0 \
  --checkpoint models/hand_action_assembly101_lstm.pth
```

## Important limitations

Assembly101 is licensed CC BY-NC 4.0, so check the license before using the raw dataset for commercial purposes.

Assembly101 annotations are trimmed action intervals, while `main.py` currently classifies a rolling 30-frame camera window. For production SOP recognition, add action-start/action-end segmentation or an `idle/background` class so inference windows resemble training samples.
